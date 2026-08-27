#!/usr/bin/env python3
"""
Decode a single packet from a trace stream.
Usage: python3 decode_one.py <hex_byte> [additional_bytes...]
"""

import sys
from typing import Tuple, List


# Constants from Rust code
C_HEADER_MASK = 0b0000_0011
C_TIMESTAMP_MASK = 0b1111_1100
F_HEADER_MASK = 0b0001_1100
FHEADER_OFFSET = 2
SYNC_TYPE_MASK = 0b1110_0000
SYNC_TYPE_OFFSET = 5
TRAP_TYPE_MASK = 0b1110_0000
TRAP_TYPE_OFFSET = 5
BP_MODE_MASK = 0b11
BP_ENTRY_MASK = 0b1111_1100
BP_ENTRY_OFFSET = 2
BP_BASE_VALUE = 64

VAR_MASK = 0b1000_0000
VAR_LAST = 0b1000_0000
VAR_OFFSET = 7
VAR_VAL_MASK = 0b0111_1111


class CHeader:
    CTb = 0b00  # taken branch
    CNt = 0b01  # not taken branch
    CNa = 0b10  # not applicable
    CIj = 0b11  # inferable jump

    @staticmethod
    def to_string(value: int) -> str:
        if value == CHeader.CTb:
            return "CTb (taken branch)"
        elif value == CHeader.CNt:
            return "CNt (not taken branch)"
        elif value == CHeader.CNa:
            return "CNa (not applicable)"
        elif value == CHeader.CIj:
            return "CIj (inferable jump)"
        else:
            return f"Unknown CHeader ({value:02b})"


class FHeader:
    FTb = 0b000   # taken branch
    FNt = 0b001   # non taken branch
    FUj = 0b010   # uninferable jump
    FIj = 0b011   # inferable jump
    FTrap = 0b100 # trapping happened
    FSync = 0b101 # synchronization packet
    FRes1 = 0b110 # context change
    FRes2 = 0b111 # reserved

    @staticmethod
    def to_string(value: int) -> str:
        names = {
            FHeader.FTb: "FTb (taken branch)",
            FHeader.FNt: "FNt (not taken branch)",
            FHeader.FUj: "FUj (uninferable jump)",
            FHeader.FIj: "FIj (inferable jump)",
            FHeader.FTrap: "FTrap (trap/interrupt/exception)",
            FHeader.FSync: "FSync (synchronization)",
            FHeader.FRes1: "FRes1 (context change)",
            FHeader.FRes2: "FRes2 (reserved)",
        }
        return names.get(value, f"Unknown FHeader ({value:03b})")


class SyncType:
    SyncNone = 0b000
    SyncStart = 0b001
    SyncPeriodic = 0b010
    SyncEnd = 0b011

    @staticmethod
    def to_string(value: int) -> str:
        names = {
            SyncType.SyncNone: "SyncNone",
            SyncType.SyncStart: "SyncStart",
            SyncType.SyncPeriodic: "SyncPeriodic",
            SyncType.SyncEnd: "SyncEnd",
        }
        return names.get(value, f"Unknown SyncType ({value:03b})")


class TrapType:
    TNone = 0b000
    TException = 0b001
    TInterrupt = 0b010
    TReturn = 0b100

    @staticmethod
    def to_string(value: int) -> str:
        names = {
            TrapType.TNone: "TNone",
            TrapType.TException: "TException",
            TrapType.TInterrupt: "TInterrupt",
            TrapType.TReturn: "TReturn",
        }
        return names.get(value, f"Unknown TrapType ({value:03b})")


class Prv:
    PrvUser = 0b000
    PrvSupervisor = 0b001
    PrvHypervisor = 0b010
    PrvMachine = 0b011

    @staticmethod
    def to_string(value: int) -> str:
        names = {
            Prv.PrvUser: "User",
            Prv.PrvSupervisor: "Supervisor",
            Prv.PrvHypervisor: "Hypervisor",
            Prv.PrvMachine: "Machine",
        }
        return names.get(value, f"Unknown Prv ({value:03b})")


class BrMode:
    BrTarget = 0b00
    BrHistory = 0b01
    BrPredict = 0b10
    BrReserved = 0b11

    @staticmethod
    def to_string(value: int) -> str:
        names = {
            BrMode.BrTarget: "BrTarget",
            BrMode.BrHistory: "BrHistory",
            BrMode.BrPredict: "BrPredict",
            BrMode.BrReserved: "BrReserved",
        }
        return names.get(value, f"Unknown BrMode ({value:02b})")


def read_varint(data: List[int], offset: int) -> Tuple[int, int, bool]:
    """Read a varint from the data starting at offset. Returns (value, bytes_read, is_complete).
    is_complete is False if the varint was truncated (last byte doesn't have MSB set)."""
    if offset >= len(data):
        raise ValueError(f"Insufficient data for varint at offset {offset}")
    
    scratch = []
    count = 0
    is_complete = False
    while offset + count < len(data):
        byte = data[offset + count]
        scratch.append(byte)
        count += 1
        if byte & VAR_MASK == VAR_LAST:
            is_complete = True
            break
        if count >= 10:
            raise ValueError("Varint exceeded maximum length")
    
    # Check if we ran out of data before finding the last byte
    if not is_complete and offset + count >= len(data):
        # We've read all available bytes but the last one doesn't have MSB set
        is_complete = False
    
    if offset + count > len(data):
        raise ValueError(f"Insufficient data for varint: need {count} bytes, have {len(data) - offset}")
    
    value = 0
    for byte in reversed(scratch):
        value = (value << VAR_OFFSET) | (byte & VAR_VAL_MASK)
    
    return (value, count, is_complete)


def read_prv(data: List[int], offset: int) -> Tuple[int, int, int]:
    """Read privilege levels from a byte. Returns (from_prv, target_prv, bytes_read)."""
    if offset >= len(data):
        raise ValueError(f"Insufficient data for prv at offset {offset}")
    
    result = data[offset]
    from_prv = result & 0b111
    target_prv = (result >> 3) & 0b111
    checksum = (result >> 6) & 0b11
    
    if checksum != 0b10:
        print(f"Warning: prv checksum should be 0b10, got {checksum:02b}")
    
    return (from_prv, target_prv, 1)


def decode_packet(data: List[int]) -> dict:
    """Decode a single packet from byte data. Returns a dict with packet information."""
    if len(data) == 0:
        raise ValueError("No data provided")
    
    first_byte = data[0]
    result = {
        "first_byte": f"0x{first_byte:02x} ({first_byte:08b})",
        "is_compressed": False,
        "bytes_read": 1,
        "fields": {}
    }
    
    c_header = first_byte & C_HEADER_MASK
    
    # Check if compressed
    if c_header in [CHeader.CTb, CHeader.CNt, CHeader.CIj]:
        result["is_compressed"] = True
        result["c_header"] = CHeader.to_string(c_header)
        
        # Map CHeader to FHeader
        if c_header == CHeader.CTb:
            f_header_str = "FTb (taken branch)"
        elif c_header == CHeader.CNt:
            f_header_str = "FNt (not taken branch)"
        else:  # CIj
            f_header_str = "FIj (inferable jump)"
        
        result["f_header"] = f_header_str
        timestamp = (first_byte & C_TIMESTAMP_MASK) >> 2
        result["fields"]["timestamp"] = timestamp
        result["description"] = f"Compressed packet: {result['c_header']}, timestamp={timestamp}"
        
    elif c_header == CHeader.CNa:
        # Non-compressed packet
        result["is_compressed"] = False
        result["c_header"] = CHeader.to_string(c_header)
        
        f_header = (first_byte & F_HEADER_MASK) >> FHEADER_OFFSET
        result["f_header"] = FHeader.to_string(f_header)
        
        offset = 1
        
        if f_header in [FHeader.FTb, FHeader.FNt, FHeader.FIj]:
            # Read timestamp varint
            try:
                timestamp, count, is_complete = read_varint(data, offset)
                result["fields"]["timestamp"] = timestamp
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Timestamp varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                result["description"] = f"Non-compressed packet: {result['f_header']}, timestamp={timestamp}"
            except (ValueError, IndexError) as e:
                result["error"] = f"Insufficient data for timestamp varint: {e}"
                result["description"] = f"Non-compressed packet: {result['f_header']} (incomplete - need more bytes)"
        
        elif f_header == FHeader.FUj:
            # Read target_address and timestamp varints
            try:
                target_address, count, is_complete = read_varint(data, offset)
                result["fields"]["target_address"] = target_address
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Target address varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                timestamp, count, is_complete = read_varint(data, offset)
                result["fields"]["timestamp"] = timestamp
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Timestamp varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                result["description"] = f"Non-compressed packet: {result['f_header']}, target_address=0x{target_address:x}, timestamp={timestamp}"
            except (ValueError, IndexError) as e:
                result["error"] = f"Insufficient data for FUj packet: {e}"
                result["description"] = f"Non-compressed packet: {result['f_header']} (incomplete - need more bytes)"
        
        elif f_header == FHeader.FSync:
            # Read sync_type, prv, target_ctx, target_address, timestamp
            sync_type = (first_byte & SYNC_TYPE_MASK) >> SYNC_TYPE_OFFSET
            result["fields"]["sync_type"] = SyncType.to_string(sync_type)
            
            try:
                from_prv, target_prv, count = read_prv(data, offset)
                result["fields"]["from_prv"] = Prv.to_string(from_prv)
                result["fields"]["target_prv"] = Prv.to_string(target_prv)
                result["bytes_read"] += count
                offset += count
                
                target_ctx, count, is_complete = read_varint(data, offset)
                result["fields"]["target_ctx"] = target_ctx
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Target context varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                target_address, count, is_complete = read_varint(data, offset)
                result["fields"]["target_address"] = target_address
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Target address varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                timestamp, count, is_complete = read_varint(data, offset)
                result["fields"]["timestamp"] = timestamp
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Timestamp varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                result["description"] = (
                    f"Non-compressed packet: {result['f_header']}, "
                    f"sync_type={result['fields']['sync_type']}, "
                    f"from_prv={result['fields']['from_prv']}, "
                    f"target_prv={result['fields']['target_prv']}, "
                    f"target_ctx={target_ctx}, "
                    f"target_address=0x{target_address:x}, "
                    f"timestamp={timestamp}"
                )
            except (ValueError, IndexError) as e:
                result["error"] = f"Insufficient data for FSync packet: {e}"
                result["description"] = f"Non-compressed packet: {result['f_header']} (incomplete - need more bytes)"
        
        elif f_header == FHeader.FTrap:
            # Read trap_type, prv, from_address, target_address, timestamp
            trap_type = (first_byte & TRAP_TYPE_MASK) >> TRAP_TYPE_OFFSET
            result["fields"]["trap_type"] = TrapType.to_string(trap_type)
            
            try:
                from_prv, target_prv, count = read_prv(data, offset)
                result["fields"]["from_prv"] = Prv.to_string(from_prv)
                result["fields"]["target_prv"] = Prv.to_string(target_prv)
                result["bytes_read"] += count
                offset += count
                
                # TReturn with target_prv == PrvUser needs target_ctx
                if trap_type == TrapType.TReturn and target_prv == Prv.PrvUser:
                    target_ctx, count, is_complete = read_varint(data, offset)
                    result["fields"]["target_ctx"] = target_ctx
                    result["bytes_read"] += count
                    offset += count
                    if not is_complete:
                        result["warnings"] = result.get("warnings", [])
                        result["warnings"].append(f"Target context varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                from_address, count, is_complete = read_varint(data, offset)
                result["fields"]["from_address"] = from_address
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"From address varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                target_address, count, is_complete = read_varint(data, offset)
                result["fields"]["target_address"] = target_address
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Target address varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                timestamp, count, is_complete = read_varint(data, offset)
                result["fields"]["timestamp"] = timestamp
                result["bytes_read"] += count
                offset += count
                if not is_complete:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(f"Timestamp varint is incomplete (last byte at offset {offset-1} doesn't have MSB set)")
                
                result["description"] = (
                    f"Non-compressed packet: {result['f_header']}, "
                    f"trap_type={result['fields']['trap_type']}, "
                    f"from_prv={result['fields']['from_prv']}, "
                    f"target_prv={result['fields']['target_prv']}, "
                    f"from_address=0x{from_address:x}, "
                    f"target_address=0x{target_address:x}, "
                    f"timestamp={timestamp}"
                )
            except (ValueError, IndexError) as e:
                result["error"] = f"Insufficient data for FTrap packet: {e}"
                result["description"] = f"Non-compressed packet: {result['f_header']} (incomplete - need more bytes)"
        
        else:
            result["description"] = f"Non-compressed packet: {result['f_header']} (unhandled type)"
    
    else:
        result["error"] = f"Invalid CHeader: {c_header:02b}"
        result["description"] = f"Invalid packet: unknown CHeader {c_header:02b}"
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 decode_one.py <hex_byte> [additional_bytes...]")
        print("Example: python3 decode_one.py 80")
        print("Example: python3 decode_one.py 80 12 34 56")
        sys.exit(1)
    
    # Parse hex bytes from command line
    data = []
    for arg in sys.argv[1:]:
        try:
            # Support both "80" and "0x80" formats
            if arg.startswith("0x") or arg.startswith("0X"):
                byte_val = int(arg, 16)
            else:
                byte_val = int(arg, 16)
            
            if byte_val < 0 or byte_val > 255:
                print(f"Error: {arg} is not a valid byte (0-255)")
                sys.exit(1)
            
            data.append(byte_val)
        except ValueError:
            print(f"Error: '{arg}' is not a valid hex byte")
            sys.exit(1)
    
    if len(data) == 0:
        print("Error: No valid bytes provided")
        sys.exit(1)
    
    try:
        packet = decode_packet(data)
        
        print("=" * 60)
        print("PACKET DECODING")
        print("=" * 60)
        print(f"First byte: {packet['first_byte']}")
        print(f"Compressed: {packet['is_compressed']}")
        print(f"C Header: {packet.get('c_header', 'N/A')}")
        print(f"F Header: {packet.get('f_header', 'N/A')}")
        
        if 'error' in packet:
            print(f"\nERROR: {packet['error']}")
        
        if 'warnings' in packet:
            print("\nWARNINGS:")
            for warning in packet['warnings']:
                print(f"  ⚠ {warning}")
        
        if packet['fields']:
            print("\nFields:")
            for key, value in packet['fields'].items():
                print(f"  {key}: {value}")
        
        print(f"\nBytes read: {packet['bytes_read']} (out of {len(data)} provided)")
        
        print("\n" + "=" * 60)
        print("DESCRIPTION:")
        print(packet['description'])
        print("=" * 60)
        
        if packet['bytes_read'] < len(data):
            print(f"\nNote: {len(data) - packet['bytes_read']} extra byte(s) provided but not used")
        
    except Exception as e:
        print(f"Error decoding packet: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

