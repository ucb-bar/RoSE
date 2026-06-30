#!/usr/bin/env python3
"""Minimal stdlib-only RoSE synchronizer for bridge bring-up testing.

No gymnasium / numpy / opencv. Just enough of the RoSE sync protocol to:
  1) accept the bridge's TCP connection on the sync port (10001),
  2) define a cycle step and continuously GRANT cycle budget so the FireSim
     metasim actually advances (the Rose bridge gates sim advance on budget),
  3) log any packets the SoC sends out (TX path: SoC -> bridge -> host),
  4) optionally answer the camera request with image data routed to channel 2
     so the baremetal airsim-packettest can complete and exit.

Wire format (matches rose_packet.Control_Packet.encode / airsim.cc):
    cmd (4B LE) + num_bytes (4B LE) + data[num_bytes/4] (4B LE each)
"""
import socket
import struct
import threading
import time
import sys

SYNC_PORT = 10001

# Control headers (from rose_packet.h / gym_synchronizer CONTROL_HEADERS)
CS_GRANT_TOKEN = 0x80
CS_REQ_CYCLES  = 0x81
CS_DEFINE_STEP = 0x83
CS_CFG_BW      = 0x85
CS_CFG_ROUTE   = 0x86
# Payload headers
CS_CAMERA_LEFT = 0x11

STEP = 100000          # cycles granted per token (CS_DEFINE_STEP)
import os
IMG_WORDS = int(os.environ.get("RESP_WORDS", 8))  # words to answer with (rxsmall test wants 8)
RESP_CHANNEL = 2       # reqrsp1 (where the test reads RX)

def encode(cmd, data=None):
    data = data or []
    buf = struct.pack("<I", cmd) + struct.pack("<I", len(data) * 4)
    for d in data:
        buf += struct.pack("<I", d & 0xFFFFFFFF)
    return buf

def main():
    respond = "--respond" in sys.argv
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", SYNC_PORT))
    s.listen()
    print(f"[minimal_sync] listening on :{SYNC_PORT} (respond={respond})", flush=True)
    conn, addr = s.accept()
    print(f"[minimal_sync] bridge connected from {addr}", flush=True)

    # 1) define the cycle step
    conn.sendall(encode(CS_DEFINE_STEP, [STEP]))
    # 2) configure per-channel bandwidth. The bridge rxcontroller gates RX delivery on a
    #    bandwidth_threshold that defaults to 0 (=> blocked); set a large value to allow it.
    BW = 1 << 20
    for ch in range(3):  # DMA0, reqrsp0, reqrsp1
        conn.sendall(encode(CS_CFG_BW, [ch, BW]))
    # 3) route the camera response header -> channel 2 (so RX lands where the test reads)
    conn.sendall(encode(CS_CFG_ROUTE, [CS_CAMERA_LEFT, RESP_CHANNEL]))
    print(f"[minimal_sync] sent CS_DEFINE_STEP + CS_CFG_BW(x3, bw={BW}) + CS_CFG_ROUTE", flush=True)

    stop = threading.Event()

    # 3) continuously grant cycle budget so the sim advances
    def granter():
        n = 0
        while not stop.is_set():
            try:
                conn.sendall(encode(CS_GRANT_TOKEN))
                n += 1
                if n % 2000 == 0:
                    print(f"[minimal_sync] granted {n} tokens", flush=True)
            except OSError:
                break
            time.sleep(0.0001)
    threading.Thread(target=granter, daemon=True).start()

    # 4) read SoC->host packets; log them; optionally answer the camera request
    conn.settimeout(1.0)
    def recvword():
        b = b""
        while len(b) < 4:
            chunk = conn.recv(4 - len(b))
            if not chunk:
                return None
            b += chunk
        return struct.unpack("<I", b)[0]

    seen = 0
    statuses = {}          # status value -> count (from 0xBB status-report packets)
    responded = False
    start = time.time()
    while time.time() - start < 60:
        try:
            cmd = recvword()
            if cmd is None:
                print("[minimal_sync] connection closed by bridge", flush=True)
                break
            num_bytes = recvword()
            data = [recvword() for _ in range(num_bytes // 4)]
            seen += 1
            c = cmd & 0xFF
            if c == 0xBB:   # status report from the SoC
                st = data[0] if data else None
                if st not in statuses:
                    statuses[st] = 0
                    deq2 = "  <<< DEQ_VALID_2 (0x2) SET -> RX delivered to ch2!" if (st is not None and (st & 0x2)) else ""
                    print(f"[minimal_sync] STATUS report: 0x{st:x}{deq2}", flush=True)
                statuses[st] += 1
            else:
                print(f"[minimal_sync] RX packet #{seen}: cmd=0x{cmd:02x} num_bytes={num_bytes} data={[hex(d) for d in data[:12]]}", flush=True)
            if respond and not responded and c == CS_CAMERA_LEFT:
                payload = [0xD00D0000 + i for i in range(IMG_WORDS)]
                print(f"[minimal_sync] -> answering camera request with {IMG_WORDS} words on ch{RESP_CHANNEL}", flush=True)
                conn.sendall(encode(CS_CAMERA_LEFT, payload))
                responded = True
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[minimal_sync] socket error: {e}", flush=True)
            break

    stop.set()
    conn.close()
    s.close()
    print(f"[minimal_sync] done; saw {seen} packets from the SoC", flush=True)

if __name__ == "__main__":
    main()
