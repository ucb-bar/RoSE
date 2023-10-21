import struct
import numpy as np

CS_REQ_IMG      = 0x10
CS_RSP_IMG      = 0x11
CS_REQ_IMG_POLL = 0x16
CS_RSP_IMG_POLL = 0x17

CS_REQ_DEPTH    = 0x12
CS_RSP_DEPTH    = 0x13
CS_REQ_DEPTH_STREAM = 0x14
CS_RSP_DEPTH_STREAM = 0x15

class Blob:
    # counter = 0
    def __init__(self, latency, packet):
        self.latency = latency
        self.packet = packet
        # self.counter = Blob.counter
        # Blob.counter += 1

    def __eq__(self, other):
        if isinstance(other, Blob):
            return self.latency == other.latency
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Blob):
            return self.latency < other.latency
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, Blob):
            return self.latency > other.latency
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, Blob):
            return self.latency <= other.latency
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, Blob):
            return self.latency >= other.latency
        return NotImplemented

#TODO Remove cmd latency

# class RoSEPacket:
#     cmd_latency_dict = {CS_RSP_IMG: 0.0, CS_RSP_DEPTH: 0.0, CS_RSP_IMG_POLL: 0.0, CS_RSP_DEPTH_STREAM: 0.0}

#     def __init__(self):
#         self.cmd             = None
#         self.num_bytes       = None
#         self.data            = None
#         self.latency_enabled = None
#         self.latency         = None
#         self.dtype           = np.uint8

#     def __str__(self):
#         return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)

#     def init(self, cmd, num_bytes, data, dtype=np.uint8):
#         self.cmd       = cmd
#         self.num_bytes = num_bytes
#         self.data      = data
#         self.latency_enabled = cmd in RoSEPacket.cmd_latency_dict.keys()
#         self.latency   = RoSEPacket.cmd_latency_dict.get(cmd, 0)
#         self.dtype     = dtype

#     def decode(self, buffer):
#         self.cmd       = int.from_bytes(buffer[0:4], "little", signed="False")
#         self.num_bytes = int.from_bytes(buffer[4:8], "little", signed="False")
#         data_array = [] 
#         for i in range(self.num_bytes // 4):    
#             if self.dtype == np.uint8:
#                 data_array.append(int.from_bytes(buffer[4 * i + 8 : 4 * i + 12],  "little", signed="False"))
#             else:
#                 data_array.append(struct.unpack("f", buffer[4 * i + 8 : 4 * i + 12])[0])
#         self.data = data_array

#     def encode(self):
#         if self.latency_enabled:
#         # TODO: refactor the code to get firesim_step
#             buffer = self.cmd.to_bytes(4, 'little') + (round((self.latency)*RoSEPacket.firesim_step).to_bytes(4, 'little') if self.latency > 0 else (0).to_bytes(4, 'little')) + self.num_bytes.to_bytes(4, 'little')
#         else:
#             buffer = self.cmd.to_bytes(4, 'little') + self.num_bytes.to_bytes(4, 'little')
#         if self.num_bytes > 0:
#             for datum in self.data:
#                 if self.dtype == np.uint8:
#                     buffer = buffer + datum.to_bytes(4, "little", signed=False)
#                 else:                
#                     buffer = buffer + struct.pack("f", datum) 
#         print(f"Encoded packet: {buffer}, len: {len(buffer)}")
#         # print("---------------------------------------------------")
#         return buffer

class RoSEPacket:
    cmd_latency_dict = {CS_RSP_IMG: 0.0, CS_RSP_DEPTH: 0.0, CS_RSP_IMG_POLL: 0.0, CS_RSP_DEPTH_STREAM: 0.0}

    def __init__(self):
        self.cmd             = None
        self.num_bytes       = None
        self.data            = None
        self.latency_enabled = None
        self.latency         = None
        self.dtype           = np.uint8

    def __str__(self):
        return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)

    def init(self, cmd, num_bytes, data):
        self.cmd       = cmd
        self.num_bytes = num_bytes
        # print(f"Packet data: {data}, {type(data)}")
        if data is None:
            self.data = np.array([], dtype=np.uint32)
        elif type(data) is list:
            self.data = np.array(data, dtype=np.uint32)
        else:
            self.data = np.frombuffer(data.tobytes(), dtype=np.uint32)
        self.latency_enabled = cmd in RoSEPacket.cmd_latency_dict.keys()
        self.latency   = RoSEPacket.cmd_latency_dict.get(cmd, 0)
        # print(f"Initialized packet: {self}")

    def decode(self, buffer):
        self.cmd       = int.from_bytes(buffer[0:4], "little", signed="False")
        self.num_bytes = int.from_bytes(buffer[4:8], "little", signed="False")
        data_array = []
        for i in range(self.num_bytes // 4):    
            data_array.append(int.from_bytes(buffer[4 * i + 8 : 4 * i + 12],  "little", signed="False"))
        self.data = np.array(data_array, dtype=np.uint32)

    def encode(self):
        if self.latency_enabled:
        # TODO: refactor the code to get firesim_step
            buffer = self.cmd.to_bytes(4, 'little') + (round((self.latency)*RoSEPacket.firesim_step).to_bytes(4, 'little') if self.latency > 0 else (0).to_bytes(4, 'little')) + self.num_bytes.to_bytes(4, 'little')
        else:
            buffer = self.cmd.to_bytes(4, 'little') + self.num_bytes.to_bytes(4, 'little')
        if self.num_bytes > 0:
            buffer =  buffer + self.data.tobytes()
        # print(f"Encoded packet: {buffer}, len: {len(buffer)}")
        # print("---------------------------------------------------")
        return buffer

# class RoSEPacket:
#     cmd_latency_dict = {CS_RSP_IMG: 0.0, CS_RSP_DEPTH: 0.0, CS_RSP_IMG_POLL: 0.0, CS_RSP_DEPTH_STREAM: 0.0}

#     def __init__(self):
#         self.cmd             = None
#         self.num_bytes       = None
#         self.data            = None
#         self.latency_enabled = None
#         self.latency         = None

#     def __str__(self):
#         return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)

#     def init(self, cmd, num_bytes, data=np.array([], dtype=np.uint32)):
#         self.cmd       = cmd
#         self.num_bytes = num_bytes
#         # Convert the data into a np.uint32 array, ensuring byte-exactness
#         if data is not None:
#             self.data = np.frombuffer(data.tobytes(), dtype=np.uint32)
#         else:
#             data = np.array([], dtype=np.uint32)
#         self.latency_enabled = cmd in RoSEPacket.cmd_latency_dict.keys()
#         self.latency   = RoSEPacket.cmd_latency_dict.get(cmd, 0)

#     def decode(self, buffer):
#         self.cmd       = int.from_bytes(buffer[0:4], "little", signed=False)
#         self.num_bytes = int.from_bytes(buffer[4:8], "little", signed=False)
#         data_bytes = buffer[8:8+self.num_bytes]
#         # Convert the byte buffer into a np.uint32 array
#         self.data = np.frombuffer(data_bytes, dtype=np.uint32)

#     def encode(self):
#         buffer = self.cmd.to_bytes(4, 'little')
#         if self.latency_enabled:
#             header += (round((self.latency)*RoSEPacket.firesim_step).to_bytes(4, 'little') if self.latency > 0 else (0).to_bytes(4, 'little'))
#         buffer += self.num_bytes.to_bytes(4, 'little')
        
#         # Convert the np.uint32 data into a byte buffer
#         if self.num_bytes > 0:
#             buffer += self.data.tobytes()
#         print(f"Encoded packet: {buffer}, len: {len(buffer)}")
#         return buffer
