import numpy as np

class Packet:
    # public class attribute
    cmd_latency_dict = {}

    def __init__(self, cmd, num_bytes, data):
        self.cmd = cmd
        self.num_bytes = num_bytes
        if data is None:
            self.data = np.array([], dtype=np.uint32)
        elif type(data) is list:
            self.data = np.array(data, dtype=np.uint32)
        else:
            self.data = np.frombuffer(data.tobytes(), dtype=np.uint32)
        self.dtype = np.uint8

    def __str__(self):
        return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)
    
    def __eq__(self, other):
        if isinstance(other, Packet):
            return self.latency == other.latency
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Packet):
            return self.latency < other.latency
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, Packet):
            return self.latency > other.latency
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, Packet):
            return self.latency <= other.latency
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, Packet):
            return self.latency >= other.latency
        return NotImplemented

    def decode(self, buffer):
        self.cmd = int.from_bytes(buffer[0:4], "little", signed="False")
        self.num_bytes = int.from_bytes(buffer[4:8], "little", signed="False")
        self.data = np.array([int.from_bytes(buffer[4*(i+2) : 4*(i+3)], "little", signed="False") for i in range(self.num_bytes//4)], dtype=np.uint32)

    def encode(self):
        return NotImplemented

class Control_Packet(Packet):
    def __init__(self, cmd, num_bytes, data):
        super().__init__(cmd, num_bytes, data)
        self.latency = 0
    
    def encode(self):
        buffer = self.cmd.to_bytes(4, 'little') + self.num_bytes.to_bytes(4, 'little')
        return buffer + self.data.tobytes() if self.num_bytes > 0 else buffer

class Payload_Packet(Packet):
    def __init__(self, cmd, num_bytes, data):
        super().__init__(cmd, num_bytes, data)
        self.latency = Packet.cmd_latency_dict.get(cmd, 0)
    
    def encode(self):
        buffer = self.cmd.to_bytes(4, 'little') + (round((self.latency)*Packet.firesim_step).to_bytes(4, 'little') if self.latency > 0 else (0).to_bytes(4, 'little')) + self.num_bytes.to_bytes(4, 'little')
        return buffer + self.data.tobytes() if self.num_bytes > 0 else buffer