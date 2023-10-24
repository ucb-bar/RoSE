import threading
import socket
import struct

from rose_packet import RoSEPacket

# TODO SOLVE THIS
INTCMDS = []

# TODO Stop using queues from the synchronizer class.
# Each socket thread should have its own set of queues
# That the synnchronizer accesses
class SocketThread (threading.Thread):
   def __init__(self, syn):
      threading.Thread.__init__(self)
      self.syn = syn
      self.killed = False

   def read_word(self):
        data = self.sync_conn.recv(1)
        datum = None
        if data:
            for i in range(3):
               while True:
                   datum = self.sync_conn.recv(1)
                   if datum:
                       data = data + datum
                       break
            return data
        return None

   def kill(self):
        self.killed = True

   def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.syn.sync_host, self.syn.sync_port))
            s.listen()
            self.syn.server_started = True
            self.sync_conn, self.sync_addr = s.accept()
            count = 0
            with self.sync_conn:
                print(f"sync_connected by {self.sync_addr}")
                while True:
                    if self.killed:
                        s.close()
                        raise SystemExit()
                    self.sync_conn.settimeout(0.1)
                    count = count + 1
                    if count > 100:
                        # print("Synchronizer heartbeat")
                        count = 0
                    try:
                        cmd_data = self.read_word()
                        if cmd_data:
                            queue = None
                            cmd = int.from_bytes(cmd_data, "little", signed="False")
                            if cmd > 0x80:
                                queue = self.syn.sync_rxqueue
                            else:
                                queue = self.syn.data_rxqueue
                            num_bytes = None
                            while True:
                                # num_bytes_data = self.sync_conn.recv(4)
                                num_bytes_data = self.read_word()
                                if num_bytes_data:
                                    num_bytes = int.from_bytes(num_bytes_data, "little", signed="False")
                                    break
                            data = []
                            for i in range(num_bytes//4):
                                while True:
                                    if num_bytes:
                                        datum = self.read_word()
                                        data.append(int.from_bytes(datum, "little", signed="False"))
                                        break
                            packet = RoSEPacket()
                            packet.init(cmd, num_bytes, data)
                            # print(f"Got packet: {packet}")
                            queue.append(packet)
                    except Exception as e:
                        pass
                    #process the txqueue
                    if len(self.syn.txqueue) > 0:
                        packet = self.syn.txqueue.pop(0)
                        # print(f"Sending packet: {packet}")
                        self.sync_conn.sendall(packet.encode())