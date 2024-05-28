import threading
import socket
import struct

from rose_packet import *

# This thread set and listens to socket connections
# Will be joined by main after all nodes are connected
class ServerThread (threading.Thread):
    def __init__(self, syn, pc):
        threading.Thread.__init__(self) 
        self.syn = syn
        self.connected_sockets = 0
        self.num_sockets = syn.n_fsim_nodes
        self.pc = pc

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.syn.sync_host, self.syn.sync_port))
        s.listen()
        while self.connected_sockets < self.num_sockets:
            sync_conn, sync_addr = s.accept()
            sync_conn.settimeout(0.1)
            socket_thread = SocketThread(sync_conn)
            self.syn.nodes.append(socket_thread)
            self.connected_sockets += 1
            # socket_thread.start()
        with self.pc:
            self.pc.wait()
        for node in self.syn.nodes:
            node.kill()
            node.join()
        s.close()

# TODO Stop using queues from the synchronizer class.
# Each socket thread should have its own set of queues
# That the synnchronizer accesses
class SocketThread (threading.Thread):
   def __init__(self, sync_conn):
        threading.Thread.__init__(self)
        self.sync_conn = sync_conn
        self.txqueue = []
        self.txpq = []
        self.data_rxqueue = []
        self.sync_rxqueue = [] 
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
        while not self.killed:
            try:
                cmd_data = self.read_word()
                if cmd_data:
                    cmd = int.from_bytes(cmd_data, "little", signed="False")
                    target_queue = self.sync_rxqueue if cmd > 0x80 else self.data_rxqueue
                    num_bytes = None
                    while True:
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
                    packet = Control_Packet(cmd, num_bytes, data) if (cmd > 0x80) else Payload_Packet(cmd, num_bytes, data)
                    target_queue.append(packet)
            except Exception as e:
                pass
            #process the txqueue
            if len(self.txqueue) > 0:
                packet = self.txqueue.pop(0)
                # print(f"Sending packet: {packet}")
                self.sync_conn.sendall(packet.encode())
        self.sync_conn.close()