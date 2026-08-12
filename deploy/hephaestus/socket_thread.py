import threading
import socket
import struct
import os

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
            # Small idle recv timeout so the SocketThread loop re-checks its txqueue
            # promptly. With the old 0.1 s a grant appended while the thread was blocked
            # in recv() waited out the FULL 100 ms before transmission -- ~105 ms/grant,
            # the dominant co-sim sync-barrier cost (seam decomposition, task #98).
            sync_conn.settimeout(float(os.environ.get("ROSE_SYNC_RECV_TIMEOUT", "0.001")))
            socket_thread = SocketThread(sync_conn)
            self.syn.nodes.append(socket_thread)
            self.connected_sockets += 1
            # socket_thread.start()
        with self.pc:
            print("Entered server pc waiting")
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
        self.stream_txqueue = {}
        self.data_rxqueue = []
        self.sync_rxqueue = []
        self.killed = False
        # Set once the bridge (spike) closes the socket, so the synchronizer's
        # ack busy-waits can stop instead of spinning a core forever on a dead peer.
        self.disconnected = False
        # Byte-accumulating RX buffer. Packets are framed as
        #   [cmd:u32][num_bytes:u32][data: num_bytes] (all little-endian).
        # We must NEVER lose partially-read bytes on a recv timeout (the old
        # byte-at-a-time read_word() did, which desynced the stream and silently
        # dropped every data-carrying packet, e.g. drone_control controls).
        self._rxbuf = bytearray()

   def _fill(self):
        """Pull whatever bytes are available into the RX buffer (one timed recv).
        Returns False iff the connection was closed."""
        try:
            chunk = self.sync_conn.recv(4096)
            if not chunk:
                return False          # orderly shutdown
            self._rxbuf.extend(chunk)
        except socket.timeout:
            pass                      # no data this window; caller still services txqueue
        except Exception:
            pass
        return True

   def _parse_packets(self):
        """Parse every COMPLETE packet currently buffered (partial packets stay
        buffered until the rest arrives -- no desync, no byte loss)."""
        while len(self._rxbuf) >= 8:
            cmd = int.from_bytes(self._rxbuf[0:4], "little")
            num_bytes = int.from_bytes(self._rxbuf[4:8], "little")
            if len(self._rxbuf) < 8 + num_bytes:
                break                 # rest of this packet hasn't arrived yet
            data = [int.from_bytes(self._rxbuf[8 + 4*i : 12 + 4*i], "little")
                    for i in range(num_bytes // 4)]
            del self._rxbuf[:8 + num_bytes]
            packet = Control_Packet(cmd, num_bytes, data) if (cmd > 0x80) else Payload_Packet(cmd, num_bytes, data)
            (self.sync_rxqueue if cmd > 0x80 else self.data_rxqueue).append(packet)
            if os.environ.get("ROSE_RX_DEBUG"):
                self._rxc = getattr(self, "_rxc", {})
                self._rxc[cmd] = self._rxc.get(cmd, 0) + 1
                if sum(self._rxc.values()) % 25 == 0:
                    print("RX counts: " + " ".join(f"0x{k:x}={v}" for k, v in sorted(self._rxc.items())), flush=True)

   def kill(self):
        self.killed = True

   def run(self):
        while not self.killed:
            # Service the txqueue FIRST so a queued grant/response is transmitted
            # immediately, rather than after the recv() timeout in _fill(). TX and RX
            # are independent directions; the old order (recv-then-send) added up to
            # the full recv timeout of latency PER GRANT, which was ~105 ms/grant --
            # the single largest term in the co-sim sync barrier (task #98).
            while len(self.txqueue) > 0:
                packet = self.txqueue.pop(0)
                self.sync_conn.sendall(packet.encode())
            if not self._fill():
                self.disconnected = True   # peer (bridge) closed the socket
                break
            self._parse_packets()
        self.sync_conn.close()