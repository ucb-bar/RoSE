"""Stress test for SocketThread framing under interleaved bidirectional traffic.

Reproduces the RoSE drone_control bug: high-rate interleaved request (0x12, 0 data)
+ control (0x20, 4 words) packets, mimicking the guest, with grant-like gaps between
them. Verifies EVERY packet is received intact (correct cmd + data). The socket uses
the same 0.1 s timeout the synchronizer sets on bridge connections.

The `split` mode injects a gap BETWEEN a data packet's header and its data words --
this is what actually broke drone_control: read_word() lost partial bytes on a
mid-word recv timeout and desynced, so data-carrying packets (0x20) were dropped
while dataless ones (0x12/0x84) still parsed.
"""
import os, sys, socket, struct, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from socket_thread import SocketThread

def send_pkt(sock, cmd, data=()):
    buf = struct.pack('<II', cmd, len(data) * 4)
    for d in data:
        buf += struct.pack('<I', d)
    sock.sendall(buf)

def send_pkt_split(sock, cmd, data, gap):
    """Header and data in separate segments with a mid-packet gap."""
    sock.sendall(struct.pack('<II', cmd, len(data) * 4))
    time.sleep(gap)
    for d in data:
        sock.sendall(struct.pack('<I', d))

def run(N, delay, label, split=False):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('localhost', 0)); srv.listen()
    port = srv.getsockname()[1]
    cli = socket.create_connection(('localhost', port))
    conn, _ = srv.accept()
    conn.settimeout(0.1)                 # same as ServerThread.run()
    st = SocketThread(conn); st.start()

    for i in range(N):
        send_pkt(cli, 0x12)              # request (0 data)
        if delay: time.sleep(delay)
        if split:
            send_pkt_split(cli, 0x20, (i, i, i, i), delay or 0.15)  # gap between hdr+data
        else:
            send_pkt(cli, 0x20, (i, i, i, i))
        if delay: time.sleep(delay)
    time.sleep(2.0)
    st.kill(); st.join(timeout=2); cli.close(); conn.close(); srv.close()

    reqs  = [p for p in st.data_rxqueue if p.cmd == 0x12]
    ctrls = [p for p in st.data_rxqueue if p.cmd == 0x20]
    bad = [c for c in ctrls if list(c.data) != [c.data[0]] * 4] if ctrls else []
    ok = (len(reqs) == N and len(ctrls) == N and not bad)
    print(f"[{label}] sent {N} req + {N} ctrl -> got {len(reqs)} req, {len(ctrls)} ctrl, "
          f"corrupt={len(bad)}  {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    results = [
        run(200, 0.0,  "no-gap        "),
        run(100, 0.15, "gap-between   "),                 # gap between packets
        run(60,  0.0,  "split-hdr-data", split=True),     # gap WITHIN a data packet
    ]
    print("ALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
