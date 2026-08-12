#!/usr/bin/env python3
"""Post-fix validation launcher (task #98). Unlike seam_instr_run.py, this does NOT
override SocketThread.run() -- it exercises the REAL tracked run() (the applied fix:
txqueue-before-_fill + ROSE_SYNC_RECV_TIMEOUT). Timestamps are taken non-invasively:

  A t_queued   : Synchronizer.grant_firesim_token appended CS_GRANT_TOKEN
  B t_sent     : Control_Packet.encode() called for cmd==0x80 (run() calls this
                 immediately before sync_conn.sendall -> == the on-wire moment)
  C t_rsp_recv : SocketThread._parse_packets appended a CS_RSP_STALL to sync_rxqueue
  D t_rsp_seen : Synchronizer.check_token_exhaustion popped the RSP

Also records data-packet RX (reqrsp cmd 0x11/0x12 etc.) so correctness is visible.
"""
import os, sys, time, threading, argparse

HEPH = os.environ.get("ROSE_HEPH_DIR") or os.path.join(
    os.environ.get("ROSE_ROOT", ""), "deploy", "hephaestus")
if not os.path.isdir(HEPH):
    sys.exit(f"[seam2] hephaestus dir not found: {HEPH!r} "
             "(set ROSE_ROOT or ROSE_HEPH_DIR)")
sys.path.insert(0, HEPH)

import gym_synchronizer as GS
import socket_thread as ST
import rose_packet as RP
from socket_thread import ServerThread

CS_GRANT_TOKEN = 0x80
CS_RSP_STALL   = 0x84

LOG = {"A": [], "B": [], "C": [], "D": []}
DATA_RX = {}   # cmd -> count of data (payload) packets the sync received from the guest

# ---- A + D : Synchronizer (main thread) ----
_orig_grant = GS.Synchronizer.grant_firesim_token
def grant_firesim_token(self, t):
    LOG["A"].append(time.perf_counter())
    return _orig_grant(self, t)
GS.Synchronizer.grant_firesim_token = grant_firesim_token

_orig_check = GS.Synchronizer.check_token_exhaustion
def check_token_exhaustion(self, t):
    r = _orig_check(self, t)
    LOG["D"].append(time.perf_counter())
    return r
GS.Synchronizer.check_token_exhaustion = check_token_exhaustion

# ---- B : Control_Packet.encode() for the grant (called inside the real run()) ----
_orig_encode = RP.Control_Packet.encode
def encode(self):
    if self.cmd == CS_GRANT_TOKEN:
        LOG["B"].append(time.perf_counter())
    return _orig_encode(self)
RP.Control_Packet.encode = encode

# ---- C : detect RSP_STALL BEFORE the real parse consumes it.
# Race-free: _rxbuf is owned solely by the socket thread (the main thread only pops
# from sync_rxqueue). We scan the complete packets currently in _rxbuf, timestamp any
# RSP_STALL, count data packets, THEN call the real _parse_packets. Reading
# sync_rxqueue *after* parse loses entries the busy-spinning main thread already popped.
_orig_parse = ST.SocketThread._parse_packets
def _parse_packets(self):
    buf = self._rxbuf
    off = 0
    while len(buf) - off >= 8:
        cmd = int.from_bytes(buf[off:off+4], "little")
        num_bytes = int.from_bytes(buf[off+4:off+8], "little")
        if len(buf) - off < 8 + num_bytes:
            break
        if cmd == CS_RSP_STALL:
            LOG["C"].append(time.perf_counter())
        elif cmd <= 0x80:
            DATA_RX[cmd] = DATA_RX.get(cmd, 0) + 1
        off += 8 + num_bytes
    _orig_parse(self)
ST.SocketThread._parse_packets = _parse_packets


def dump(path, nwarmup=3):
    n = min(len(LOG["A"]), len(LOG["B"]), len(LOG["C"]), len(LOG["D"]))
    with open(path, "w") as f:
        f.write("grant,A_queued,B_sent,C_rsp_recv,D_rsp_seen,"
                "send_ms(B-A),rtt_ms(C-B),pickup_ms(D-C),total_ms(D-A)\n")
        for i in range(n):
            A, B, C, D = LOG["A"][i], LOG["B"][i], LOG["C"][i], LOG["D"][i]
            f.write(f"{i},{A:.6f},{B:.6f},{C:.6f},{D:.6f},"
                    f"{(B-A)*1e3:.3f},{(C-B)*1e3:.3f},{(D-C)*1e3:.3f},{(D-A)*1e3:.3f}\n")
    def stats(kp):
        vals = sorted((LOG[kp[1]][i]-LOG[kp[0]][i])*1e3 for i in range(nwarmup, n))
        if not vals: return (0,0,0,0)
        return (vals[0], vals[len(vals)//2], sum(vals)/len(vals), vals[-1])
    to = os.environ.get("ROSE_SYNC_RECV_TIMEOUT", "0.001")
    print(f"\n==== POST-FIX SEAM DECOMPOSITION (n={n} grants, {nwarmup} warmup skipped, "
          f"ROSE_SYNC_RECV_TIMEOUT={to}s) ====", flush=True)
    print("component            min      median     mean      max   (ms)")
    for label, kp in [("send  (B-A)", ("A","B")), ("rtt   (C-B)", ("B","C")),
                      ("pickup(D-C)", ("C","D")), ("TOTAL (D-A)", ("A","D"))]:
        mn, md, me, mx = stats(kp)
        print(f"  {label}   {mn:8.2f} {md:8.2f} {me:8.2f} {mx:8.2f}", flush=True)
    print(f"[correctness] data (payload) packets RX from guest: "
          f"{ {hex(k): v for k, v in sorted(DATA_RX.items())} }", flush=True)
    print(f"csv -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_path", default=None)
    ap.add_argument("--max_grants", type=int, default=int(os.environ.get("SEAM_MAX_GRANTS", "60")))
    args = ap.parse_args()
    out = os.environ.get("SEAM_OUT", "/tmp/seam_postfix.csv")

    sync = GS.Synchronizer(yaml_path=args.yaml_path)
    print(f"[seam2] env={sync.env.spec.id} firesim_step={sync.firesim_step} "
          f"freq={sync.firesim_freq} recv_timeout={os.environ.get('ROSE_SYNC_RECV_TIMEOUT','0.001')}s",
          flush=True)
    condition = threading.Condition()
    server = ServerThread(sync, condition)
    server.start()
    print(f"[seam2] listening on {sync.sync_host}:{sync.sync_port} — waiting for bridge...", flush=True)
    while server.connected_sockets < server.num_sockets:
        pass
    print("[seam2] bridge connected — starting loop", flush=True)

    def stopper():
        while len(LOG["D"]) < args.max_grants:
            time.sleep(0.05)
        print(f"[seam2] reached {args.max_grants} grants — dumping", flush=True)
        dump(out)
        os._exit(0)
    threading.Thread(target=stopper, daemon=True).start()
    sync.run()
