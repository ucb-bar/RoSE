#!/usr/bin/env python3
"""Run the REAL RoSE gym synchronizer WITHOUT the FireSim manager.

rose.py --task run couples the synchronizer to `firesim runworkload` (the manager).
This launcher runs just the synchronizer + socket server, so it can be paired with a
directly-launched metasim (VFireSim). Start this first (it listens on the bridge's
sync port), then launch VFireSim; the bridge connects and the synchronizer drives the
co-sim protocol (step/bw/route/budget) and serves env observations (e.g. camera images).

Usage (in the .venv-rose environment):
    python run_sync_only.py [--yaml_path <config_gym_*.yaml>]
"""
import argparse
import threading
import gym_synchronizer
from socket_thread import ServerThread

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_path", default=None, help="gym sim config yaml (default: from config_deploy_gym.yaml)")
    args = ap.parse_args()

    sync = gym_synchronizer.Synchronizer(yaml_path=args.yaml_path)
    print(f"[run_sync_only] synchronizer ready: env={sync.env.spec.id} "
          f"nodes={sync.n_fsim_nodes} firesim_step={sync.firesim_step} "
          f"gym_step_per_firesim_step={sync.gym_step_per_firesim_step}", flush=True)

    condition = threading.Condition()
    server = ServerThread(sync, condition)
    server.start()
    print(f"[run_sync_only] listening on {sync.sync_host}:{sync.sync_port} — "
          f"waiting for {server.num_sockets} bridge connection(s)...", flush=True)

    while server.connected_sockets < server.num_sockets:
        pass
    print("[run_sync_only] bridge connected — starting synchronizer loop", flush=True)

    sync.run()   # drives the co-sim protocol; runs until the sim ends / is killed
