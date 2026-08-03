#!/usr/bin/env python3
"""Parallel RoSE co-sim stress harness.

Runs many closed-loop RoSE cells (Isaac Sim physics + Spike SoC) concurrently to sweep the
stress matrix in docs/ROSE_FLIGHT_STRESS_PLAN.md (noise x scenario x seed) far faster than
one-at-a-time. Each cell is a fully independent (synchronizer + rose_spike_sim) pair on its
OWN sync port, so N cells share one host without colliding:

    cell k  ->  ROSE_SYNC_PORT = base_port + slot   (synchronizer binds it; spike --rose-port
                matches it)  ->  per-cell ground-truth CSV  ->  scored offline by
                rose_stress_score.py.

Concurrency is GPU-bound (Isaac Sim), not CPU-bound, so --jobs defaults low; raise it if the
GPU has headroom. Runs are driven to a fixed SIM time (a target CSV row count = duration /
control_dt), not wall-clock, so every cell sees the same flight length regardless of host load.

A "cell" is just a name + a dict of environment overrides. Today the overrides are the clean
baseline (plus seeds); once the env gains the ROSE_SENSOR_NOISE_* / ROSE_SCENARIO_* hooks from
the plan, the SAME machinery sweeps them — only the matrix grows.

Usage:
    deploy/scripts/rose_stress_harness.py --matrix baseline --jobs 3 --duration 12
    deploy/scripts/rose_stress_harness.py --matrix baseline --out-dir /path/results
    # then: deploy/scripts/rose_stress_score.py <out-dir>

Env knobs (with sensible defaults for the RoSE dev host):
    ROSE_ISAAC_PYTHON   interpreter that can import Isaac (default: env_isaaclab conda python)
    ROSE_STRESS_ELF     guest ELF (default: the rose_flight_controller build)
    ROSE_SYNC_BASE_PORT base sync port (default 10010)
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                               cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()

DEFAULT_ISAAC_PY = os.environ.get(
    "ROSE_ISAAC_PYTHON", "/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python")
DEFAULT_ELF = os.environ.get(
    "ROSE_STRESS_ELF",
    os.path.join(REPO, "soc/sim/zephyr_rose_builds/rose_flight_controller/zephyr/zephyr.elf"))
DEFAULT_GYM_YAML = os.path.join(REPO, "deploy/config/config_gym_IsaacCrazyflieSensorEnv-v0.yaml")
BASE_PORT = int(os.environ.get("ROSE_SYNC_BASE_PORT", "10010"))
CTRL_DT = 0.005          # 200 Hz control step (matches gym_timestep)
BOOT_TIMEOUT = 300.0     # seconds to wait for Isaac to boot + sync to listen
POLL = 0.5

# ---- Matrices -------------------------------------------------------------------------
# Each cell: (name, {env overrides}). Baseline = the clean, validated hover, over a few seeds
# (seeds are a no-op until the noise layer reads ROSE_SENSOR_NOISE_SEED, but wiring them now
# means adding noise cells later is a one-line change).

def matrix_baseline():
    return [("clean_seed%d" % s, {"ROSE_SENSOR_NOISE_SEED": str(s)}) for s in range(3)]

def matrix_smoke():
    return [("clean", {})]

MATRICES = {"baseline": matrix_baseline, "smoke": matrix_smoke}


class Cell:
    def __init__(self, name, env, slot, args):
        self.name = name
        self.env_over = env
        self.slot = slot
        self.port = BASE_PORT + slot
        self.args = args
        self.csv = os.path.join(args.out_dir, name + ".csv")
        self.sync_log = os.path.join(args.out_dir, name + ".sync.log")
        self.spike_log = os.path.join(args.out_dir, name + ".spike.log")
        self.result = {"name": name, "port": self.port, "csv": self.csv}

    def _wait_listening(self, proc):
        deadline = time.time() + BOOT_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                return False  # sync died during boot
            try:
                with open(self.sync_log) as f:
                    if "listening on" in f.read():
                        return True
            except FileNotFoundError:
                pass
            time.sleep(POLL)
        return False

    def _wait_done(self, spike, target_rows):
        """Return when the sim reaches target_rows OR spike exits OR the run-timeout fires."""
        deadline = time.time() + self.args.run_timeout
        while time.time() < deadline:
            if spike.poll() is not None:
                return "spike_exit"
            try:
                # cheap row count without loading the file
                with open(self.csv, "rb") as f:
                    rows = sum(1 for _ in f)
                if rows >= target_rows + 1:  # +1 header
                    return "target_rows"
            except FileNotFoundError:
                pass
            time.sleep(POLL)
        return "timeout"

    def run(self):
        target_rows = int(round(self.args.duration / CTRL_DT))
        env = dict(os.environ)
        env.update({
            "ROSE_DIR": REPO,
            "ROSE_SYNC_PORT": str(self.port),
            "ROSE_TRAJ_CSV": self.csv,
            "ROSE_ENV_DEBUG": "0",
        })
        env.update(self.env_over)
        for f in (self.csv, self.sync_log, self.spike_log):
            if os.path.exists(f):
                os.remove(f)

        sync = spike = None
        try:
            # 1) synchronizer (Isaac physics) on this cell's port
            with open(self.sync_log, "w") as slog:
                sync = subprocess.Popen(
                    [self.args.isaac_python, "run_sync_only.py", "--yaml_path", self.args.gym_yaml],
                    cwd=os.path.join(REPO, "deploy/hephaestus"),
                    env=env, stdout=slog, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid)
            if not self._wait_listening(sync):
                self.result["status"] = "boot_fail"
                return self.result

            # 2) Spike SoC bridging to the same port
            senv = dict(env)
            senv["ROSE_SPIKE_TIMEOUT"] = str(int(self.args.run_timeout))
            with open(self.spike_log, "w") as plog:
                spike = subprocess.Popen(
                    ["bash", os.path.join(REPO, "soc/sim/run_spike_rose_lockstep.sh"),
                     self.args.elf, "1"],
                    cwd=REPO, env=senv, stdout=plog, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid)

            reason = self._wait_done(spike, target_rows)
            self.result["status"] = "ok" if reason in ("target_rows", "spike_exit") else reason
            self.result["reason"] = reason
            try:
                with open(self.csv, "rb") as f:
                    self.result["rows"] = sum(1 for _ in f) - 1
            except FileNotFoundError:
                self.result["rows"] = 0
            return self.result
        finally:
            for p in (spike, sync):
                _kill_group(p)


def _kill_group(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default="baseline", choices=list(MATRICES),
                    help="which cell matrix to run (default: baseline)")
    ap.add_argument("--jobs", type=int, default=3, help="max concurrent cells (GPU-bound)")
    ap.add_argument("--duration", type=float, default=12.0, help="sim seconds per cell")
    ap.add_argument("--run-timeout", type=float, default=1800.0,
                    help="wall-clock ceiling per cell (s)")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "deploy/hephaestus/logs/stress"),
                    help="results dir for per-cell CSV/logs + summary.json")
    ap.add_argument("--isaac-python", default=DEFAULT_ISAAC_PY)
    ap.add_argument("--elf", default=DEFAULT_ELF)
    ap.add_argument("--gym-yaml", default=DEFAULT_GYM_YAML)
    args = ap.parse_args()

    args.elf = os.path.abspath(args.elf)
    if not os.path.exists(args.elf):
        sys.exit("ELF not found: %s (build it first)" % args.elf)
    if not os.path.exists(args.isaac_python):
        sys.exit("Isaac python not found: %s (set ROSE_ISAAC_PYTHON)" % args.isaac_python)
    os.makedirs(args.out_dir, exist_ok=True)

    spec = MATRICES[args.matrix]()
    print("[harness] matrix=%s cells=%d jobs=%d duration=%.1fs out=%s"
          % (args.matrix, len(spec), args.jobs, args.duration, args.out_dir), flush=True)

    # Slot = index within the concurrency window; reused as cells complete so ports stay in a
    # small band. A ThreadPoolExecutor of size --jobs gives at most --jobs live cells; we assign
    # each a slot from a free-list to keep ports distinct among the *concurrent* set.
    free_slots = list(range(args.jobs))
    slot_lock = threading.Lock()

    def run_one(name, env):
        with slot_lock:
            slot = free_slots.pop()
        try:
            print("[harness] START %-16s port=%d" % (name, BASE_PORT + slot), flush=True)
            res = Cell(name, env, slot, args).run()
            print("[harness] DONE  %-16s status=%s rows=%s"
                  % (name, res.get("status"), res.get("rows")), flush=True)
            return res
        finally:
            with slot_lock:
                free_slots.append(slot)

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, n, e): n for n, e in spec}
        try:
            for fut in as_completed(futs):
                results.append(fut.result())
        except KeyboardInterrupt:
            print("[harness] interrupted — cancelling", flush=True)
            for f in futs:
                f.cancel()
            raise

    summary = os.path.join(args.out_dir, "summary.json")
    with open(summary, "w") as f:
        json.dump({"matrix": args.matrix, "duration": args.duration,
                   "cells": sorted(results, key=lambda r: r["name"])}, f, indent=2)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print("[harness] complete: %d/%d cells ran. summary -> %s" % (ok, len(results), summary),
          flush=True)
    print("[harness] score with: deploy/scripts/rose_stress_score.py %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
