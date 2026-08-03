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

def _noise(level, seed):
    return {"ROSE_SENSOR_NOISE_LEVEL": str(level), "ROSE_SENSOR_NOISE_SEED": str(seed)}

def matrix_baseline():
    # clean hover over a few seeds (level 0 -> deterministic, seeds are a no-op but confirm
    # reproducibility of the harness path itself)
    return [("clean_seed%d" % s, _noise(0, s)) for s in range(3)]

def matrix_noise():
    # stress plan section 5.3 noise axis: {clean, L1, L2} x seeds. L0 deterministic (1 cell).
    cells = [("L0_clean", _noise(0, 0))]
    for lvl in (1, 2):
        for seed in range(3):
            cells.append(("L%d_seed%d" % (lvl, seed), _noise(lvl, seed)))
    return cells

def matrix_noise_quick():
    # one cell per level at a fixed seed — cheap Phase-1 validation (clean/L1/L2)
    return [("L%d_seed0" % lvl, _noise(lvl, 0)) for lvl in (0, 1, 2)]

def matrix_delay():
    # section 2: sensor transport delay (control steps). d=0 must equal Phase-1 L1 (regression).
    base = _noise(1, 0)
    cells = [("delay0", base)]
    for d in (1, 2, 3):
        env = dict(base)
        env.update({"ROSE_SENSOR_DELAY_FLOW": str(d), "ROSE_SENSOR_DELAY_TOF": str(d),
                    "ROSE_SENSOR_DELAY_ACCEL": str(min(d, 1)), "ROSE_SENSOR_DELAY_GYRO": str(min(d, 1))})
        cells.append(("delay%d" % d, env))
    return cells

def matrix_scenario():
    # section 3.1: harder initial conditions (level flight vs tilt/offset/velocity), clean sensors
    cells = [("ic_none", _noise(0, 0))]
    hard = {"tilt": {"ROSE_IC_TILT_DEG": "15"},
            "offset": {"ROSE_IC_POS": "0.2", "ROSE_IC_Z": "0.15"},
            "velocity": {"ROSE_IC_VEL": "0.3", "ROSE_IC_RATE": "0.5"}}
    for name, over in hard.items():
        for seed in range(2):
            env = _noise(0, seed); env.update(over); env["ROSE_SCENARIO_SEED"] = str(seed)
            cells.append(("ic_%s_s%d" % (name, seed), env))
    return cells

def matrix_combined():
    # a representative noise x scenario cross-section (the section 5.3 matrix, trimmed)
    cells = [("clean", _noise(0, 0))]
    for lvl in (1, 2):
        cells.append(("L%d" % lvl, _noise(lvl, 0)))
        env = _noise(lvl, 0); env["ROSE_IC_TILT_DEG"] = "15"
        cells.append(("L%d_tilt" % lvl, env))
        env = _noise(lvl, 0); env.update({"ROSE_SENSOR_DELAY_FLOW": "2", "ROSE_SENSOR_DELAY_TOF": "2"})
        cells.append(("L%d_delay2" % lvl, env))
    return cells

def matrix_smoke():
    return [("clean", {})]

MATRICES = {
    "baseline": matrix_baseline,
    "noise": matrix_noise,
    "noise_quick": matrix_noise_quick,
    "delay": matrix_delay,
    "scenario": matrix_scenario,
    "combined": matrix_combined,
    "smoke": matrix_smoke,
}


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
        # optional SoC-clock override (0 = inherit config); not a speedup, see argparse note
        if self.args.firesim_freq:
            env["ROSE_FIRESIM_FREQ"] = str(self.args.firesim_freq)
        if self.args.firesim_step:
            env["ROSE_FIRESIM_STEP"] = str(self.args.firesim_step)
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
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # ALWAYS escalate to SIGKILL on the whole group: the `bash run_spike_rose_lockstep.sh`
    # wrapper dies fast on SIGTERM (so proc.wait returns), but its `rose_spike_sim`
    # grandchild can survive SIGTERM and orphan (ppid=1), pinning a CPU core forever and
    # starving later cells. Killing the group unconditionally reaps the grandchild too.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default="baseline", choices=list(MATRICES),
                    help="which cell matrix to run (default: baseline)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="max concurrent cells. The loop is CPU-bound on spike (1 core/cell, "
                         "GPU idle for the tiny scene), so on a many-core host this can be high "
                         "(~10-14 on 48 cores); leave headroom on a shared box.")
    # NOTE: a controlled A/B (5M@1GHz vs 1.5M@300MHz) showed only ~7% wall-clock difference,
    # i.e. the SoC cycle budget is NOT the bottleneck -- Isaac Sim's per-step CPU cost is (the
    # GPU is idle; spike's 100% CPU is a busy-wait spin in rose_rx, not real work). So these
    # default to inheriting config_deploy_gym.yaml (1 GHz). They remain available to model a
    # different SoC clock for a co-design study, but they do NOT speed up experimentation --
    # raise --jobs for that instead.
    ap.add_argument("--firesim-freq", type=int, default=0,
                    help="override modeled SoC clock (Hz); 0 = inherit config. Does NOT speed "
                         "up the sim (Isaac per-step CPU is the bottleneck), only changes the "
                         "modeled clock for a co-design study.")
    ap.add_argument("--firesim-step", type=int, default=0,
                    help="override SoC cycles/step; 0 = inherit config. Keep "
                         "firesim-step/firesim-freq == the env gym_timestep.")
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
