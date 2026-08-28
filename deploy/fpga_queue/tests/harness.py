"""Test harness: a real daemon, a real socket, a fake FPGA pool.

These tests start the actual ``fq`` daemon as a subprocess and drive it
through the actual client over the actual unix socket.  The only thing that
is faked is the hardware -- the mock backend's "FPGAs" are directories and its
"simulations" are real background processes.  So locking, dispatch, privilege
handling, teardown and recovery are all the shipping code paths.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fq.backends.mock import mock_probe_templates  # noqa: E402
from fq.client import FqClient, JobSpec  # noqa: E402


class Harness:
    """A throwaway pool + daemon."""

    def __init__(self, lanes: int = 2, capacity: int = 1,
                 lane_trees: Optional[dict[str, list[str]]] = None,
                 probe: bool = False, reclaim_policy: str = "manual",
                 reclaim_grace_s: int = 1800,
                 max_fpgas_per_user: int = 0,
                 reserve_after_s: int = 900,
                 admin: bool = False,
                 lane_specs: Optional[list[dict]] = None):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="fqtest-"))
        self.state = self.dir / "state"
        self.mock_root = self.dir / "mock"
        (self.mock_root / "hosts").mkdir(parents=True)
        # A fake chipyard tree -- the mock backend only checks it is named.
        self.tree = self.dir / "tree"
        (self.tree / "sims" / "firesim" / "deploy").mkdir(parents=True)
        (self.tree / "env.sh").write_text("# fake\n")

        if lane_specs is None:
            lane_specs = []
            for i in range(lanes):
                lane_specs.append({
                    "name": f"lane{i}",
                    "mode": "hosts",
                    "hosts": [f"h{i}"],
                    "capacity": capacity,
                })
        self.lane_specs = lane_specs

        pool: dict[str, Any] = {
            "state_dir": str(self.state),
            "socket_path": str(self.state / "fq.sock"),
            "backend": "mock",
            "backend_options": {"mock_root": str(self.mock_root)},
            "defaults": {"priority": 5, "timeout_s": 60,
                         "max_timeout_s": 86400},
            "scheduling": {"poll_interval_s": 0.25,
                           "reserve_after_s": reserve_after_s,
                           "max_fpgas_per_user": max_fpgas_per_user},
            "probe": {**mock_probe_templates(str(self.mock_root)),
                      "enabled": probe, "interval_s": 1,
                      "driver": "sim"},
            "reclaim": {"policy": reclaim_policy, "grace_s": reclaim_grace_s},
            "admins": ([__import__("getpass").getuser()] if admin else []),
            "lanes": lane_specs,
        }
        self.pool_path = self.dir / "pool.json"
        self.pool_path.write_text(json.dumps(pool, indent=2))
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = self.dir / "daemon.log"

    # -- lifecycle -------------------------------------------------------
    def start(self, timeout: float = 20.0) -> "Harness":
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        self._log = self.log_path.open("ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "fq.cli", "daemon",
             "--pool", str(self.pool_path), "-v"],
            env=env, cwd=str(ROOT),
            stdout=self._log, stderr=subprocess.STDOUT)
        sock = self.state / "fq.sock"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if sock.exists():
                try:
                    self.client().ping()
                    return self
                except Exception:  # noqa: BLE001
                    pass
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"daemon died on startup:\n{self.daemon_log()}")
            time.sleep(0.1)
        raise RuntimeError(f"daemon never came up:\n{self.daemon_log()}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        try:
            self._log.close()
        except Exception:  # noqa: BLE001
            pass

    def kill_daemon(self) -> None:
        """SIGKILL -- simulates a daemon crash, no cleanup."""
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def restart(self) -> None:
        self.kill_daemon()
        self.start()

    def cleanup(self) -> None:
        self.stop()
        # Reap every fake simulator this harness created. Crash tests kill the
        # runner before teardown runs, which orphans its phase subprocess and
        # the sim under it -- exactly the real-world orphan case -- so pid
        # files alone are not enough. The marker in argv[0] catches the rest.
        for pf in (self.mock_root / "hosts").rglob("sim_*.pid"):
            try:
                os.kill(int(pf.read_text().strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        subprocess.run(["pkill", "-9", "-f", f"fqmock:{self.mock_root}"],
                       capture_output=True)
        subprocess.run(["pkill", "-9", "-f", f"--workdir {self.state}/jobs"],
                       capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- access ----------------------------------------------------------
    def client(self) -> FqClient:
        return FqClient(str(self.state / "fq.sock"), timeout=20.0)

    def daemon_log(self) -> str:
        try:
            return self.log_path.read_text()
        except OSError:
            return "(no log)"

    def spec(self, **kw: Any) -> JobSpec:
        base: dict[str, Any] = {
            "tree": str(self.tree),
            "hw_config": "mock_hw",
            "timeout_s": 60,
        }
        base.update(kw)
        mock = base.pop("mock", None)
        s = JobSpec(**base)
        d = s.to_dict()
        if mock is not None:
            d["mock"] = mock
        return d  # type: ignore[return-value]

    # -- observation -----------------------------------------------------
    def sims_on(self, host: str) -> int:
        d = self.mock_root / "hosts" / host
        return len(list(d.glob("sim_*.pid"))) if d.is_dir() else 0

    def total_sims(self) -> int:
        return len(list((self.mock_root / "hosts").rglob("sim_*.pid")))

    def start_foreign_sim(self, host: str) -> subprocess.Popen:
        """Start a simulation nobody owns -- the orphan case."""
        d = self.mock_root / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(["sleep", "100000"])
        (d / "sim_0.pid").write_text(str(p.pid))
        return p

    def wait_for(self, fn, timeout: float = 30.0, interval: float = 0.15,
                 what: str = "condition"):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = fn()
            if last:
                return last
            time.sleep(interval)
        raise AssertionError(
            f"timed out waiting for {what} (last={last!r})\n"
            f"--- daemon log ---\n{self.daemon_log()}")

    def wait_state(self, job_id: int, states, timeout: float = 30.0) -> dict:
        if isinstance(states, str):
            states = (states,)
        c = self.client()
        out = self.wait_for(
            lambda: (lambda j: j if j["state"] in states else None)(
                c.get(job_id)),
            timeout=timeout, what=f"job {job_id} in {states}")
        return out

    def wait_terminal(self, job_id: int, timeout: float = 60.0) -> dict:
        return self.wait_state(
            job_id, ("DONE", "FAILED", "CANCELLED", "TIMEOUT"), timeout)
