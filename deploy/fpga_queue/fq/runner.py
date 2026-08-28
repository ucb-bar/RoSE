"""Per-job runner process.

One runner process per job.  It is spawned by the daemon, inherits the lane's
flock on a file descriptor, and holds that lock until it exits.  Everything
about a running job lives here; the daemon only observes.

Why a separate process rather than a thread in the daemon
---------------------------------------------------------
Because the lane lock must outlive the daemon.  If the daemon is restarted --
for a config change, a bug fix, or a crash -- a job that is mid-``infrasetup``
on a $2/hour FPGA must keep its hardware.  A thread would die with the daemon
and the lock would drop, letting the next daemon dispatch a second job onto an
FPGA that is still being reflashed by the first.  A separate process holding
an inherited fd gives exactly the right lifetime: the lock is released by the
kernel when *the job* ends, not when the daemon does.

The runner writes its progress to ``<workdir>/status.json``.  That file is
**advisory**: the submitter owns the directory and could write anything into
it.  It is used to display phase and to record exit codes.  It is never used
to decide whether a lane is free -- only the flock decides that.  See
``docs/FPGA_QUEUE_DESIGN.md``.

Completion
----------
The RUN phase ends on the first of these, in priority order:

  1. **the run command exiting by itself** -> DONE / FAILED by rc.
     This is the normal, expected path.  A guest that calls
     ``sys_reboot(SYS_REBOOT_COLD)`` writes the HTIF exit word, FireSim's TSI
     bridge stops the simulation, and ``runworkload`` returns and copies
     results back on its own.  No polling involved.
  2. ``sentinel_regex`` matching the live uartlog -> DONE   (opt-in)
  3. ``fail_regex`` matching the live uartlog     -> FAILED (opt-in)
  4. an explicit ``<workdir>/done`` marker        -> DONE   (client signal)
  5. ``<workdir>/cancel``                         -> CANCELLED
  6. ``timeout_s`` elapsing -> TIMEOUT, or DONE under completion mode
     ``timeout`` where expiry is the intended ending.

(6) is a backstop, not a mechanism of first resort: a guest that never exits
holds its FPGA for the whole timeout.  See fq/completion.py.

KILL and COLLECT then always run, whichever of those fired.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from . import completion
from .backends import get_backend
from .backends.base import JobContext, LaneInfo, Phase

POLL_S = 2.0
# How long to wait for a phase to die politely before SIGKILL.
GRACE_S = 15.0



@contextlib.contextmanager
def _infrasetup_gate(path, deadline_s: float = 900.0, say=None):
    """Serialise infrasetup across concurrently dispatched jobs.

    FireSim's infrasetup extracts a SHARED ``driver-bundle.tar.gz`` on the
    manager.  Two jobs entering infrasetup within a few seconds of each other
    race on that one file and a loser dies with
    ``Requested: tar -xf driver-bundle.tar.gz`` rc=2 -- observed on jobs 94,
    127, 131 and 138, on different lanes, each with a neighbour dispatched
    seconds earlier.  Infrasetup is short when warm, so serialising it costs
    far less than a lost job plus the FPGA time it was holding.

    Bounded rather than blocking: if the holder wedges, later jobs proceed
    anyway after ``deadline_s`` instead of deadlocking the whole pool.
    """
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    held = False
    end = time.time() + deadline_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError:
                if time.time() >= end:
                    if say:
                        say("infrasetup gate: timed out waiting; proceeding "
                            "unserialised (a holder may be wedged)")
                    break
                time.sleep(2.0)
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


class Runner:
    def __init__(self, workdir: pathlib.Path, lock_fd: Optional[int] = None):
        self.workdir = workdir
        self.lock_fd = lock_fd
        self.plan: dict[str, Any] = json.loads(
            (workdir / "runner.json").read_text())
        self.spec: dict[str, Any] = self.plan["spec"]
        lane = self.plan["lane"]
        self.lane = LaneInfo(
            name=lane["name"],
            mode=lane.get("mode", "hosts"),
            tag=lane.get("tag"),
            capacity=int(lane.get("capacity", 1)),
            hosts=tuple(lane.get("hosts") or ()),
            instance_type=lane.get("instance_type"),
            manage_hosts=bool(lane.get("manage_hosts", False)),
        )
        self.backend = get_backend(self.plan["backend"],
                                   self.plan.get("backend_options"))
        results = self.plan.get("results_dir")
        self.ctx = JobContext(
            job_id=int(self.plan["job_id"]),
            user=self.plan["user"],
            uid=int(self.plan["uid"]),
            gid=self.plan.get("gid"),
            spec=self.spec,
            lane=self.lane,
            workdir=workdir,
            results_dir=pathlib.Path(results) if results else None,
        )
        self.infrasetup_timeout_s = float(
            self.plan.get("infrasetup_timeout_s") or 0)
        # Completion contract: exit (default) -> sentinel -> timeout.
        # See fq/completion.py for why the ordering matters.
        self.policy = completion.from_spec(
            self.spec, int(self.plan.get("timeout_s") or 0))
        self.timeout_s = float(self.policy.timeout_s or 0)
        self.done_re, self.fail_re = self.policy.compiled()
        self.log = (workdir / "job.log").open("ab", buffering=0)
        self.state = "RUNNING"
        self.exit_code: Optional[int] = None
        self.message = ""
        self.started_at = time.time()

    # ------------------------------------------------------------------
    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        with contextlib.suppress(OSError):
            self.log.write(line.encode())

    def write_status(self, phase: str) -> None:
        """Atomically publish progress.  Advisory -- see module docstring."""
        doc = {
            "job_id": self.ctx.job_id,
            "state": self.state,
            "phase": phase,
            "pid": os.getpid(),
            "lane": self.lane.name,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "exit_code": self.exit_code,
            "message": self.message,
            "completion": self.policy.to_dict(),
        }
        tmp = self.workdir / "status.json.tmp"
        try:
            tmp.write_text(json.dumps(doc) + "\n")
            os.replace(tmp, self.workdir / "status.json")
        except OSError:
            pass

    # ------------------------------------------------------------------
    def cancelled(self) -> bool:
        return (self.workdir / "cancel").exists()

    def client_done(self) -> bool:
        return (self.workdir / "done").exists()

    def read_uartlog(self) -> str:
        argv = self.backend.uartlog_argv(self.ctx)
        if not argv:
            return ""
        try:
            cp = subprocess.run(argv, capture_output=True, text=True,
                                timeout=60)
            return cp.stdout or ""
        except (subprocess.TimeoutExpired, OSError):
            return ""

    # ------------------------------------------------------------------
    def _terminate(self, proc: subprocess.Popen) -> int:
        """SIGTERM the process group, then SIGKILL if it will not go."""
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            return proc.wait(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError,
                                     OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                return proc.wait(timeout=GRACE_S)
            except subprocess.TimeoutExpired:
                return -9

    def run_phase(self, phase: Phase, *, timeout: float = 0.0,
                  watch_uart: bool = False,
                  interruptible: bool = True) -> tuple[Optional[int], str]:
        """Run one phase.  Returns (rc, outcome).

        outcome is DONE / FAILED / CANCELLED / TIMEOUT / SKIPPED, describing
        why the phase ended -- not whether the job as a whole succeeded.

        ``interruptible=False`` is used for teardown.  Teardown must never
        abort on the cancel flag: the flag is still set when we get here (it
        is what brought us here), and a KILL phase that cancels itself would
        leave the simulation attached to the FPGA -- turning "cancel my job"
        into "occupy an FPGA forever". Teardown is bounded by its timeout
        only.
        """
        argv = self.backend.argv(self.ctx, phase)
        if argv is None:
            return None, "SKIPPED"
        self.write_status(phase.value)
        self.say(f"=== phase {phase.value} ===")
        self.say(f"$ {' '.join(argv)}")
        try:
            proc = subprocess.Popen(argv, cwd=str(self.workdir),
                                    stdout=self.log, stderr=self.log,
                                    start_new_session=True)
        except OSError as exc:
            self.say(f"failed to launch phase: {exc}")
            return 127, "FAILED"

        deadline = (time.time() + timeout) if timeout > 0 else None
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc, ("DONE" if rc == 0 else "FAILED")

            if interruptible and self.cancelled():
                self.say("cancel requested")
                return self._terminate(proc), "CANCELLED"

            if deadline and time.time() > deadline:
                # Under completion mode 'timeout' the cap IS the intended
                # ending, so expiry is success rather than failure.
                intended = (watch_uart and self.policy.timeout_is_success)
                self.say(f"timeout after {timeout:.0f}s in {phase.value}"
                         + (" (expected: completion mode 'timeout')"
                            if intended else ""))
                return self._terminate(proc), ("DONE" if intended
                                               else "TIMEOUT")

            if watch_uart:
                if self.client_done():
                    self.say("client signalled completion")
                    return self._terminate(proc), "DONE"
                if self.policy.watches_uart:
                    text = self.read_uartlog()
                    if self.fail_re and self.fail_re.search(text):
                        self.say(f"fail_regex matched: "
                                 f"{self.fail_re.pattern!r}")
                        return self._terminate(proc), "FAILED"
                    if self.done_re and self.done_re.search(text):
                        self.say(f"done_regex matched: "
                                 f"{self.done_re.pattern!r}")
                        return self._terminate(proc), "DONE"

            self.write_status(phase.value)
            time.sleep(POLL_S)

    # ------------------------------------------------------------------
    def main(self) -> int:
        outcome = "DONE"
        rc: Optional[int] = 0
        try:
            # --- STAGE (in-process; it is file manipulation, not a command)
            self.write_status("STAGE")
            self.say(f"job {self.ctx.job_id} on lane {self.lane.name} "
                     f"hosts={list(self.lane.hosts)}")
            try:
                self.backend.stage(self.ctx)
            except Exception as exc:  # noqa: BLE001
                self.say(f"STAGE failed: {type(exc).__name__}: {exc}")
                outcome, rc = "FAILED", 1
                raise _PhaseAbort from exc

            # Stale uartlogs on the run hosts outlive the job that wrote
            # them, so a job that never reaches RUN would otherwise collect the
            # previous occupant's log and report it as its own. Clear first.
            _clr = getattr(self.backend, "argv_clear_uartlog", None)
            if _clr is not None:
                _argv = _clr(self.ctx)
                if _argv:
                    self.say("clearing stale uartlogs on run hosts")
                    with contextlib.suppress(OSError,
                                             subprocess.SubprocessError):
                        subprocess.run(_argv, stdout=self.log, stderr=self.log,
                                       timeout=120)

            for phase, tmo, watch in (
                (Phase.LAUNCH, 0.0, False),
                (Phase.INFRASETUP, self.infrasetup_timeout_s, False),
                (Phase.RUN, self.timeout_s, True),
            ):
                if phase is Phase.INFRASETUP:
                    _gate = self.workdir.parent.parent / "infrasetup.lock"
                    with _infrasetup_gate(_gate, say=self.say):
                        rc, outcome = self.run_phase(phase, timeout=tmo,
                                                     watch_uart=watch)
                else:
                    rc, outcome = self.run_phase(phase, timeout=tmo,
                                                 watch_uart=watch)
                if outcome == "SKIPPED":
                    outcome, rc = "DONE", 0
                    continue
                if outcome != "DONE":
                    raise _PhaseAbort
        except _PhaseAbort:
            pass
        except BaseException as exc:  # noqa: BLE001
            self.say(f"unexpected: {type(exc).__name__}: {exc}")
            outcome, rc = "FAILED", 1
        finally:
            # ---- teardown ALWAYS runs -------------------------------------
            # This is the path that frees the hardware.  It runs after
            # success, failure, cancel, timeout and unexpected exceptions
            # alike, and each step is independently guarded so that one
            # failure cannot skip the next.
            for phase in (Phase.KILL, Phase.TERMINATE):
                try:
                    krc, _ = self.run_phase(phase, timeout=600.0,
                                            interruptible=False)
                    if krc not in (None, 0):
                        self.say(f"{phase.value} returned {krc} "
                                 f"(continuing anyway)")
                except Exception as exc:  # noqa: BLE001
                    self.say(f"{phase.value} raised {exc} (continuing)")
            try:
                self.collect()
            except Exception as exc:  # noqa: BLE001
                self.say(f"COLLECT raised {exc} (continuing)")

        self.state = outcome if outcome != "SKIPPED" else "DONE"
        self.exit_code = rc if rc is not None else 0
        self.message = self.message or ""
        self.write_status(self.state)
        self.say(f"job {self.ctx.job_id} finished state={self.state} rc={rc}")
        with contextlib.suppress(OSError):
            self.log.close()
        # The lane lock is released here, implicitly, when the process exits
        # and the kernel closes the inherited fd.  Nothing to do.
        return 0 if self.state == "DONE" else 1

    def collect(self) -> None:
        dest = self.ctx.results_dir or (self.workdir / "results")
        dest = pathlib.Path(dest)
        self.write_status("COLLECT")
        self.say(f"=== phase COLLECT -> {dest} ===")
        dest.mkdir(parents=True, exist_ok=True)
        fn = getattr(self.backend, "argv_collect", None)
        if fn is not None:
            argv = fn(self.ctx, dest)
            if argv:
                self.say(f"$ {' '.join(argv)}")
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(argv, stdout=self.log, stderr=self.log,
                                   timeout=900)
        # Always keep a snapshot of the UART, even if the run failed early --
        # it is usually the only evidence of what the guest actually did.
        text = self.read_uartlog()
        if text:
            with contextlib.suppress(OSError):
                (dest / "uartlog").write_text(text)
        for name in ("job.log", "config_runtime.yaml", "runner.json",
                     "status.json", "workload.json", "config_hwdb.yaml"):
            src = self.workdir / name
            if src.exists():
                with contextlib.suppress(OSError):
                    (dest / name).write_bytes(src.read_bytes())


class _PhaseAbort(Exception):
    """Internal: stop the forward phase sequence, go straight to teardown."""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fq-runner")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--lock-fd", type=int, default=None,
                    help="inherited fd holding this lane's flock; the runner "
                         "just keeps it open for its lifetime")
    args = ap.parse_args(argv)
    wd = pathlib.Path(args.workdir)

    # Do not let the lock fd leak into the phase subprocesses we spawn: if a
    # long-lived grandchild inherited it, the lane would stay locked after
    # this runner exited.
    if args.lock_fd is not None:
        with contextlib.suppress(OSError):
            os.set_inheritable(args.lock_fd, False)

    runner = Runner(wd, args.lock_fd)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
