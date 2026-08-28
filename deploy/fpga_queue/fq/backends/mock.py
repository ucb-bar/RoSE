"""Mock backend: a fake FPGA pool made of directories.

This exists so the entire daemon -- scheduling, locking, dispatch, timeout
enforcement, teardown, occupancy probing and reclamation -- can be exercised
on a laptop with no AWS, no FireSim and no FPGA.

It is a *behavioural* mock, not a stub.  It reproduces the properties that
make the real system hard:

  * a "simulation" is a real background process on a fake host, so killing it
    is a real kill and the lane's occupancy is a real observable;
  * by default the simulation **never terminates** -- exactly like a
    bare-metal Zephyr guest that prints and then idles -- so the timeout and
    uartlog-sentinel paths are the ones under test;
  * ``infrasetup`` can be made slow, and any phase can be made to fail.

Every mock host is a directory under ``mock_root/hosts/<host>/``.  A live sim
is a ``sim_<slot>.pid`` file there, which is what the occupancy Prober counts
-- so the tests drive the real ``Prober`` code, just pointed at a template
that looks at files instead of ssh'ing to EC2.

Per-job knobs live in ``spec["mock"]``:

    infrasetup_s   float  how long INFRASETUP takes           (default 0)
    run_s          float  how long RUN lasts before exiting 0 (default: never)
    never_exit     bool   RUN blocks forever                  (default True)
    uart           list   lines written to the fake uartlog at RUN start
    uart_delay_s   float  delay before those lines appear     (default 0)
    fail_phase     str    phase name that should exit nonzero (default None)
"""

from __future__ import annotations

import os
import pathlib
import shlex
from typing import Any, Optional

from .base import Backend, JobContext, Phase


class MockBackend(Backend):
    name = "mock"

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        root = self.options.get("mock_root") or os.environ.get("FQ_MOCK_ROOT")
        if not root:
            raise ValueError(
                "mock backend needs 'mock_root' in backend_options "
                "(or $FQ_MOCK_ROOT)")
        self.root = pathlib.Path(root)
        (self.root / "hosts").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def host_dir(self, host: str) -> pathlib.Path:
        return self.root / "hosts" / host

    def _knobs(self, ctx: JobContext) -> dict:
        return dict(ctx.spec.get("mock") or {})

    def validate(self, spec: dict, cfg: Any) -> list[str]:
        errs = []
        if not spec.get("tree"):
            errs.append("'tree' is required")
        if not spec.get("hw_config") and not spec.get("agfi"):
            errs.append("one of 'hw_config' or 'agfi' is required")
        if int(spec.get("num_fpgas", 1)) < 1:
            errs.append("num_fpgas must be >= 1")
        elf = spec.get("elf")
        if elf and not pathlib.Path(elf).exists():
            errs.append(f"elf not found: {elf}")
        return errs

    # ------------------------------------------------------------------
    def stage(self, ctx: JobContext) -> None:
        for h in ctx.lane.hosts:
            self.host_dir(h).mkdir(parents=True, exist_ok=True)
        ctx.runtime_yaml.write_text(
            "# mock config_runtime\n"
            f"lane: {ctx.lane.name}\n"
            f"hosts: {list(ctx.lane.hosts)}\n"
            f"hw_config: {ctx.hw_config}\n"
            f"no_net_num_nodes: {ctx.num_fpgas}\n"
            f"suffix_tag: fq{ctx.job_id}\n")

    # ------------------------------------------------------------------
    def argv(self, ctx: JobContext, phase: Phase) -> Optional[list[str]]:
        k = self._knobs(ctx)
        hosts = list(ctx.lane.hosts) or [f"{ctx.lane.name}-host0"]
        hd = [str(self.host_dir(h)) for h in hosts]
        q = shlex.quote

        if k.get("fail_phase") == phase.value:
            return ["bash", "-c",
                    f"echo 'mock: forced failure in {phase.value}' >&2; exit 3"]

        if phase is Phase.INFRASETUP:
            secs = float(k.get("infrasetup_s", 0))
            mk = "; ".join(f"mkdir -p {q(d)}" for d in hd)
            return ["bash", "-c",
                    f"{mk}; echo 'mock: infrasetup hw={ctx.hw_config} "
                    f"nodes={ctx.num_fpgas}'; sleep {secs}; "
                    f"echo 'mock: FPGAs programmed'"]

        if phase is Phase.RUN:
            uart = k.get("uart") or []
            delay = float(k.get("uart_delay_s", 0))
            never = bool(k.get("never_exit", True)) and "run_s" not in k
            run_s = float(k.get("run_s", 0))
            # A real background process per slot, whose pid file is what the
            # occupancy probe counts.  `exec sleep` so the pid we record is
            # the process that is actually alive.
            # argv[0] carries a marker so a test harness (and a human) can
            # find and reap these fake simulators unambiguously.
            spawn = "; ".join(
                f"mkdir -p {q(d)} && (exec -a {q('fqmock:' + str(self.root))} "
                f"sleep 100000) & echo $! > {q(d)}/sim_0.pid"
                for d in hd)
            writes = ""
            if uart:
                body = "".join(f"echo {q(str(line))} >> $D/uartlog; "
                               for line in uart)
                writes = ("( sleep %s; for D in %s; do mkdir -p $D; %s done ) & "
                          % (delay, " ".join(q(d) for d in hd), body))
            tail = ("sleep 100000" if never
                    else f"sleep {run_s}; echo 'mock: guest exited'")
            return ["bash", "-c",
                    f"{spawn}; {writes}echo 'mock: simulation started'; {tail}"]

        if phase is Phase.KILL:
            # Mirrors the real backend: host-wide, kills every slot's sim.
            kills = "; ".join(
                f"for p in {q(d)}/sim_*.pid; do [ -f \"$p\" ] && "
                f"kill -9 \"$(cat $p)\" 2>/dev/null; rm -f \"$p\"; done"
                for d in hd)
            return ["bash", "-c",
                    f"shopt -s nullglob; {kills}; echo 'mock: killed'"]

        return None

    # ------------------------------------------------------------------
    def uartlog_argv(self, ctx: JobContext) -> Optional[list[str]]:
        hosts = list(ctx.lane.hosts) or [f"{ctx.lane.name}-host0"]
        cats = "; ".join(
            f"cat {shlex.quote(str(self.host_dir(h) / 'uartlog'))} 2>/dev/null"
            for h in hosts)
        return ["bash", "-c", f"{cats}; true"]

    def argv_collect(self, ctx: JobContext, dest: pathlib.Path) -> list[str]:
        hosts = list(ctx.lane.hosts) or [f"{ctx.lane.name}-host0"]
        q = shlex.quote
        cps = "; ".join(
            f"[ -f {q(str(self.host_dir(h) / 'uartlog'))} ] && "
            f"cp {q(str(self.host_dir(h) / 'uartlog'))} "
            f"{q(str(dest))}/uartlog-{q(h)} || true"
            for h in hosts)
        return ["bash", "-c", f"mkdir -p {q(str(dest))}; {cps}; true"]


# Probe/reclaim templates that make the real occupancy.Prober work against
# the mock pool.  Tests pass these through the pool config, so the code under
# test is the shipping Prober, not a test double.
def mock_probe_templates(mock_root: str) -> dict[str, str]:
    root = shlex.quote(str(mock_root))
    # Count with `set -- <glob>; echo $#` rather than `ls | wc -l`: under
    # nullglob an unmatched glob leaves `ls` with no arguments, and it then
    # helpfully lists the current directory instead of reporting zero.
    count = "shopt -s nullglob; set -- %s/hosts/{host}/sim_*.pid; echo $#"
    return {
        "template": "bash -c '" + (count % root) + "'",
        "reclaim_template": (
            "bash -c 'shopt -s nullglob; "
            "for p in %s/hosts/{host}/sim_*.pid; do "
            "kill -9 \"$(cat $p)\" 2>/dev/null; rm -f \"$p\"; done; "
            % root) + (count % root) + "'",
    }
