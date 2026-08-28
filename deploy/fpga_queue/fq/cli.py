"""`fq` command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

from .client import FqClient, FqError, JobSpec

TERMINAL = ("DONE", "FAILED", "CANCELLED", "TIMEOUT")


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


_STATE_COLOR = {
    "QUEUED": "33", "DISPATCHING": "36", "RUNNING": "32",
    "DONE": "92", "FAILED": "31", "TIMEOUT": "31", "CANCELLED": "90",
}


def _state(s: str) -> str:
    return _c(f"{s:<11}", _STATE_COLOR.get(s, "0"))


def _age(ts: Optional[float]) -> str:
    if not ts:
        return "-"
    d = max(0, time.time() - float(ts))
    if d < 90:
        return f"{d:.0f}s"
    if d < 5400:
        return f"{d/60:.0f}m"
    return f"{d/3600:.1f}h"


# ---------------------------------------------------------------------------
def cmd_submit(a: argparse.Namespace, c: FqClient) -> int:
    comp: dict[str, Any] = {}
    if a.completion:
        comp["mode"] = a.completion
    if a.sentinel:
        comp["sentinel_regex"] = a.sentinel
    if a.fail_regex:
        comp["fail_regex"] = a.fail_regex

    spec = JobSpec(
        tree=os.path.abspath(os.path.expanduser(a.tree)),
        hw_config=a.hw_config,
        agfi=a.agfi,
        num_fpgas=a.num_fpgas,
        elf=(os.path.abspath(os.path.expanduser(a.elf)) if a.elf else None),
        workload=a.workload,
        rootfs=a.rootfs,
        runtime_args=dict(kv.split("=", 1) for kv in a.set) if a.set else {},
        timeout_s=a.timeout,
        completion=comp or None,
        priority=a.priority,
        lane_hint=a.lane,
        project=a.project,
        comment=a.comment,
        results_dir=(os.path.abspath(os.path.expanduser(a.results))
                     if a.results else None),
    )
    res = c._call("submit", spec={**spec.to_dict(), "cwd": os.getcwd()})
    jid = res["job_id"]
    print(f"submitted job {_c(str(jid), '1')}  "
          f"(completion={res['completion']['mode']}, "
          f"timeout={res['completion']['timeout_s']}s)")
    if res.get("advisory"):
        print(_c("note: " + res["advisory"], "33"))
    print(f"  logs:   fq logs {jid} -f")
    print(f"  status: fq status {jid}")
    if a.wait:
        return _follow(c, jid, tail=not a.quiet)
    return 0


def _follow(c: FqClient, jid: int, tail: bool = True) -> int:
    """Block until the job finishes, streaming its log."""
    offset = 0
    last = None
    while True:
        job = c.get(jid)
        key = (job["state"], job.get("phase"))
        if key != last:
            print(_c(f"--- {job['state']}"
                     + (f" / {job['phase']}" if job.get("phase") else "")
                     + (f" on lane {job['lane']}" if job.get("lane") else "")
                     + " ---", "90"))
            last = key
        if tail and job.get("lane"):
            try:
                chunk = c.logs(jid, offset=offset)
                if chunk["data"]:
                    sys.stdout.write(chunk["data"])
                    sys.stdout.flush()
                offset = chunk["offset"]
            except FqError:
                pass
        if job["state"] in TERMINAL:
            if tail:
                try:
                    chunk = c.logs(jid, offset=offset)
                    if chunk["data"]:
                        sys.stdout.write(chunk["data"])
                except FqError:
                    pass
            print(_c(f"\njob {jid}: {job['state']} "
                     f"(exit_code={job.get('exit_code')})",
                     _STATE_COLOR.get(job["state"], "0")))
            if job.get("message"):
                print(f"  {job['message']}")
            return 0 if job["state"] == "DONE" else 1
        time.sleep(2)


def cmd_status(a: argparse.Namespace, c: FqClient) -> int:
    if a.job_id is not None:
        job = c.get(a.job_id)
        if a.json:
            print(json.dumps(job, indent=2))
            return 0
        for k in ("id", "user", "state", "phase", "lane", "priority",
                  "num_fpgas", "tree", "hw_config", "workload", "timeout_s",
                  "exit_code", "message", "blocked_reason", "workdir"):
            v = job.get(k)
            if v not in (None, ""):
                print(f"{k:>15}: {v}")
        print(f"{'submitted':>15}: {_age(job.get('submitted_at'))} ago")
        if job.get("started_at"):
            print(f"{'started':>15}: {_age(job.get('started_at'))} ago")
        evs = c.events(a.job_id)
        if evs:
            print("\n  events:")
            for e in evs:
                print(f"    {time.strftime('%H:%M:%S', time.localtime(e['ts']))}"
                      f"  {e['kind']:<20} {e.get('detail') or ''}")
        return 0

    jobs = c.list(user=a.user, all=a.all, limit=a.limit)
    if a.json:
        print(json.dumps(jobs, indent=2))
        return 0
    if not jobs:
        print("no jobs")
        return 0
    print(f"{'ID':>6}  {'STATE':<11} {'PHASE':<11} {'USER':<10} "
          f"{'LANE':<10} {'PRI':>3} {'FPGA':>4} {'AGE':>6}  HW / WHY")
    for j in sorted(jobs, key=lambda x: x["id"]):
        why = j.get("hw_config") or "-"
        if j["state"] == "QUEUED" and j.get("blocked_reason"):
            why = _c(j["blocked_reason"], "90")
        print(f"{j['id']:>6}  {_state(j['state'])} "
              f"{(j.get('phase') or '-'):<11} {j['user']:<10} "
              f"{(j.get('lane') or '-'):<10} {j['priority']:>3} "
              f"{j['num_fpgas']:>4} {_age(j['submitted_at']):>6}  {why}")
    return 0


def cmd_lanes(a: argparse.Namespace, c: FqClient) -> int:
    lanes = c.lanes()
    if a.json:
        print(json.dumps(lanes, indent=2))
        return 0
    print(f"{'LANE':<12} {'MODE':<8} {'CAP':>3} {'STATE':<12} {'JOB':>6}  "
          f"{'WARM (tree/hw)':<34} HOSTS")
    for ln in lanes:
        if ln["drain"]:
            st, col = "DRAIN", "33"
        elif not ln["enabled"]:
            st, col = "DISABLED", "90"
        elif ln["locked"]:
            st, col = "BUSY", "32"
        elif ln["disposition"] == "FOREIGN":
            st, col = "ORPHANED", "31"
        elif ln["disposition"] == "UNREACHABLE":
            st, col = "UNREACHABLE", "31"
        else:
            st, col = "FREE", "92"
        warm = f"{os.path.basename(ln.get('last_tree') or '-')}/" \
               f"{(ln.get('last_hw_config') or '-')}"
        print(f"{ln['name']:<12} {ln['mode']:<8} {ln['capacity']:>3} "
              f"{_c(f'{st:<12}', col)} {str(ln.get('job') or '-'):>6}  "
              f"{warm[:33]:<34} {','.join(ln['hosts'])}")
        if ln["disposition"] in ("FOREIGN", "UNREACHABLE") and ln.get(
                "probe_detail"):
            print(f"{'':<12} " + _c("  ! " + ln["probe_detail"], "31"))
            if ln["disposition"] == "FOREIGN":
                hint = f"    reclaim with: fq reclaim {ln['name']}"
                print(f"{'':<12} " + _c(hint, "90"))
    return 0


def cmd_cancel(a: argparse.Namespace, c: FqClient) -> int:
    for jid in a.job_id:
        try:
            r = c.cancel(jid)
            print(f"job {jid}: {r.get('note') or r.get('state')}")
        except FqError as exc:
            print(f"job {jid}: {exc}", file=sys.stderr)
    return 0


def cmd_done(a: argparse.Namespace, c: FqClient) -> int:
    print(c.signal_done(a.job_id).get("note", "ok"))
    return 0


def cmd_logs(a: argparse.Namespace, c: FqClient) -> int:
    offset = 0
    while True:
        chunk = c.logs(a.job_id, stream=a.stream, offset=offset)
        if chunk["data"]:
            sys.stdout.write(chunk["data"])
            sys.stdout.flush()
        offset = chunk["offset"]
        if not a.follow:
            return 0
        if chunk["eof"] and not chunk["data"]:
            return 0
        time.sleep(1.5)


def cmd_wait(a: argparse.Namespace, c: FqClient) -> int:
    return _follow(c, a.job_id, tail=not a.quiet)


def cmd_drain(a: argparse.Namespace, c: FqClient) -> int:
    r = c.drain(a.lane, on=not a.off)
    print(f"lane {r['lane']}: drain={r['drain']}")
    return 0


def cmd_reclaim(a: argparse.Namespace, c: FqClient) -> int:
    r = c._call("reclaim", lane=a.lane, dry_run=a.dry_run)
    print(json.dumps(r, indent=2))
    return 0 if (r.get("reclaimed") or a.dry_run) else 1


def cmd_ping(a: argparse.Namespace, c: FqClient) -> int:
    print(json.dumps(c.ping(), indent=2))
    return 0


def cmd_daemon(a: argparse.Namespace, _c: Any) -> int:
    from .config import load_pool, default_pool_path
    from .daemon import Daemon
    path = a.pool or default_pool_path()
    if not path:
        print("error: no pool config (use --pool, or set $FQ_POOL)",
              file=sys.stderr)
        return 2
    cfg = load_pool(path)
    if a.socket:
        import pathlib
        cfg.socket_path = pathlib.Path(a.socket)
    return Daemon(cfg, verbose=a.verbose).run()


def cmd_check(a: argparse.Namespace, _c: Any) -> int:
    """Validate a pool config without starting anything."""
    from .config import load_pool, ConfigError
    try:
        cfg = load_pool(a.pool)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    total = sum(l.capacity for l in cfg.lanes if l.enabled)
    print(f"OK: {len(cfg.lanes)} lane(s), {total} FPGA(s), "
          f"backend={cfg.backend}, reclaim={cfg.reclaim_policy}")
    for ln in cfg.lanes:
        print(f"  {ln.name:<12} mode={ln.mode:<7} cap={ln.capacity} "
              f"tag={ln.tag or '-'} hosts={','.join(ln.hosts) or '-'}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fq", description="Queue FireSim jobs onto a shared FPGA pool.")
    p.add_argument("--socket", default=None,
                   help="daemon socket (default: $FQ_SOCKET or a well-known path)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="enqueue a job")
    s.add_argument("--tree", required=True,
                   help="chipyard tree that BUILT the bitstream (required: "
                        "the driver must match the bitstream)")
    s.add_argument("--hw-config", dest="hw_config",
                   help="hwdb key in that tree's config_hwdb.yaml")
    s.add_argument("--agfi", help="explicit AGFI (needs --tree; cross-checked "
                                  "against hwdb when --hw-config is also given)")
    s.add_argument("--elf", help="guest ELF to boot")
    s.add_argument("--workload", help="FireSim workload name (default: per-job)")
    s.add_argument("--rootfs", help="rootfs image; omit for bare metal")
    s.add_argument("-n", "--num-fpgas", dest="num_fpgas", type=int, default=1)
    s.add_argument("-p", "--priority", type=int, default=5,
                   help="0 low / 5 normal / 10 high")
    s.add_argument("-t", "--timeout", type=int, default=0,
                   help="wall-clock backstop in seconds (0 = pool default)")
    s.add_argument("--completion", choices=("exit", "sentinel", "timeout"),
                   help="how the job ends (default: exit -- the guest "
                        "terminates the sim itself, e.g. Zephyr sys_reboot)")
    s.add_argument("--sentinel", help="uartlog regex that means success")
    s.add_argument("--fail-regex", dest="fail_regex",
                   help="uartlog regex that means failure")
    s.add_argument("--lane", help="pin to a specific lane")
    s.add_argument("--set", action="append", default=[], metavar="A.B=C",
                   help="patch a config_runtime field (repeatable)")
    s.add_argument("--results", help="directory to copy artifacts into")
    s.add_argument("--project")
    s.add_argument("--comment")
    s.add_argument("-w", "--wait", action="store_true",
                   help="block until the job finishes")
    s.add_argument("-q", "--quiet", action="store_true",
                   help="with --wait, do not stream the log")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("status", help="queue view, or one job")
    s.add_argument("job_id", nargs="?", type=int)
    s.add_argument("--user")
    s.add_argument("-a", "--all", action="store_true",
                   help="include finished jobs")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("list", help="alias for status")
    s.add_argument("job_id", nargs="?", type=int)
    s.add_argument("--user")
    s.add_argument("-a", "--all", action="store_true")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("lanes", help="show the FPGA pool")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_lanes)

    s = sub.add_parser("cancel", help="cancel job(s)")
    s.add_argument("job_id", type=int, nargs="+")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("signal-done",
                       help="tell a running job its work is finished")
    s.add_argument("job_id", type=int)
    s.set_defaults(func=cmd_done)

    s = sub.add_parser("logs", help="show a job's log")
    s.add_argument("job_id", type=int)
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("--stream", default="stdout",
                   choices=("stdout", "job", "status"))
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("wait", help="block until a job finishes")
    s.add_argument("job_id", type=int)
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_wait)

    s = sub.add_parser("drain", help="stop scheduling onto a lane (admin)")
    s.add_argument("lane")
    s.add_argument("--off", action="store_true", help="undrain instead")
    s.set_defaults(func=cmd_drain)

    s = sub.add_parser("reclaim",
                       help="kill an orphaned simulation holding a lane (admin)")
    s.add_argument("lane")
    s.add_argument("--dry-run", action="store_true",
                   help="report what would be killed, kill nothing")
    s.set_defaults(func=cmd_reclaim)

    s = sub.add_parser("ping", help="check the daemon")
    s.set_defaults(func=cmd_ping)

    s = sub.add_parser("daemon", help="run the daemon (blocking)")
    s.add_argument("--pool", help="pool config (default: $FQ_POOL)")
    s.add_argument("--socket")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_daemon, _no_client=True)

    s = sub.add_parser("check", help="validate a pool config and exit")
    s.add_argument("pool")
    s.set_defaults(func=cmd_check, _no_client=True)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "_no_client", False):
        return args.func(args, None)
    client = FqClient(args.socket)
    try:
        return args.func(args, client)
    except FqError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
