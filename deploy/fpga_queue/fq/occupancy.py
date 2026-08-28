"""Occupancy probing and orphan reclamation.

The problem this solves
-----------------------
On this pool, **"the instance is up" tells you nothing about whether the FPGA
is usable.**  The steady state of the eight f2.6xlarge hosts is that every one
of them has a live ``FireSim-f2`` process left over from an earlier run: a
bare-metal guest prints its output and then idles forever, `runworkload` never
sees a completion, and `terminate_on_completion: false` leaves the simulation
attached to the FPGA indefinitely.  Eight instances "running" in EC2 currently
means eight FPGAs *occupied*.

So a lane has three distinct dispositions, and the daemon must tell them apart:

  ``FREE``       nothing running on the host; the lane's flock is available.
  ``OURS``       the lane's flock is held by one of our runners.  Whatever is
                 running belongs to a job we are tracking.
  ``FOREIGN``    the flock is free but a simulation is running anyway.  Someone
                 started it outside the queue, or it outlived a daemon that no
                 longer exists.  This is the orphan case.
  ``UNREACHABLE`` the probe failed (ssh down, host wedged).  Treated as
                 unusable, never as free.

Why the flock is checked first
------------------------------
``FOREIGN`` is defined as *sim running AND flock free*.  That ordering is the
whole safety argument for reclamation: if any fq job holds the lane, the sim on
it is that job's, and we must never touch it.  Only when the kernel tells us
nobody owns the lane can a running process be an orphan.  Note this is
conservative in the right direction -- a race where a runner takes the flock
between our probe and our reclaim is handled by taking the flock *ourselves*
before reclaiming, so reclamation and dispatch can never overlap.

Reclamation policy (``reclaim.policy`` in the pool config)
----------------------------------------------------------
``manual`` (default)
    Detect, report loudly, refuse to schedule on the lane, and wait for an
    operator to run ``fq reclaim <lane>``.  This is the default because
    killing a simulation is destructive and irreversible: the sim may be a
    colleague's hand-driven run that simply predates the queue -- which is
    exactly the situation on this pool today.  Silence is not safe here, but
    neither is autonomy; the daemon's job is to make the problem visible.
``auto``
    Reclaim a FOREIGN lane once it has been continuously foreign for
    ``reclaim.grace_s`` (default 30 min).  The grace period exists so that a
    sim someone started thirty seconds ago is not killed by a daemon that
    happened to boot.  Use this once the pool is genuinely queue-owned.
``never``
    Never reclaim, never even offer to.  For a pool shared with humans who
    are not queue users.

Reclaiming means, in order: take the lane flock (so no dispatch can race),
re-probe to confirm it is still foreign, then run the backend's reclaim
command (``firesim kill``-equivalent -- on F2 a host-wide
``pkill -SIGKILL FireSim-f2``), then re-probe to confirm it worked.
"""

from __future__ import annotations

import dataclasses
import enum
import shlex
import subprocess
import time
from typing import Any, Optional


class LaneDisposition(str, enum.Enum):
    FREE = "FREE"
    OURS = "OURS"
    FOREIGN = "FOREIGN"
    UNREACHABLE = "UNREACHABLE"


@dataclasses.dataclass
class ProbeResult:
    lane: str
    disposition: LaneDisposition
    sims_running: int = 0
    detail: str = ""
    checked_at: float = dataclasses.field(default_factory=time.time)
    per_host: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def schedulable(self) -> bool:
        return self.disposition is LaneDisposition.FREE

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["disposition"] = self.disposition.value
        d["schedulable"] = self.schedulable
        return d


# The default probe: count live simulator processes on the host.  `pgrep -c`
# exits 1 with "0" when there is no match, which is why the command ends with
# `|| true` -- an absent process is a valid answer, not an error.
DEFAULT_PROBE_TEMPLATE = (
    "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    "-o ConnectTimeout={connect_timeout} -i {key} {user}@{host} "
    "'pgrep -c {driver} || true'"
)

DEFAULT_RECLAIM_TEMPLATE = (
    "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    "-o ConnectTimeout={connect_timeout} -i {key} {user}@{host} "
    "'sudo pkill -SIGKILL {driver} || true; sleep 2; pgrep -c {driver} || true'"
)


class Prober:
    """Runs the occupancy probe for a lane's hosts.

    Every command is a plain shell string built from a template, so an
    operator can reproduce exactly what the daemon saw by copy-pasting it,
    and tests can substitute a template that touches local files instead of
    talking to EC2.
    """

    def __init__(self, options: Optional[dict[str, Any]] = None):
        o = dict(options or {})
        self.enabled: bool = bool(o.get("enabled", True))
        self.template: str = o.get("template") or DEFAULT_PROBE_TEMPLATE
        self.reclaim_template: str = (
            o.get("reclaim_template") or DEFAULT_RECLAIM_TEMPLATE)
        self.driver: str = o.get("driver", "FireSim-f2")
        self.key: str = o.get("key", "~/firesim.pem")
        self.user: str = o.get("user", "ubuntu")
        self.connect_timeout: int = int(o.get("connect_timeout", 10))
        self.timeout: float = float(o.get("timeout", 30.0))

    def _fmt(self, template: str, host: str) -> str:
        return template.format(
            host=host,
            key=self.key,
            user=self.user,
            driver=self.driver,
            connect_timeout=self.connect_timeout,
        )

    def _run(self, cmd: str) -> tuple[int, str]:
        try:
            cp = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=self.timeout)
            return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, "probe timed out"
        except OSError as exc:
            return 127, f"probe failed to launch: {exc}"

    @staticmethod
    def _parse_count(out: str) -> Optional[int]:
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.isdigit():
                return int(line)
        return None

    def probe_hosts(self, hosts: tuple[str, ...]) -> tuple[int, dict, list[str]]:
        """Return (total_sims, per_host detail, unreachable hosts)."""
        total = 0
        per_host: dict[str, Any] = {}
        unreachable: list[str] = []
        for h in hosts:
            cmd = self._fmt(self.template, h)
            rc, out = self._run(cmd)
            n = self._parse_count(out)
            if n is None:
                unreachable.append(h)
                per_host[h] = {"error": out.strip()[:400], "rc": rc}
                continue
            total += n
            per_host[h] = {"sims": n}
        return total, per_host, unreachable

    def probe_lane(self, lane_name: str, hosts: tuple[str, ...],
                   lock_free: bool) -> ProbeResult:
        """Classify a lane.  `lock_free` must come from a real flock probe."""
        if not lock_free:
            return ProbeResult(lane_name, LaneDisposition.OURS,
                               detail="lane flock is held by an fq runner")
        if not self.enabled or not hosts:
            # No probe configured: trust the flock alone.  Documented as a
            # weaker guarantee -- an orphan sim will not be noticed.
            return ProbeResult(lane_name, LaneDisposition.FREE,
                               detail="probing disabled")
        total, per_host, unreachable = self.probe_hosts(hosts)
        if unreachable:
            return ProbeResult(
                lane_name, LaneDisposition.UNREACHABLE,
                sims_running=total, per_host=per_host,
                detail=f"unreachable host(s): {', '.join(unreachable)}")
        if total > 0:
            return ProbeResult(
                lane_name, LaneDisposition.FOREIGN, sims_running=total,
                per_host=per_host,
                detail=(f"{total} live {self.driver} process(es) but no fq job "
                        f"owns this lane -- orphaned simulation"))
        return ProbeResult(lane_name, LaneDisposition.FREE, per_host=per_host)

    def reclaim_lane(self, hosts: tuple[str, ...]) -> tuple[bool, str]:
        """Kill orphaned simulators on these hosts.  Returns (ok, detail).

        Callers MUST already hold the lane's flock -- see the module docstring.
        """
        msgs = []
        ok = True
        for h in hosts:
            cmd = self._fmt(self.reclaim_template, h)
            rc, out = self._run(cmd)
            remaining = self._parse_count(out)
            if remaining is None:
                ok = False
                msgs.append(f"{h}: reclaim command failed (rc={rc}): "
                            f"{out.strip()[:200]}")
            elif remaining > 0:
                ok = False
                msgs.append(f"{h}: {remaining} sim(s) still alive after kill")
            else:
                msgs.append(f"{h}: clear")
        return ok, "; ".join(msgs)


def should_auto_reclaim(policy: str, foreign_since: Optional[float],
                        grace_s: float, now: float) -> bool:
    """Pure decision function for the auto policy (unit-tested directly)."""
    if policy != "auto":
        return False
    if foreign_since is None:
        return False
    return (now - foreign_since) >= grace_s
