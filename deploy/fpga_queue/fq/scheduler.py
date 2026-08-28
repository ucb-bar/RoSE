"""Scheduling policy.

This module is a pure function of (queued jobs, lane states, clock) -> list of
placements.  It touches no files, no sockets and no database, which is what
makes the policy testable without hardware and reviewable without tracing
through the daemon.

Policy, in one screen
---------------------
1.  **Order.**  Queued jobs are sorted by

        (-priority, fpgas_that_user_is_already_running, submitted_at, id)

    Priority is the coarse control (10 high / 5 normal / 0 background).
    The second key is the fairness term: a user who already holds FPGAs
    yields to a user who holds none *within the same priority tier*.  It is
    deliberately "current usage", not "historical usage": we are sharing a
    small pool of expensive machines right now, and past consumption should
    not let someone monopolise the present.  (firesim-queue's single-FPGA
    round-robin by ``last_user_served`` does not generalise to a pool; this
    replaces it.)

2.  **Placement, with backfill.**  Walk that order and give each job the best
    free lane that can host it.  A job whose lane is busy does *not* block the
    jobs behind it -- that is requirement "no head-of-line blocking".

3.  **Reservation, to stop starvation.**  Pure backfill starves wide jobs: a
    4-FPGA job can wait forever behind a stream of 1-FPGA jobs.  So the first
    job in the order that (a) cannot be placed now and (b) has waited longer
    than ``reserve_after_s`` becomes the *reservation holder*.  From that
    point on, lanes it could eventually use are withheld from jobs later in
    the order.  This is textbook conservative backfill with a single
    reservation -- one reservation rather than many, because one is enough to
    bound the wait and is far easier to reason about when something is stuck.

4.  **Lane choice.**  Among lanes that can host the job, prefer in order:
      a. lanes already warm for this exact (tree, hw_config) -- FireSim's
         ``infrasetup`` re-elaborates and rebuilds the driver for a config it
         has not seen, which costs tens of minutes.  Reusing a warm lane is
         the single largest wall-clock win available to this scheduler.
      b. lanes warm for the same tree,
      c. the smallest lane that still fits (best fit -- do not burn an
         8-FPGA lane on a 1-FPGA job while a wide job is waiting),
      d. lane name, for determinism.

Everything here is deterministic: same inputs, same output.  That matters
when you have to explain to a colleague why their job did not start.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional


@dataclasses.dataclass(frozen=True)
class JobView:
    """The scheduler's view of a queued job."""

    id: int
    user: str
    priority: int
    submitted_at: float
    num_fpgas: int = 1
    tree: Optional[str] = None
    hw_config: Optional[str] = None
    # Lane the submitter explicitly pinned to, if any.
    lane_hint: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class LaneView:
    """The scheduler's view of a lane."""

    name: str
    capacity: int = 1
    free: bool = True
    enabled: bool = True
    drain: bool = False
    trees: tuple[str, ...] = ()
    last_tree: Optional[str] = None
    last_hw_config: Optional[str] = None

    def usable(self) -> bool:
        return self.enabled and not self.drain

    def accepts_tree(self, tree: Optional[str]) -> bool:
        if not self.trees:
            return True
        if tree is None:
            return False
        return tree in self.trees


@dataclasses.dataclass(frozen=True)
class Placement:
    job_id: int
    lane: str
    # Why this lane -- surfaced by `fq status` so a scheduling decision is
    # always explainable after the fact.
    reason: str


@dataclasses.dataclass(frozen=True)
class Plan:
    placements: tuple[Placement, ...]
    reserved_job_id: Optional[int] = None
    # job_id -> human-readable reason it is still queued.
    blocked: dict[int, str] = dataclasses.field(default_factory=dict)


def _can_host(lane: LaneView, job: JobView) -> bool:
    """Could this lane *ever* run this job (ignoring current freeness)?"""
    if not lane.usable():
        return False
    if lane.capacity < job.num_fpgas:
        return False
    if not lane.accepts_tree(job.tree):
        return False
    if job.lane_hint and lane.name != job.lane_hint:
        return False
    return True


def _lane_rank(lane: LaneView, job: JobView) -> tuple:
    """Sort key for choosing among candidate lanes.  Lower is better."""
    exact_warm = (lane.last_tree is not None
                  and lane.last_tree == job.tree
                  and lane.last_hw_config is not None
                  and lane.last_hw_config == job.hw_config)
    tree_warm = lane.last_tree is not None and lane.last_tree == job.tree
    return (
        0 if exact_warm else 1,
        0 if tree_warm else 1,
        lane.capacity,          # best fit
        lane.name,
    )


def _lane_reason(lane: LaneView, job: JobView) -> str:
    if lane.last_tree == job.tree and lane.last_hw_config == job.hw_config:
        return "warm: same tree + hw_config (infrasetup should be fast)"
    if lane.last_tree == job.tree:
        return "warm tree, different hw_config (infrasetup will rebuild)"
    if lane.last_tree is None:
        return "cold lane (never used)"
    return "cold: different tree (full infrasetup expected)"


def order_jobs(jobs: Iterable[JobView],
               running_fpgas_by_user: dict[str, int]) -> list[JobView]:
    """Canonical queue order.  See rule 1 in the module docstring."""
    return sorted(
        jobs,
        key=lambda j: (
            -j.priority,
            running_fpgas_by_user.get(j.user, 0),
            j.submitted_at,
            j.id,
        ),
    )


def plan(jobs: Iterable[JobView],
         lanes: Iterable[LaneView],
         now: float,
         running_fpgas_by_user: Optional[dict[str, int]] = None,
         reserve_after_s: float = 900.0,
         max_fpgas_per_user: int = 0) -> Plan:
    """Compute lane assignments.  Pure; safe to call in tests and dry-runs."""
    running_fpgas_by_user = dict(running_fpgas_by_user or {})
    lane_list = list(lanes)
    ordered = order_jobs(jobs, running_fpgas_by_user)

    free_lanes = {ln.name: ln for ln in lane_list if ln.free and ln.usable()}
    placements: list[Placement] = []
    blocked: dict[int, str] = {}
    reserved_job: Optional[JobView] = None
    reserved_lane_names: set[str] = set()
    # Projected usage as we place jobs in this pass, for the per-user cap.
    projected = dict(running_fpgas_by_user)

    for job in ordered:
        # --- hard admission checks -------------------------------------
        hostable = [ln for ln in lane_list if _can_host(ln, job)]
        if not hostable:
            blocked[job.id] = (
                f"no lane can host it (needs {job.num_fpgas} FPGA(s)"
                + (f", tree {job.tree}" if job.tree else "")
                + (f", pinned to lane {job.lane_hint}" if job.lane_hint else "")
                + ")")
            continue

        if max_fpgas_per_user:
            if projected.get(job.user, 0) + job.num_fpgas > max_fpgas_per_user:
                blocked[job.id] = (
                    f"user {job.user} would exceed max_fpgas_per_user="
                    f"{max_fpgas_per_user}")
                continue

        # --- candidate free lanes --------------------------------------
        cands = [ln for ln in hostable if ln.name in free_lanes]
        if reserved_job is not None:
            # A reservation is active and this job is behind it: it may only
            # backfill onto lanes the reserved job cannot use.
            cands = [ln for ln in cands if ln.name not in reserved_lane_names]

        if not cands:
            waited = now - job.submitted_at
            if reserved_job is None and waited >= reserve_after_s:
                reserved_job = job
                reserved_lane_names = {ln.name for ln in hostable}
                blocked[job.id] = (
                    f"waiting {waited:.0f}s for a lane; RESERVED -- later jobs "
                    f"will not be given lanes it needs")
            elif reserved_job is not None and reserved_job.id == job.id:
                blocked[job.id] = "reserved, waiting for its lane to free"
            else:
                if reserved_job is not None:
                    blocked[job.id] = (
                        f"free lanes are reserved for job {reserved_job.id}")
                else:
                    blocked[job.id] = "all matching lanes busy"
            continue

        best = min(cands, key=lambda ln: _lane_rank(ln, job))
        placements.append(Placement(job.id, best.name, _lane_reason(best, job)))
        del free_lanes[best.name]
        projected[job.user] = projected.get(job.user, 0) + job.num_fpgas

    return Plan(
        placements=tuple(placements),
        reserved_job_id=reserved_job.id if reserved_job else None,
        blocked=blocked,
    )
