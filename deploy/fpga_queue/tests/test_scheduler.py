"""Scheduling policy — pure, no daemon, no hardware."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq.scheduler import JobView, LaneView, Plan, order_jobs, plan


def J(i, **kw):
    d = dict(user="alice", priority=5, submitted_at=1000.0 + i)
    d.update(kw)
    return JobView(id=i, **d)


def L(name, **kw):
    d = dict(capacity=1, free=True)
    d.update(kw)
    return LaneView(name=name, **d)


class TestOrdering(unittest.TestCase):
    def test_priority_beats_arrival(self):
        jobs = [J(1, priority=0), J(2, priority=10), J(3, priority=5)]
        self.assertEqual([j.id for j in order_jobs(jobs, {})], [2, 3, 1])

    def test_fifo_within_a_tier(self):
        jobs = [J(3, submitted_at=300), J(1, submitted_at=100),
                J(2, submitted_at=200)]
        self.assertEqual([j.id for j in order_jobs(jobs, {})], [1, 2, 3])

    def test_fairness_yields_to_a_user_holding_nothing(self):
        # alice already holds 2 FPGAs and submitted first; bob holds none.
        # Within the tier bob goes first.
        jobs = [J(1, user="alice", submitted_at=100),
                J(2, user="bob", submitted_at=200)]
        got = order_jobs(jobs, {"alice": 2})
        self.assertEqual([j.id for j in got], [2, 1])

    def test_fairness_never_overrides_priority(self):
        jobs = [J(1, user="alice", priority=10, submitted_at=100),
                J(2, user="bob", priority=5, submitted_at=90)]
        self.assertEqual([j.id for j in order_jobs(jobs, {"alice": 8})], [1, 2])


class TestPlacement(unittest.TestCase):
    def test_places_in_priority_order(self):
        p = plan([J(1, priority=0), J(2, priority=10)],
                 [L("a")], now=2000)
        self.assertEqual([(x.job_id, x.lane) for x in p.placements],
                         [(2, "a")])
        self.assertIn(1, p.blocked)

    def test_no_head_of_line_blocking(self):
        """A job whose lane is busy must not block the job behind it."""
        jobs = [J(1, num_fpgas=4), J(2, num_fpgas=1)]
        lanes = [L("wide", capacity=4, free=False), L("small", capacity=1)]
        p = plan(jobs, lanes, now=1001)   # job 1 has waited ~1s
        self.assertEqual([(x.job_id, x.lane) for x in p.placements],
                         [(2, "small")])
        self.assertIsNone(p.reserved_job_id)

    def test_reservation_stops_starvation_of_a_wide_job(self):
        """Once the wide job has waited long enough it holds its lanes."""
        jobs = [J(1, num_fpgas=4, submitted_at=0), J(2, num_fpgas=1)]
        # The wide lane is free but too small... no: make it busy-free mix.
        lanes = [L("wide", capacity=4, free=True), L("small", capacity=1,
                                                     free=False)]
        # job 1 fits on "wide"; it should simply be placed.
        p = plan(jobs, lanes, now=10_000, reserve_after_s=900)
        self.assertEqual(p.placements[0].job_id, 1)

        # Now make "wide" busy so job 1 cannot run, and let it age past the
        # reservation threshold. Job 2 must NOT be given "wide" when it frees.
        lanes = [L("wide", capacity=4, free=False), L("small", capacity=1,
                                                      free=False)]
        p = plan(jobs, lanes, now=10_000, reserve_after_s=900)
        self.assertEqual(p.reserved_job_id, 1)

        # "wide" frees. Job 2 could fit there, but it is reserved for job 1.
        lanes = [L("wide", capacity=4, free=True), L("small", capacity=1,
                                                     free=False)]
        jobs = [J(1, num_fpgas=4, submitted_at=0),
                J(2, num_fpgas=1, submitted_at=9999)]
        p = plan(jobs, lanes, now=10_000, reserve_after_s=900)
        self.assertEqual([(x.job_id, x.lane) for x in p.placements],
                         [(1, "wide")])

    def test_backfill_onto_lanes_the_reserved_job_cannot_use(self):
        """Reservation must not stop unrelated work."""
        jobs = [J(1, num_fpgas=4, submitted_at=0),
                J(2, num_fpgas=1, submitted_at=9999)]
        lanes = [L("wide", capacity=4, free=False),
                 L("tiny", capacity=1, free=True)]
        p = plan(jobs, lanes, now=10_000, reserve_after_s=900)
        self.assertEqual(p.reserved_job_id, 1)
        # job 2 backfills onto "tiny", which job 1 can never use (cap 1 < 4).
        self.assertEqual([(x.job_id, x.lane) for x in p.placements],
                         [(2, "tiny")])

    def test_tree_restriction_is_honoured(self):
        jobs = [J(1, tree="/trees/rose")]
        lanes = [L("a", trees=("/trees/other",))]
        p = plan(jobs, lanes, now=2000)
        self.assertEqual(p.placements, ())
        self.assertIn("no lane can host it", p.blocked[1])

    def test_lane_hint_pins(self):
        jobs = [J(1, lane_hint="b")]
        p = plan(jobs, [L("a"), L("b")], now=2000)
        self.assertEqual(p.placements[0].lane, "b")

    def test_prefers_warm_lane(self):
        """A lane already programmed for this (tree, hw) is worth a lot:
        a cold infrasetup costs tens of minutes."""
        jobs = [J(1, tree="/t", hw_config="cfgA")]
        lanes = [L("cold"), L("warm", last_tree="/t", last_hw_config="cfgA")]
        p = plan(jobs, lanes, now=2000)
        self.assertEqual(p.placements[0].lane, "warm")
        self.assertIn("warm", p.placements[0].reason)

    def test_best_fit_leaves_wide_lanes_for_wide_jobs(self):
        jobs = [J(1, num_fpgas=1)]
        lanes = [L("big", capacity=8), L("small", capacity=1)]
        self.assertEqual(plan(jobs, lanes, now=2000).placements[0].lane,
                         "small")

    def test_drained_and_disabled_lanes_are_skipped(self):
        jobs = [J(1)]
        self.assertEqual(plan(jobs, [L("a", drain=True)], now=2000).placements,
                         ())
        self.assertEqual(plan(jobs, [L("a", enabled=False)],
                              now=2000).placements, ())

    def test_per_user_fpga_cap(self):
        jobs = [J(1, user="alice"), J(2, user="alice"), J(3, user="bob")]
        lanes = [L("a"), L("b"), L("c")]
        p = plan(jobs, lanes, now=2000, max_fpgas_per_user=1)
        placed = {x.job_id for x in p.placements}
        self.assertEqual(placed, {1, 3})
        self.assertIn("max_fpgas_per_user", p.blocked[2])

    def test_one_lane_is_never_given_to_two_jobs(self):
        jobs = [J(1), J(2), J(3)]
        p = plan(jobs, [L("only")], now=2000)
        self.assertEqual(len(p.placements), 1)
        lanes_used = [x.lane for x in p.placements]
        self.assertEqual(len(lanes_used), len(set(lanes_used)))

    def test_determinism(self):
        jobs = [J(i) for i in range(1, 6)]
        lanes = [L(f"l{i}") for i in range(4)]
        a = plan(jobs, lanes, now=2000)
        b = plan(jobs, lanes, now=2000)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
