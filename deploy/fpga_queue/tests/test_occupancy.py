"""Occupancy probing and orphan reclamation."""

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq.backends.mock import mock_probe_templates
from fq.occupancy import (LaneDisposition, Prober, should_auto_reclaim)


class TestProber(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="fqocc-"))
        (self.root / "hosts" / "h0").mkdir(parents=True)
        self.prober = Prober({**mock_probe_templates(str(self.root)),
                              "enabled": True})
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.kill()
            p.wait(timeout=5)
        shutil.rmtree(self.root, ignore_errors=True)

    def _start_sim(self, host="h0", slot=0):
        d = self.root / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(["sleep", "300"])
        self.procs.append(p)
        (d / f"sim_{slot}.pid").write_text(str(p.pid))
        return p

    def test_idle_host_is_free(self):
        r = self.prober.probe_lane("l0", ("h0",), lock_free=True)
        self.assertIs(r.disposition, LaneDisposition.FREE)
        self.assertTrue(r.schedulable)

    def test_held_lock_means_ours_without_probing(self):
        self._start_sim()
        r = self.prober.probe_lane("l0", ("h0",), lock_free=False)
        self.assertIs(r.disposition, LaneDisposition.OURS)
        self.assertFalse(r.schedulable)

    def test_running_sim_with_free_lock_is_foreign(self):
        """The orphan case: EC2 says the instance is up, but the FPGA is
        occupied by a sim nobody owns."""
        self._start_sim()
        r = self.prober.probe_lane("l0", ("h0",), lock_free=True)
        self.assertIs(r.disposition, LaneDisposition.FOREIGN)
        self.assertEqual(r.sims_running, 1)
        self.assertFalse(r.schedulable)
        self.assertIn("orphaned", r.detail)

    def test_unreachable_host_is_never_treated_as_free(self):
        prober = Prober({"enabled": True,
                         "template": "bash -c 'exit 255'",
                         "reclaim_template": "true"})
        r = prober.probe_lane("l0", ("gone",), lock_free=True)
        self.assertIs(r.disposition, LaneDisposition.UNREACHABLE)
        self.assertFalse(r.schedulable)

    def test_probing_disabled_falls_back_to_the_lock(self):
        p = Prober({"enabled": False})
        self.assertIs(p.probe_lane("l0", ("h0",), True).disposition,
                      LaneDisposition.FREE)
        self.assertIs(p.probe_lane("l0", ("h0",), False).disposition,
                      LaneDisposition.OURS)

    def test_reclaim_kills_the_orphan(self):
        p = self._start_sim()
        ok, detail = self.prober.reclaim_lane(("h0",))
        self.assertTrue(ok, detail)
        p.wait(timeout=10)
        self.assertIsNotNone(p.poll())
        r = self.prober.probe_lane("l0", ("h0",), lock_free=True)
        self.assertIs(r.disposition, LaneDisposition.FREE)

    def test_multi_host_lane_sums_occupancy(self):
        (self.root / "hosts" / "h1").mkdir(parents=True, exist_ok=True)
        self._start_sim("h0")
        self._start_sim("h1")
        r = self.prober.probe_lane("l0", ("h0", "h1"), lock_free=True)
        self.assertEqual(r.sims_running, 2)


class TestAutoReclaimDecision(unittest.TestCase):
    def test_manual_policy_never_auto_reclaims(self):
        self.assertFalse(should_auto_reclaim("manual", 0, 60, 1e9))

    def test_never_policy(self):
        self.assertFalse(should_auto_reclaim("never", 0, 60, 1e9))

    def test_auto_respects_the_grace_period(self):
        """A sim started thirty seconds ago must not be killed by a daemon
        that happened to boot."""
        self.assertFalse(should_auto_reclaim("auto", 1000.0, 1800, 1030.0))
        self.assertTrue(should_auto_reclaim("auto", 1000.0, 1800, 2900.0))

    def test_auto_needs_a_foreign_since_timestamp(self):
        self.assertFalse(should_auto_reclaim("auto", None, 60, 1e9))


if __name__ == "__main__":
    unittest.main()
