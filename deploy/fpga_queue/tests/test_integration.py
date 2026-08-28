"""End-to-end: a real daemon, a real socket, a fake FPGA pool.

Covers the scenarios that matter for expensive, slow, stateful hardware:
concurrent submission, slot exclusivity, crash recovery, timeout enforcement,
priority ordering, cancellation, and orphan reclamation.
"""

import os
import pathlib
import signal
import sys
import threading
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq.client import FqError
from tests.harness import Harness

# A job that never ends on its own -- the bare-metal-guest case.
NEVER = {"never_exit": True}


class Base(unittest.TestCase):
    lanes = 2
    kw: dict = {}

    def setUp(self):
        self.h = Harness(lanes=self.lanes, **self.kw)
        self.h.start()
        self.c = self.h.client()

    def tearDown(self):
        self.h.cleanup()


# ---------------------------------------------------------------------------
class TestBasics(Base):
    def test_ping(self):
        info = self.c.ping()
        self.assertEqual(info["backend"], "mock")
        self.assertEqual(info["lanes"], 2)

    def test_job_runs_to_completion(self):
        jid = self.c.submit(self.h.spec(mock={"run_s": 1}))
        job = self.h.wait_terminal(jid)
        self.assertEqual(job["state"], "DONE", job.get("message"))
        self.assertEqual(job["exit_code"], 0)

    def test_lane_is_released_after_success(self):
        jid = self.c.submit(self.h.spec(mock={"run_s": 1}))
        self.h.wait_terminal(jid)
        lanes = self.h.wait_for(
            lambda: (lambda ls: ls if all(not l["locked"] for l in ls)
                     else None)(self.c.lanes()),
            what="all lanes released")
        self.assertTrue(all(not l["locked"] for l in lanes))
        # And the fake simulator process is gone -- teardown really ran.
        self.assertEqual(self.h.total_sims(), 0)

    def test_failed_job_still_releases_its_lane(self):
        jid = self.c.submit(self.h.spec(mock={"fail_phase": "INFRASETUP"}))
        job = self.h.wait_terminal(jid)
        self.assertEqual(job["state"], "FAILED")
        self.h.wait_for(lambda: all(not l["locked"] for l in self.c.lanes()),
                        what="lane freed after failure")

    def test_invalid_spec_is_rejected_at_submit(self):
        with self.assertRaises(FqError) as cm:
            self.c.submit({"tree": "", "hw_config": "x"})
        self.assertIn("tree", str(cm.exception))

    def test_job_too_wide_for_the_pool_is_rejected_immediately(self):
        """Fail at the submitting terminal, not after an hour in the queue."""
        with self.assertRaises(FqError) as cm:
            self.c.submit(self.h.spec(num_fpgas=99))
        self.assertIn("no lane in this pool can ever host", str(cm.exception))

    def test_results_are_collected(self):
        dest = self.h.dir / "myresults"
        jid = self.c.submit(self.h.spec(
            results_dir=str(dest),
            mock={"run_s": 1, "uart": ["Hello World! chipyard_riscv64"]}))
        self.h.wait_terminal(jid)
        self.assertTrue((dest / "uartlog").exists(), list(dest.iterdir())
                        if dest.exists() else "no results dir")
        self.assertIn("Hello World", (dest / "uartlog").read_text())
        self.assertTrue((dest / "job.log").exists())


# ---------------------------------------------------------------------------
class TestExclusivity(Base):
    lanes = 1

    def test_only_one_job_runs_on_a_lane_at_a_time(self):
        ids = [self.c.submit(self.h.spec(mock=NEVER, timeout_s=6))
               for _ in range(3)]
        # Sample the pool repeatedly; more than one live sim would mean two
        # jobs were dispatched onto the same FPGA.
        peak = 0
        deadline = time.time() + 12
        while time.time() < deadline:
            running = [j for j in self.c.list() if j["state"] == "RUNNING"]
            peak = max(peak, len(running), self.h.total_sims())
            self.assertLessEqual(len(running), 1,
                                 f"two jobs running at once: {running}")
            self.assertLessEqual(self.h.total_sims(), 1,
                                 "two simulations on one FPGA")
            if all(self.c.get(i)["state"] in
                   ("DONE", "FAILED", "TIMEOUT", "CANCELLED") for i in ids):
                break
            time.sleep(0.2)
        self.assertEqual(peak, 1)
        for i in ids:
            self.assertEqual(self.h.wait_terminal(i, timeout=40)["state"],
                             "TIMEOUT")

    def test_lane_lock_blocks_an_outsider(self):
        """Even a process that bypasses the queue is blocked by the flock."""
        from fq.locks import LaneLock
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=8))
        self.h.wait_state(jid, ("DISPATCHING", "RUNNING"))
        self.h.wait_for(
            lambda: not LaneLock(self.h.state / "lanes", "lane0").is_free(),
            what="lane0 locked")
        outsider = LaneLock(self.h.state / "lanes", "lane0")
        self.assertFalse(outsider.acquire(job_id=999, user="rogue"))
        self.c.cancel(jid)
        self.h.wait_terminal(jid)


# ---------------------------------------------------------------------------
class TestConcurrentSubmission(Base):
    lanes = 4

    def test_many_independent_submitters_do_not_corrupt_state(self):
        """Simulates unrelated processes submitting at the same moment."""
        n = 24
        ids: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def submit(k):
            try:
                c = self.h.client()          # its own connection, as a real
                jid = c.submit(self.h.spec(  # separate process would have
                    priority=k % 3 * 5, mock={"run_s": 0.2}))
                with lock:
                    ids.append(jid)
            except Exception as exc:         # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=submit, args=(k,))
                   for k in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(ids), n)
        self.assertEqual(len(set(ids)), n, "job ids must be unique")

        for jid in ids:
            self.assertEqual(self.h.wait_terminal(jid, timeout=120)["state"],
                             "DONE")
        # Every lane freed, every simulation reaped.
        self.assertEqual(self.h.total_sims(), 0)

    def test_no_lane_is_ever_double_booked_under_load(self):
        ids = [self.c.submit(self.h.spec(mock={"run_s": 0.4}))
               for _ in range(16)]
        deadline = time.time() + 90
        while time.time() < deadline:
            active = [j for j in self.c.list()
                      if j["state"] in ("DISPATCHING", "RUNNING")]
            lanes = [j["lane"] for j in active if j["lane"]]
            self.assertEqual(len(lanes), len(set(lanes)),
                             f"a lane was double-booked: {active}")
            self.assertLessEqual(self.h.total_sims(), self.lanes)
            if all(self.c.get(i)["state"] not in
                   ("QUEUED", "DISPATCHING", "RUNNING") for i in ids):
                break
            time.sleep(0.15)
        for i in ids:
            self.assertEqual(self.h.wait_terminal(i, timeout=60)["state"],
                             "DONE")


# ---------------------------------------------------------------------------
class TestPriority(Base):
    lanes = 1

    def test_high_priority_job_runs_first(self):
        # Fill the only lane so everything else must queue behind it.
        blocker = self.c.submit(self.h.spec(mock=NEVER, timeout_s=4))
        self.h.wait_state(blocker, ("DISPATCHING", "RUNNING"))

        low = self.c.submit(self.h.spec(priority=0, mock={"run_s": 0.3}))
        time.sleep(0.3)
        high = self.c.submit(self.h.spec(priority=10, mock={"run_s": 0.3}))
        normal = self.c.submit(self.h.spec(priority=5, mock={"run_s": 0.3}))

        self.h.wait_terminal(blocker, timeout=40)
        for jid in (high, normal, low):
            self.h.wait_terminal(jid, timeout=60)

        order = sorted(
            [self.c.get(j) for j in (low, high, normal)],
            key=lambda j: j["started_at"])
        self.assertEqual([j["id"] for j in order], [high, normal, low],
                         "jobs must start in priority order, "
                         "not submission order")


# ---------------------------------------------------------------------------
class TestNoHeadOfLineBlocking(Base):
    """A queued job that cannot fit must not stall jobs that can."""
    lanes = 2
    kw = {"lane_specs": [
        {"name": "wide", "mode": "hosts", "hosts": ["hw"], "capacity": 2},
        {"name": "narrow", "mode": "hosts", "hosts": ["hn"], "capacity": 1},
    ]}

    def test_narrow_job_runs_while_wide_job_waits(self):
        # Occupy the only 2-FPGA lane.
        hog = self.c.submit(self.h.spec(num_fpgas=2, mock=NEVER, timeout_s=8))
        self.h.wait_state(hog, ("DISPATCHING", "RUNNING"))

        waiter = self.c.submit(self.h.spec(num_fpgas=2, mock={"run_s": 0.3}))
        runner = self.c.submit(self.h.spec(num_fpgas=1, mock={"run_s": 0.3}))

        # `runner` must finish long before `waiter` even starts.
        job = self.h.wait_terminal(runner, timeout=40)
        self.assertEqual(job["state"], "DONE")
        self.assertEqual(job["lane"], "narrow")
        self.assertEqual(self.c.get(waiter)["state"], "QUEUED")
        self.assertIn("busy", (self.c.get(waiter)["blocked_reason"] or "")
                      + (self.c.get(waiter)["blocked_reason"] or ""))

        self.h.wait_terminal(hog, timeout=40)
        self.assertEqual(self.h.wait_terminal(waiter, timeout=60)["state"],
                         "DONE")


# ---------------------------------------------------------------------------
class TestTimeout(Base):
    lanes = 1

    def test_timeout_terminates_a_guest_that_never_exits(self):
        """The stock Zephyr hello_world case: prints, then idles forever."""
        t0 = time.time()
        jid = self.c.submit(self.h.spec(
            timeout_s=5, mock={**NEVER, "uart": ["Hello World!"]}))
        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "TIMEOUT")
        elapsed = time.time() - t0
        self.assertGreater(elapsed, 4)
        self.assertLess(elapsed, 45)
        # The FPGA is actually free again: teardown killed the simulation.
        self.h.wait_for(lambda: self.h.total_sims() == 0,
                        what="simulation killed on timeout")
        self.h.wait_for(lambda: all(not l["locked"] for l in self.c.lanes()),
                        what="lane freed after timeout")

    def test_sentinel_finishes_early_instead_of_burning_the_timeout(self):
        t0 = time.time()
        jid = self.c.submit(self.h.spec(
            timeout_s=120,
            completion={"mode": "sentinel",
                        "sentinel_regex": r"Hello World! \w+"},
            mock={**NEVER, "uart": ["*** Booting Zephyr OS ***",
                                    "Hello World! chipyard_riscv64"],
                  "uart_delay_s": 1}))
        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "DONE", job.get("message"))
        self.assertLess(time.time() - t0, 45,
                        "sentinel must end the job, not the 120s timeout")

    def test_fail_regex_marks_the_job_failed(self):
        jid = self.c.submit(self.h.spec(
            timeout_s=60,
            completion={"mode": "sentinel", "sentinel_regex": "NEVERMATCHES",
                        "fail_regex": "KERNEL PANIC"},
            mock={**NEVER, "uart": ["KERNEL PANIC: oops"]}))
        self.assertEqual(self.h.wait_terminal(jid, timeout=60)["state"],
                         "FAILED")

    def test_timeout_mode_treats_expiry_as_success(self):
        jid = self.c.submit(self.h.spec(
            timeout_s=4,
            completion={"mode": "timeout", "timeout_s": 4},
            mock=NEVER))
        self.assertEqual(self.h.wait_terminal(jid, timeout=60)["state"],
                         "DONE")

    def test_client_can_signal_completion(self):
        jid = self.c.submit(self.h.spec(timeout_s=120, mock=NEVER))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        self.c.signal_done(jid)
        self.assertEqual(self.h.wait_terminal(jid, timeout=60)["state"],
                         "DONE")


# ---------------------------------------------------------------------------
class TestCancel(Base):
    lanes = 1

    def test_cancel_queued(self):
        blocker = self.c.submit(self.h.spec(mock=NEVER, timeout_s=6))
        self.h.wait_state(blocker, ("DISPATCHING", "RUNNING"))
        queued = self.c.submit(self.h.spec(mock={"run_s": 1}))
        self.c.cancel(queued)
        self.assertEqual(self.c.get(queued)["state"], "CANCELLED")
        self.c.cancel(blocker)

    def test_cancel_running_tears_down_and_frees_the_lane(self):
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=300))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        self.h.wait_for(lambda: self.h.total_sims() == 1, what="sim started")
        self.c.cancel(jid)
        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "CANCELLED")
        # Teardown ran: the simulation was killed and the lock released.
        self.h.wait_for(lambda: self.h.total_sims() == 0,
                        what="sim killed by cancel")
        self.h.wait_for(lambda: all(not l["locked"] for l in self.c.lanes()),
                        what="lane freed by cancel")


# ---------------------------------------------------------------------------
class TestCrashRecovery(Base):
    lanes = 1

    def test_crashed_runner_frees_the_lane_and_is_reconciled(self):
        """A killed runner is exactly a crashed client: no cleanup runs."""
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=300))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        pid = self.h.wait_for(lambda: self.c.get(jid)["runner_pid"],
                              what="runner pid")
        os.kill(pid, signal.SIGKILL)

        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "FAILED")
        self.assertIn("crash", (job["message"] or "").lower())
        # The kernel released the flock when the runner died, so the lane is
        # immediately reusable -- no stale-lock cleanup code was involved.
        nxt = self.c.submit(self.h.spec(mock={"run_s": 0.3}))
        self.assertEqual(self.h.wait_terminal(nxt, timeout=60)["state"],
                         "DONE")

    def test_daemon_restart_readopts_a_running_job(self):
        """A daemon restart must not disturb a job mid-flight on an FPGA."""
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=90))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        pid_before = self.c.get(jid)["runner_pid"]
        self.h.wait_for(lambda: self.h.total_sims() == 1,
                        what="simulation started")

        self.h.restart()                      # SIGKILL + start again
        self.c = self.h.client()

        job = self.h.wait_for(
            lambda: (lambda j: j if j["state"] == "RUNNING" else None)(
                self.c.get(jid)),
            timeout=40, what="job re-adopted as RUNNING")
        self.assertEqual(job["runner_pid"], pid_before,
                         "the original runner must still own the job")
        self.assertEqual(self.h.total_sims(), 1,
                         "the simulation must be untouched by the restart")
        self.assertIn("re-adopted", self.h.daemon_log())

        # The lane is still locked, so nothing else can be dispatched onto it.
        other = self.c.submit(self.h.spec(mock={"run_s": 0.2}))
        time.sleep(2)
        self.assertEqual(self.c.get(other)["state"], "QUEUED")

        self.c.cancel(jid)
        self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(self.h.wait_terminal(other, timeout=60)["state"],
                         "DONE")

    def test_daemon_restart_reconciles_a_job_whose_runner_died(self):
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=300))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        pid = self.c.get(jid)["runner_pid"]
        self.h.kill_daemon()
        os.kill(pid, signal.SIGKILL)          # both daemon and runner gone
        self.h.start()
        self.c = self.h.client()
        job = self.h.wait_terminal(jid, timeout=40)
        self.assertEqual(job["state"], "FAILED")


# ---------------------------------------------------------------------------
class TestOrphanReclamation(Base):
    lanes = 2
    kw = {"probe": True, "admin": True}

    def test_orphaned_sim_makes_a_lane_unschedulable(self):
        """An idle-but-occupied FPGA is the pool's normal steady state."""
        self.h.start_foreign_sim("h0")
        lane = self.h.wait_for(
            lambda: next((l for l in self.c.lanes()
                          if l["name"] == "lane0"
                          and l["disposition"] == "FOREIGN"), None),
            timeout=30, what="lane0 detected as FOREIGN")
        self.assertIn("orphaned", lane["probe_detail"])

        # Work still flows -- onto the healthy lane only.
        jid = self.c.submit(self.h.spec(mock={"run_s": 0.3}))
        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "DONE")
        self.assertEqual(job["lane"], "lane1")

    def test_manual_reclaim_frees_the_lane(self):
        proc = self.h.start_foreign_sim("h0")
        self.h.wait_for(
            lambda: next((l for l in self.c.lanes()
                          if l["name"] == "lane0"
                          and l["disposition"] == "FOREIGN"), None),
            timeout=30, what="FOREIGN")

        dry = self.c._call("reclaim", lane="lane0", dry_run=True)
        self.assertTrue(dry["dry_run"])
        self.assertFalse(dry["reclaimed"])
        self.assertEqual(self.h.sims_on("h0"), 1, "dry run must kill nothing")

        res = self.c._call("reclaim", lane="lane0", dry_run=False)
        self.assertTrue(res["reclaimed"], res)
        self.assertEqual(self.h.sims_on("h0"), 0)
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, -9, "orphan must be SIGKILLed")

        self.h.wait_for(
            lambda: next((l for l in self.c.lanes()
                          if l["name"] == "lane0"
                          and l["disposition"] == "FREE"), None),
            timeout=30, what="lane0 FREE again")

    def test_reclaim_refuses_to_touch_a_live_fq_job(self):
        """The safety interlock: a lane we hold is never 'orphaned'."""
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=60))
        job = self.h.wait_state(jid, ("RUNNING",), timeout=40)
        self.h.wait_for(lambda: self.h.total_sims() == 1, what="sim started")
        res = self.c._call("reclaim", lane=job["lane"], dry_run=False)
        self.assertFalse(res["reclaimed"])
        self.assertIn("held by a live fq job", res["detail"])
        self.assertEqual(self.h.total_sims(), 1, "the live sim was killed!")
        self.c.cancel(jid)
        self.h.wait_terminal(jid, timeout=60)


class TestCrashLeavesAnOrphanedSim(Base):
    """The nastiest real failure: a runner is SIGKILLed, so its teardown
    never runs.  The kernel frees the lane lock, but the simulation it
    started is still attached to the FPGA.  The lock says free; the hardware
    is not.  The occupancy probe is what closes this gap, and the daemon must
    not schedule onto the lane until it is reclaimed."""

    lanes = 1
    kw = {"probe": True, "admin": True}

    def test_lane_is_not_reused_while_an_orphaned_sim_holds_the_fpga(self):
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=300))
        self.h.wait_state(jid, ("RUNNING",), timeout=40)
        self.h.wait_for(lambda: self.h.total_sims() == 1, what="sim started")
        pid = self.c.get(jid)["runner_pid"]

        os.kill(pid, signal.SIGKILL)          # no teardown runs

        job = self.h.wait_terminal(jid, timeout=60)
        self.assertEqual(job["state"], "FAILED")
        # The simulation outlived its runner and still owns the FPGA.
        self.assertGreaterEqual(self.h.total_sims(), 1)

        lane = self.h.wait_for(
            lambda: next((l for l in self.c.lanes()
                          if l["disposition"] == "FOREIGN"), None),
            timeout=30, what="lane detected FOREIGN after runner crash")
        self.assertEqual(lane["name"], "lane0")

        # Critically: the daemon must NOT dispatch onto that lane.
        nxt = self.c.submit(self.h.spec(mock={"run_s": 0.3}))
        time.sleep(3)
        self.assertEqual(self.c.get(nxt)["state"], "QUEUED",
                         "scheduler used a lane whose FPGA is still busy")

        # Reclaim clears it, and work resumes.
        res = self.c._call("reclaim", lane="lane0", dry_run=False)
        self.assertTrue(res["reclaimed"], res)
        self.assertEqual(self.h.wait_terminal(nxt, timeout=60)["state"],
                         "DONE")


class TestAutoReclamation(Base):
    lanes = 1
    kw = {"probe": True, "reclaim_policy": "auto",
          "reclaim_grace_s": 0, "admin": True}

    def test_auto_policy_reclaims_and_then_schedules(self):
        self.h.start_foreign_sim("h0")
        self.h.wait_for(lambda: self.h.sims_on("h0") == 0,
                        timeout=40, what="orphan auto-reclaimed")
        jid = self.c.submit(self.h.spec(mock={"run_s": 0.3}))
        self.assertEqual(self.h.wait_terminal(jid, timeout=60)["state"],
                         "DONE")


# ---------------------------------------------------------------------------
class TestMultiTenantSafety(Base):
    lanes = 1

    def test_cannot_cancel_another_users_job(self):
        """Identity comes from SO_PEERCRED, so it cannot be forged. Here we
        fake ownership in the DB and check the daemon refuses."""
        import sqlite3
        jid = self.c.submit(self.h.spec(mock=NEVER, timeout_s=5))
        self.h.wait_state(jid, ("DISPATCHING", "RUNNING"))
        db = sqlite3.connect(str(self.h.state / "queue.db"))
        db.execute("UPDATE jobs SET uid=?, user=? WHERE id=?",
                   (os.getuid() + 12345, "someone-else", jid))
        db.commit()
        db.close()
        with self.assertRaises(FqError) as cm:
            self.c.cancel(jid)
        self.assertIn("belongs to", str(cm.exception))

    def test_state_db_is_not_writable_by_clients(self):
        st = os.stat(self.h.state / "queue.db")
        self.assertEqual(st.st_mode & 0o077, 0,
                         "the state DB must be daemon-private; clients go "
                         "through the socket")

    def test_admin_only_operations_are_refused(self):
        # This test process is not root and not in `admins`.
        if os.getuid() == 0:
            self.skipTest("running as root; admin checks trivially pass")
        with self.assertRaises(FqError) as cm:
            self.c.drain("lane0")
        self.assertIn("admin-only", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
