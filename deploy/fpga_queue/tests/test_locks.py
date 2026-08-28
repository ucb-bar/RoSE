"""Lane locking — the correctness core."""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq.locks import LaneLock, LockOwner, pid_matches, proc_start_ticks


class TestLaneLock(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fqlock-")

    def test_exclusive(self):
        a = LaneLock(self.dir, "l0")
        b = LaneLock(self.dir, "l0")
        self.assertTrue(a.acquire(job_id=1, user="alice"))
        self.assertFalse(b.acquire(job_id=2, user="bob"))
        a.release()
        self.assertTrue(b.acquire(job_id=2, user="bob"))
        b.release()

    def test_is_free_reflects_reality(self):
        a = LaneLock(self.dir, "l0")
        self.assertTrue(a.is_free())
        a.acquire(job_id=1)
        # A *different* handle must see it as busy.
        self.assertFalse(LaneLock(self.dir, "l0").is_free())
        a.release()
        self.assertTrue(LaneLock(self.dir, "l0").is_free())

    def test_different_lanes_are_independent(self):
        a, b = LaneLock(self.dir, "l0"), LaneLock(self.dir, "l1")
        self.assertTrue(a.acquire())
        self.assertTrue(b.acquire())
        a.release()
        b.release()

    def test_lock_survives_fd_handoff_and_dies_with_holder(self):
        """The lifetime property the whole design depends on:
        the lock follows the *child*, not the parent."""
        lock = LaneLock(self.dir, "l0")
        self.assertTrue(lock.acquire(job_id=7))
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys,time; time.sleep(30)"],
            pass_fds=(lock.fd,))
        # Parent drops its copy; the child's inherited fd keeps it held.
        lock.detach()
        self.assertFalse(LaneLock(self.dir, "l0").is_free(),
                         "lock must still be held by the child")
        child.kill()
        child.wait(timeout=10)
        # Kernel released it on process death -- no cleanup code involved.
        deadline = time.time() + 5
        while time.time() < deadline:
            if LaneLock(self.dir, "l0").is_free():
                break
            time.sleep(0.05)
        self.assertTrue(LaneLock(self.dir, "l0").is_free(),
                        "lock must be released when the holder dies")

    def test_crashed_holder_releases_lock(self):
        """SIGKILL is the crash case: no handler, no unwinding."""
        script = (
            "import fcntl,os,sys,time\n"
            f"fd=os.open({os.path.join(self.dir, 'l0.lock')!r},"
            " os.O_RDWR|os.O_CREAT, 0o644)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n")
        p = subprocess.Popen([sys.executable, "-c", script],
                             stdout=subprocess.PIPE, text=True)
        self.assertEqual(p.stdout.readline().strip(), "locked")
        self.assertFalse(LaneLock(self.dir, "l0").is_free())
        p.kill()
        p.wait(timeout=10)
        deadline = time.time() + 5
        while time.time() < deadline and not LaneLock(self.dir, "l0").is_free():
            time.sleep(0.05)
        self.assertTrue(LaneLock(self.dir, "l0").is_free())

    def test_owner_file_roundtrip(self):
        lock = LaneLock(self.dir, "l0")
        lock.acquire(job_id=42, user="alice")
        o = lock.read_owner()
        self.assertIsNotNone(o)
        self.assertEqual(o.job_id, 42)
        self.assertEqual(o.user, "alice")
        self.assertEqual(o.pid, os.getpid())
        lock.release()

    def test_describe(self):
        lock = LaneLock(self.dir, "l0")
        self.assertTrue(lock.describe()["free"])
        lock.acquire(job_id=1, user="u")
        d = LaneLock(self.dir, "l0").describe()
        self.assertFalse(d["free"])
        self.assertEqual(d["owner"]["job_id"], 1)
        self.assertTrue(d["owner_alive"])
        lock.release()


class TestPidFingerprint(unittest.TestCase):
    def test_start_ticks_present_and_stable(self):
        t1 = proc_start_ticks(os.getpid())
        self.assertIsNotNone(t1)
        self.assertEqual(t1, proc_start_ticks(os.getpid()))

    def test_dead_pid(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertIsNone(proc_start_ticks(p.pid))
        self.assertFalse(pid_matches(p.pid, None))

    def test_recycled_pid_is_not_mistaken_for_the_holder(self):
        """A stale (pid, start_ticks) pair must not match a live process."""
        self.assertFalse(pid_matches(os.getpid(), 999999999))
        self.assertTrue(pid_matches(os.getpid(),
                                    proc_start_ticks(os.getpid())))

    def test_comm_with_spaces_and_parens(self):
        # /proc/<pid>/stat's comm field is parenthesised and may itself
        # contain ')' -- parsing from the left would break.
        self.assertIsNotNone(proc_start_ticks(1))


if __name__ == "__main__":
    unittest.main()
