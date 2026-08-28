"""Per-lane exclusion.

The whole correctness story of this daemon rests on this file, so it is kept
as small and as boring as possible.

Design
------
Each lane has a lock file ``<state>/lanes/<lane>.lock``.  Exclusion is
``flock(LOCK_EX)``.  We use flock rather than a PID file or a DB row because
**the kernel releases it when the holding process dies** -- there is no code
path, no cleanup handler and no daemon liveness check involved.  That gives us
crashed-client recovery for free, which is the failure mode that actually
happens.

Ownership transfer to the job runner works via fd inheritance:

  1. the daemon opens the lock file and takes ``LOCK_EX|LOCK_NB``;
  2. it forks/execs the runner, which inherits that fd;
  3. the daemon closes *its* copy.

flock locks live on the open file *description*, which is shared across
fork and preserved across exec, and released only when the last fd referring
to it is closed.  So after step 3 the runner is the sole holder, and the lock
survives a daemon restart but not a runner death.  It also survives the
runner dropping privileges, because flock needs only an open fd -- not write
permission, and not the file's ownership.

A sidecar ``<lane>.owner`` JSON file records who holds it.  That file is
*advisory only*: it is for humans and for `fq lanes` output.  Anything that
must be correct asks the kernel with a non-blocking flock probe instead.

PID reuse
---------
The owner file records the holder's process start time (field 22 of
``/proc/<pid>/stat``, in clock ticks since boot) alongside its PID, so that a
recycled PID cannot be mistaken for a live holder.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import pathlib
import time
from typing import Any, Optional


@dataclasses.dataclass
class LockOwner:
    pid: int
    start_ticks: Optional[int]
    job_id: Optional[int]
    user: Optional[str]
    acquired_at: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Optional["LockOwner"]:
        try:
            d = json.loads(text)
            return cls(
                pid=int(d["pid"]),
                start_ticks=(int(d["start_ticks"])
                             if d.get("start_ticks") is not None else None),
                job_id=(int(d["job_id"]) if d.get("job_id") is not None else None),
                user=d.get("user"),
                acquired_at=float(d.get("acquired_at", 0.0)),
            )
        except (ValueError, KeyError, TypeError):
            return None


def proc_start_ticks(pid: int) -> Optional[int]:
    """Process start time in clock ticks since boot, or None if it is gone.

    Used together with the PID to make a fingerprint that survives PID reuse.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm (field 2) is parenthesised and may contain spaces/parens, so split
    # on the LAST ')' rather than tokenising from the left.
    close = data.rfind(b")")
    if close < 0:
        return None
    fields = data[close + 2:].split()
    # After comm, field 3 is state; starttime is overall field 22, i.e.
    # index 19 of this remainder.
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def pid_matches(pid: int, start_ticks: Optional[int]) -> bool:
    """True if `pid` is alive and (if known) is the same incarnation."""
    if pid <= 0:
        return False
    cur = proc_start_ticks(pid)
    if cur is None:
        return False
    if start_ticks is None:
        return True
    return cur == start_ticks


class LaneLock:
    """Non-blocking exclusive lock over one lane."""

    def __init__(self, locks_dir: str | os.PathLike[str], lane: str):
        self.lane = lane
        self.dir = pathlib.Path(locks_dir)
        self.path = self.dir / f"{lane}.lock"
        self.owner_path = self.dir / f"{lane}.owner"
        self.fd: Optional[int] = None

    # -- acquisition ------------------------------------------------------
    def acquire(self, job_id: Optional[int] = None,
                user: Optional[str] = None,
                pid: Optional[int] = None) -> bool:
        """Take the lock without blocking.  Returns False if already held.

        The fd is left open and is deliberately *not* marked close-on-exec, so
        it can be handed to a child process.
        """
        if self.fd is not None:
            return True
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        os.set_inheritable(fd, True)
        self.fd = fd
        holder = pid if pid is not None else os.getpid()
        self.write_owner(LockOwner(
            pid=holder,
            start_ticks=proc_start_ticks(holder),
            job_id=job_id,
            user=user,
            acquired_at=time.time(),
        ))
        return True

    def write_owner(self, owner: LockOwner) -> None:
        tmp = self.owner_path.with_suffix(".owner.tmp")
        try:
            tmp.write_text(owner.to_json() + "\n")
            os.replace(tmp, self.owner_path)
        except OSError:
            # Advisory metadata only -- never fail a job because we could not
            # write a breadcrumb.
            with contextlib.suppress(OSError):
                tmp.unlink()

    def read_owner(self) -> Optional[LockOwner]:
        try:
            return LockOwner.from_json(self.owner_path.read_text())
        except OSError:
            return None

    def release(self) -> None:
        if self.fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self.fd)
        self.fd = None
        with contextlib.suppress(OSError):
            self.owner_path.unlink()

    def detach(self) -> None:
        """Close our fd *without* unlocking, after handing it to a child.

        flock is held by the open file description, which the child inherited,
        so the lock stays taken until the child exits.
        """
        if self.fd is None:
            return
        with contextlib.suppress(OSError):
            os.close(self.fd)
        self.fd = None

    # -- inspection -------------------------------------------------------
    def is_free(self) -> bool:
        """Ask the kernel whether the lane is unlocked, right now.

        This is the authoritative answer.  Never infer freeness from the DB,
        from the owner file, or from a job's self-reported status -- a
        submitter can write those, and a crashed daemon can leave them stale.
        """
        if self.fd is not None:
            return False
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    # Lane held by someone else. That is a normal answer, not
                    # an error -- but this path used to return WITHOUT closing
                    # fd, leaking one descriptor per busy-lane probe. With a
                    # reconcile every few seconds across 8 busy lanes that
                    # exhausts the 1024 soft limit in wall-clock under an hour,
                    # after which accept() fails EMFILE and the API dies. That
                    # is what took the pool down twice on 2026-08-28.
                    return False
                raise
            finally_free = True
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                finally_free = False
            return finally_free
        finally:
            os.close(fd)

    def describe(self) -> dict[str, Any]:
        free = self.is_free()
        owner = None if free else self.read_owner()
        return {
            "lane": self.lane,
            "free": free,
            "owner": dataclasses.asdict(owner) if owner else None,
            "owner_alive": (pid_matches(owner.pid, owner.start_ticks)
                            if owner else None),
        }
