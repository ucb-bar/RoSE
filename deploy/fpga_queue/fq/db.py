"""Durable queue state.

SQLite, WAL mode, **written only by the daemon**.  That is the main departure
from firesim-queue, which had every client open the shared DB read/write.  The
DB there had to be group-writable by every submitter, which meant any submitter
could ``UPDATE`` any row -- including another user's state, or their own
``owner_uid``.  Here the file is daemon-private (0600) and all mutation goes
through the socket, where the caller's uid comes from ``SO_PEERCRED`` and
cannot be forged.

The DB is *not* the source of truth for whether a lane is busy.  The flock is
(see fq/locks.py).  The DB records intent and history; the kernel records
ownership.  When they disagree, the kernel wins and the daemon reconciles.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional

# Terminal states -- a job in one of these will never run again.
TERMINAL = ("DONE", "FAILED", "CANCELLED", "TIMEOUT")
ACTIVE = ("QUEUED", "DISPATCHING", "RUNNING")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user          TEXT    NOT NULL,
    uid           INTEGER NOT NULL,
    gid           INTEGER,
    priority      INTEGER NOT NULL DEFAULT 5,
    state         TEXT    NOT NULL,
    phase         TEXT,
    submitted_at  REAL    NOT NULL,
    started_at    REAL,
    ended_at      REAL,
    exit_code     INTEGER,
    lane          TEXT,
    num_fpgas     INTEGER NOT NULL DEFAULT 1,
    tree          TEXT,
    hw_config     TEXT,
    agfi          TEXT,
    workload      TEXT,
    project       TEXT,
    timeout_s     INTEGER NOT NULL DEFAULT 0,
    runner_pid    INTEGER,
    runner_ticks  INTEGER,
    cancel_requested REAL,
    workdir       TEXT    NOT NULL,
    spec_json     TEXT    NOT NULL,
    message       TEXT,
    blocked_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, priority DESC, submitted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user  ON jobs(user, state);
CREATE INDEX IF NOT EXISTS idx_jobs_lane  ON jobs(lane, state);

-- Lane affinity / warmth.  Used by the scheduler to prefer a lane whose
-- driver already matches the job's (tree, hw_config), because a cold
-- infrasetup costs tens of minutes.
CREATE TABLE IF NOT EXISTS lane_state (
    name           TEXT PRIMARY KEY,
    last_tree      TEXT,
    last_hw_config TEXT,
    last_job_id    INTEGER,
    updated_at     REAL,
    drain          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    job_id  INTEGER,
    lane    TEXT,
    kind    TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, id);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Db:
    """Thin, explicitly-locked wrapper around the state DB.

    The daemon serves socket requests on worker threads while its main loop
    dispatches, so every statement is guarded by one mutex.  SQLite could do
    finer-grained concurrency; at this scale (tens of jobs, 5s poll) a single
    mutex is simply not the bottleneck and it removes a whole class of bug.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.path.exists()
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30.0,
                                    isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if newfile:
            # Daemon-private: clients never touch this file.
            os.chmod(self.path, 0o600)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- generic helpers --------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- kv ---------------------------------------------------------------
    def kv_get(self, key: str) -> Optional[str]:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return r["value"] if r else None

    def kv_set(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    # -- events -----------------------------------------------------------
    def event(self, kind: str, job_id: Optional[int] = None,
              lane: Optional[str] = None, detail: str = "") -> None:
        self.execute(
            "INSERT INTO events(ts, job_id, lane, kind, detail) "
            "VALUES(?,?,?,?,?)", (time.time(), job_id, lane, kind, detail))

    def job_events(self, job_id: int, limit: int = 200) -> list[dict]:
        return self.query(
            "SELECT ts, kind, lane, detail FROM events WHERE job_id=? "
            "ORDER BY id LIMIT ?", (job_id, limit))

    # -- jobs -------------------------------------------------------------
    def insert_job(self, *, user: str, uid: int, gid: Optional[int],
                   priority: int, spec: dict, workdir: str,
                   num_fpgas: int, tree: Optional[str],
                   hw_config: Optional[str], agfi: Optional[str],
                   workload: Optional[str], project: Optional[str],
                   timeout_s: int) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO jobs(user, uid, gid, priority, state, "
                " submitted_at, num_fpgas, tree, hw_config, agfi, workload, "
                " project, timeout_s, workdir, spec_json) "
                "VALUES(?,?,?,?, 'QUEUED', ?,?,?,?,?,?,?,?,?,?)",
                (user, uid, gid, priority, time.time(), num_fpgas, tree,
                 hw_config, agfi, workload, project, timeout_s, workdir,
                 json.dumps(spec)))
            return int(cur.lastrowid)

    def get_job(self, job_id: int) -> Optional[dict]:
        return self.one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def set_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE jobs SET {cols} WHERE id=?",
                     (*fields.values(), job_id))

    def queued_jobs(self) -> list[dict]:
        return self.query(
            "SELECT * FROM jobs WHERE state='QUEUED' ORDER BY id")

    def active_jobs(self) -> list[dict]:
        marks = ",".join("?" * len(ACTIVE))
        return self.query(
            f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY id", ACTIVE)

    def jobs_on_lane(self, lane: str) -> list[dict]:
        marks = ",".join("?" * len(ACTIVE))
        return self.query(
            f"SELECT * FROM jobs WHERE lane=? AND state IN ({marks})",
            (lane, *ACTIVE))

    def running_fpgas_by_user(self) -> dict[str, int]:
        rows = self.query(
            "SELECT user, SUM(num_fpgas) AS n FROM jobs "
            "WHERE state IN ('DISPATCHING','RUNNING') GROUP BY user")
        return {r["user"]: int(r["n"] or 0) for r in rows}

    def count_queued_for_user(self, user: str) -> int:
        r = self.one("SELECT COUNT(*) AS n FROM jobs WHERE user=? AND "
                     "state='QUEUED'", (user,))
        return int(r["n"]) if r else 0

    def finish_job(self, job_id: int, state: str, exit_code: Optional[int],
                   message: str = "") -> None:
        self.set_job(job_id, state=state, exit_code=exit_code,
                     ended_at=time.time(), runner_pid=None,
                     runner_ticks=None, message=message or None,
                     phase=state)

    # -- lane state -------------------------------------------------------
    def lane_row(self, name: str) -> dict:
        r = self.one("SELECT * FROM lane_state WHERE name=?", (name,))
        if r:
            return r
        self.execute(
            "INSERT OR IGNORE INTO lane_state(name, updated_at) VALUES(?,?)",
            (name, time.time()))
        return self.one("SELECT * FROM lane_state WHERE name=?", (name,)) or {
            "name": name, "last_tree": None, "last_hw_config": None,
            "last_job_id": None, "updated_at": time.time(), "drain": 0}

    def mark_lane_used(self, name: str, tree: Optional[str],
                       hw_config: Optional[str], job_id: int) -> None:
        self.lane_row(name)
        self.execute(
            "UPDATE lane_state SET last_tree=?, last_hw_config=?, "
            "last_job_id=?, updated_at=? WHERE name=?",
            (tree, hw_config, job_id, time.time(), name))

    def set_lane_drain(self, name: str, drain: bool) -> None:
        self.lane_row(name)
        self.execute("UPDATE lane_state SET drain=?, updated_at=? WHERE name=?",
                     (1 if drain else 0, time.time(), name))
