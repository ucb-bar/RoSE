"""The daemon: socket server + dispatch loop + recovery.

Structure
---------
Two concerns, deliberately separated:

  * an **API thread pool** answering unix-socket requests (submit, status,
    cancel, ...).  These only read and write the DB.
  * a single **dispatch loop** that reconciles reality, plans, and spawns
    runners.  Only this loop starts jobs, so there is exactly one place where
    a lane can be handed out.

Every iteration of the dispatch loop does the same three things in the same
order, which is what makes the daemon's behaviour explainable:

  1. **Reconcile.**  Look at the world -- lane flocks, runner processes,
     status files, occupancy probes -- and make the DB agree with it.  The
     kernel is the authority; the DB is the record.
  2. **Plan.**  Call the pure scheduler with the reconciled state.
  3. **Dispatch.**  For each placement, take the lane's flock, spawn a runner
     holding it, and hand the fd over.

Restart safety
--------------
A restarted daemon does not assume anything is dead.  For each lane it asks
the kernel whether the flock is held:

  * held  -> a runner from a previous daemon is still working.  Re-adopt it:
             keep the job RUNNING and resume monitoring it.  Its hardware and
             its lock are untouched.
  * free  -> nothing owns the lane.  Any job the DB still calls RUNNING there
             is orphaned, and is reconciled to FAILED.

This is why the lock lives in a runner process and not in the daemon.  A
daemon restart during a 40-minute ``infrasetup`` costs nothing.
"""

from __future__ import annotations

import contextlib
import getpass
import grp
import json
import os
import pathlib
import pwd
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Optional

from . import completion, proto
from .backends import get_backend
from .config import PoolConfig
from .db import Db
from .locks import LaneLock, pid_matches, proc_start_ticks
from .occupancy import LaneDisposition, Prober, should_auto_reclaim
from .scheduler import JobView, LaneView, plan as plan_jobs

TERMINAL = ("DONE", "FAILED", "CANCELLED", "TIMEOUT")


def _username(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid{uid}"


class Daemon:
    def __init__(self, cfg: PoolConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        cfg.locks_dir.mkdir(parents=True, exist_ok=True)
        self.db = Db(cfg.db_path)
        self.prober = Prober(cfg.probe)
        self.locks = {ln.name: LaneLock(cfg.locks_dir, ln.name)
                      for ln in cfg.lanes}
        self.backend = get_backend(cfg.backend, cfg.backend_options)
        self.stop = threading.Event()
        self._probe_cache: dict[str, Any] = {}
        self._foreign_since: dict[str, float] = {}
        self.sock: Optional[socket.socket] = None
        self.am_root = (os.geteuid() == 0)

    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)

    def vlog(self, msg: str) -> None:
        if self.verbose:
            self.log(msg)

    # ==================================================================
    # API
    # ==================================================================
    def serve_socket(self) -> None:
        path = self.cfg.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # Only safe because we hold the daemon singleton lock by then.
            path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(path))
        # World-connectable: authorisation is by SO_PEERCRED, not by file
        # mode. Anyone who can reach the socket is identified by the kernel;
        # what they are allowed to do is decided per request.
        os.chmod(path, 0o777)
        s.listen(64)
        s.settimeout(1.0)
        self.sock = s
        self.log(f"listening on {path}")
        while not self.stop.is_set():
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                # A transient accept() error must NOT kill the API thread.
                # Bailing out here unlinks the socket while the dispatch loop
                # keeps running, which leaves a live daemon that nothing can
                # talk to AND that no replacement can displace, because the
                # singleton flock is still held. That strands the whole pool.
                if self.stop.is_set():
                    break
                self.log(f"accept() error, continuing to serve: {exc!r}")
                time.sleep(0.5)
                continue
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()
        with contextlib.suppress(OSError):
            s.close()
        # Only remove the socket when we are genuinely stopping. Unlinking it on
        # any other exit path is what disconnected every client on 2026-08-28.
        if self.stop.is_set():
            with contextlib.suppress(OSError):
                path.unlink()
        self.log(f"api: serve_socket exited (stopping={self.stop.is_set()})")

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30.0)
            pid, uid, gid = proto.peer_credentials(conn)
            req = proto.recv_msg(conn)
            if not isinstance(req, dict):
                proto.send_msg(conn, proto.err("malformed request"))
                return
            op = req.get("op")
            fn = getattr(self, f"op_{op}", None)
            if fn is None:
                proto.send_msg(conn, proto.err(f"unknown op {op!r}"))
                return
            try:
                result = fn(req, uid, gid, pid)
                proto.send_msg(conn, proto.ok(result))
            except PermissionError as exc:
                proto.send_msg(conn, proto.err(str(exc), "permission"))
            except ValueError as exc:
                proto.send_msg(conn, proto.err(str(exc), "invalid"))
            except Exception as exc:  # noqa: BLE001
                self.log(f"api error: {traceback.format_exc()}")
                proto.send_msg(conn, proto.err(f"internal error: {exc}"))
        except (OSError, proto.ProtocolError):
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    # -- authorisation ---------------------------------------------------
    def _is_admin(self, uid: int) -> bool:
        return uid == 0 or _username(uid) in self.cfg.admins

    def _require_owner(self, job: dict, uid: int) -> None:
        if self._is_admin(uid):
            return
        if int(job["uid"]) != uid:
            raise PermissionError(
                f"job {job['id']} belongs to {job['user']}; you are "
                f"{_username(uid)}")

    # -- operations ------------------------------------------------------
    def op_ping(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        return {
            "version": 1,
            "pid": os.getpid(),
            "root": self.am_root,
            "backend": self.cfg.backend,
            "pool": str(self.cfg.source_path),
            "lanes": len(self.cfg.lanes),
            "you": {"uid": uid, "user": _username(uid), "admin":
                    self._is_admin(uid)},
        }

    def op_submit(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        spec = dict(req.get("spec") or {})
        user = _username(uid)

        # Multi-tenancy gate.  Running a job as its submitter needs root (we
        # drop privileges per job).  Without root we can only run as
        # ourselves, so accepting someone else's job would silently run it
        # with the wrong identity and the wrong filesystem access.
        if not self.am_root and uid != os.geteuid():
            if not self.cfg.allow_foreign_uid_without_root:
                raise PermissionError(
                    f"this daemon runs as {_username(os.geteuid())} and not as "
                    f"root, so it cannot run jobs as {user}. Either run the "
                    f"daemon as root (jobs then execute as their submitter) "
                    f"or set allow_foreign_uid_without_root in the pool "
                    f"config if you accept that jobs run as the daemon user.")

        queued = self.db.count_queued_for_user(user)
        if queued >= self.cfg.max_queued_per_user:
            raise ValueError(
                f"{user} already has {queued} queued jobs "
                f"(max_queued_per_user={self.cfg.max_queued_per_user})")

        # Resolve paths relative to the client's cwd, since the daemon's cwd
        # is meaningless to the submitter.
        cwd = spec.get("cwd") or "/"
        for key in ("elf", "results_dir", "tree"):
            v = spec.get(key)
            if v and not str(v).startswith("/"):
                spec[key] = os.path.normpath(os.path.join(cwd, str(v)))

        errs = self.backend.validate(spec, self.cfg)
        # A crashed guest hangs in Zephyr's fatal handler while the FireSim
        # driver keeps spinning, so the job reads RUNNING and squats its FPGA
        # for the entire timeout -- eight such jobs cost roughly six lane-hours
        # in one afternoon. Default every job to failing fast on a guest fault
        # unless it sets its own rule. This must happen BEFORE spec.json and
        # the db row are written, or the runner never sees it.
        _dfr = getattr(self.cfg, "default_fail_regex", None)
        if _dfr and not (spec.get("fail_regex")
                         or (spec.get("completion") or {}).get("fail_regex")):
            spec["fail_regex"] = _dfr
        errs += completion.validate(spec)
        if errs:
            raise ValueError("invalid job spec:\n  - " + "\n  - ".join(errs))

        num = int(spec.get("num_fpgas", 1))
        tree = spec.get("tree")
        lane_hint = spec.get("lane_hint")
        if lane_hint and self.cfg.lane(lane_hint) is None:
            raise ValueError(f"no such lane: {lane_hint}")
        # Fail at submit time, not after an hour in the queue.
        if not any(ln.enabled and ln.capacity >= num and ln.accepts_tree(tree)
                   and (not lane_hint or ln.name == lane_hint)
                   for ln in self.cfg.lanes):
            raise ValueError(
                f"no lane in this pool can ever host this job "
                f"(num_fpgas={num}, tree={tree}"
                + (f", lane_hint={lane_hint}" if lane_hint else "") + ")")

        timeout = int(spec.get("timeout_s") or 0) or self.cfg.default_timeout_s
        if timeout > self.cfg.max_timeout_s:
            raise ValueError(
                f"timeout_s={timeout} exceeds the pool maximum "
                f"{self.cfg.max_timeout_s}")
        spec["timeout_s"] = timeout

        prio = int(spec.get("priority", self.cfg.default_priority))
        if prio > 10 and not self._is_admin(uid):
            raise PermissionError("priority > 10 is admin-only")

        job_id = self.db.insert_job(
            user=user, uid=uid, gid=gid, priority=prio, spec=spec,
            workdir="", num_fpgas=num, tree=tree,
            hw_config=spec.get("hw_config"), agfi=spec.get("agfi"),
            workload=spec.get("workload"),
            project=spec.get("project"), timeout_s=timeout)

        workdir = self.cfg.jobs_dir / str(job_id)
        workdir.mkdir(parents=True, exist_ok=True)
        # The submitter owns their job dir: the runner writes into it as them.
        if self.am_root:
            with contextlib.suppress(OSError, KeyError):
                os.chown(workdir, uid, gid if gid is not None else -1)
        os.chmod(workdir, 0o2775)
        (workdir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        self.db.set_job(job_id, workdir=str(workdir))
        self.db.event("submitted", job_id, detail=f"user={user} prio={prio}")
        pol = completion.from_spec(spec, timeout)
        self.log(f"job {job_id} submitted by {user} "
                 f"(prio={prio}, {num} FPGA(s), hw={spec.get('hw_config')}, "
                 f"completion={pol.mode})")
        return {"job_id": job_id, "workdir": str(workdir),
                "completion": pol.to_dict(),
                # Shown by the CLI at submit time: by the time a runaway
                # guest matters, it is already holding an FPGA.
                "advisory": completion.advisory(spec, timeout)}

    def op_get(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        job = self.db.get_job(int(req["job_id"]))
        if not job:
            raise ValueError(f"no such job: {req['job_id']}")
        job["spec"] = json.loads(job.pop("spec_json", "{}") or "{}")
        return job

    def op_list(self, req: dict, uid: int, gid: int, pid: int) -> list[dict]:
        where, params = [], []
        if req.get("user"):
            where.append("user=?")
            params.append(req["user"])
        if req.get("state"):
            where.append("state=?")
            params.append(req["state"])
        elif not req.get("all"):
            where.append("state IN ('QUEUED','DISPATCHING','RUNNING')")
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(req.get("limit", 200)))
        rows = self.db.query(sql, params)
        for r in rows:
            r.pop("spec_json", None)
        return rows

    def op_cancel(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        job = self.db.get_job(int(req["job_id"]))
        if not job:
            raise ValueError(f"no such job: {req['job_id']}")
        self._require_owner(job, uid)
        if job["state"] in TERMINAL:
            return {"job_id": job["id"], "state": job["state"],
                    "note": "already finished"}
        if job["state"] == "QUEUED":
            self.db.finish_job(job["id"], "CANCELLED", None,
                               f"cancelled by {_username(uid)} while queued")
            self.db.event("cancelled", job["id"], detail="queued")
            return {"job_id": job["id"], "state": "CANCELLED"}
        # RUNNING/DISPATCHING: ask the runner to stop.  It owns teardown, so
        # we signal rather than kill -- a killed runner would skip `firesim
        # kill` and leave the FPGA occupied.
        wd = pathlib.Path(job["workdir"])
        with contextlib.suppress(OSError):
            (wd / "cancel").write_text(f"{_username(uid)} {time.time()}\n")
        self.db.set_job(job["id"], cancel_requested=time.time())
        self.db.event("cancel-requested", job["id"], job["lane"],
                      f"by {_username(uid)}")
        return {"job_id": job["id"], "state": job["state"],
                "note": "cancel requested; runner is tearing down"}

    def op_signal_done(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        """Explicit completion signal for guests with no natural end."""
        job = self.db.get_job(int(req["job_id"]))
        if not job:
            raise ValueError(f"no such job: {req['job_id']}")
        self._require_owner(job, uid)
        if job["state"] not in ("RUNNING", "DISPATCHING"):
            raise ValueError(f"job {job['id']} is {job['state']}, not running")
        with contextlib.suppress(OSError):
            (pathlib.Path(job["workdir"]) / "done").write_text(
                f"{_username(uid)} {time.time()}\n")
        self.db.event("client-done", job["id"], job["lane"],
                      f"by {_username(uid)}")
        return {"job_id": job["id"], "note": "completion signalled"}

    def op_lanes(self, req: dict, uid: int, gid: int, pid: int) -> list[dict]:
        out = []
        for ln in self.cfg.lanes:
            row = self.db.lane_row(ln.name)
            pr = self._probe_cache.get(ln.name)
            jobs = self.db.jobs_on_lane(ln.name)
            active = [j for j in jobs if j["state"] in
                      ("DISPATCHING", "RUNNING")]
            out.append({
                "name": ln.name,
                "mode": ln.mode,
                "tag": ln.tag,
                "capacity": ln.capacity,
                "hosts": list(ln.hosts),
                "enabled": ln.enabled,
                "drain": bool(ln.drain or row.get("drain")),
                "trees": list(ln.trees),
                "locked": not self.locks[ln.name].is_free(),
                "disposition": (pr["disposition"] if pr else "UNKNOWN"),
                "probe_detail": (pr.get("detail") if pr else ""),
                "probed_at": (pr.get("checked_at") if pr else None),
                "foreign_since": self._foreign_since.get(ln.name),
                "last_tree": row.get("last_tree"),
                "last_hw_config": row.get("last_hw_config"),
                "job": (active[0]["id"] if active else None),
            })
        return out

    def op_drain(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        if not self._is_admin(uid):
            raise PermissionError("drain is admin-only")
        lane = str(req["lane"])
        if self.cfg.lane(lane) is None:
            raise ValueError(f"no such lane: {lane}")
        on = bool(req.get("on", True))
        self.db.set_lane_drain(lane, on)
        self.db.event("drain" if on else "undrain", lane=lane)
        self.log(f"lane {lane} {'drained' if on else 'undrained'} by "
                 f"{_username(uid)}")
        return {"lane": lane, "drain": on}

    def op_reclaim(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        """Operator-triggered reclamation of an orphaned lane."""
        if not self._is_admin(uid):
            raise PermissionError("reclaim is admin-only")
        lane = str(req["lane"])
        lc = self.cfg.lane(lane)
        if lc is None:
            raise ValueError(f"no such lane: {lane}")
        return self._reclaim(lane, f"manual by {_username(uid)}",
                             dry_run=bool(req.get("dry_run", False)))

    def op_logs(self, req: dict, uid: int, gid: int, pid: int) -> dict:
        job = self.db.get_job(int(req["job_id"]))
        if not job:
            raise ValueError(f"no such job: {req['job_id']}")
        name = {"stdout": "job.log", "job": "job.log",
                "status": "status.json"}.get(req.get("stream", "stdout"),
                                             "job.log")
        path = pathlib.Path(job["workdir"]) / name
        offset = int(req.get("offset", 0))
        maxb = min(int(req.get("max_bytes", 262144)), 4 * 1024 * 1024)
        if not path.exists():
            return {"data": "", "offset": offset, "eof": job["state"] in TERMINAL}
        with path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(maxb)
        return {"data": data.decode("utf-8", "replace"),
                "offset": offset + len(data),
                "eof": job["state"] in TERMINAL}

    def op_events(self, req: dict, uid: int, gid: int, pid: int) -> list[dict]:
        return self.db.job_events(int(req["job_id"]))

    # ==================================================================
    # Reconciliation
    # ==================================================================
    def adopt(self) -> None:
        """Startup pass: work out what is already running and say so.

        A restarted daemon inherits a live pool.  Rather than inferring this
        silently, it states what it found -- which lanes are still held, by
        which job, and which DB rows were stale.  "What is on the hardware
        right now" is the first question anyone asks after a restart.
        """
        for ln in self.cfg.lanes:
            lock = self.locks[ln.name]
            if lock.is_free():
                continue
            owner = lock.read_owner()
            jid = owner.job_id if owner else None
            job = self.db.get_job(jid) if jid else None
            if job and job["state"] not in TERMINAL:
                self.db.set_job(jid, state="RUNNING", lane=ln.name,
                                runner_pid=owner.pid,
                                runner_ticks=owner.start_ticks)
                self.db.event("re-adopted", jid, ln.name,
                              f"runner pid {owner.pid}")
                self.log(f"re-adopted job {jid} ({job['user']}) still running "
                         f"on lane {ln.name}, runner pid {owner.pid} -- its "
                         f"lock and its FPGA were never disturbed")
            else:
                self.log(f"lane {ln.name} is locked by pid "
                         f"{owner.pid if owner else '?'} but no active job "
                         f"claims it; leaving the lock alone")

    def reconcile(self) -> None:
        """Make the DB agree with the kernel.  Runs every loop iteration."""
        for ln in self.cfg.lanes:
            lock = self.locks[ln.name]
            free = lock.is_free()
            active = [j for j in self.db.jobs_on_lane(ln.name)
                      if j["state"] in ("DISPATCHING", "RUNNING")]

            if not free:
                owner = lock.read_owner()
                for job in active:
                    self._absorb_status(job)
                if not active and owner and owner.job_id:
                    # A runner from a previous daemon incarnation. Re-adopt.
                    j = self.db.get_job(owner.job_id)
                    if j and j["state"] not in TERMINAL:
                        self.db.set_job(owner.job_id, state="RUNNING",
                                        lane=ln.name, runner_pid=owner.pid,
                                        runner_ticks=owner.start_ticks)
                        self.log(f"re-adopted job {owner.job_id} still running "
                                 f"on lane {ln.name} (pid {owner.pid})")
                continue

            # Lane lock is FREE.  Anything the DB thinks is running here is
            # gone: the runner died without finishing.  Believe the kernel.
            for job in active:
                if self._absorb_status(job):
                    continue
                self.db.finish_job(
                    job["id"], "FAILED", -1,
                    "runner exited without reporting a terminal state "
                    "(daemon or runner crash); lane lock was released")
                self.db.event("orphan-reconciled", job["id"], ln.name)
                self.log(f"job {job['id']} on lane {ln.name} was orphaned "
                         f"(lock free, no terminal status) -> FAILED")
                # A runner that died without running teardown leaves its
                # phase subprocess -- and therefore the simulation -- alive
                # on the FPGA.  The lock is correctly free, but the hardware
                # is NOT.  Re-probe this lane immediately rather than waiting
                # out the probe interval, so the lane is classified FOREIGN
                # before the scheduler can hand it to the next job.
                self._probe_now(ln)

    def _absorb_status(self, job: dict) -> bool:
        """Copy a runner's status.json into the DB.  True if it is terminal.

        status.json is written by the runner, which runs as the submitter, so
        it is untrusted for *authorisation* purposes -- but it is the only
        source for phase and exit code, and a submitter faking their own job's
        exit code harms only themselves.  Crucially, a terminal status here
        does NOT free the lane; only the flock does.
        """
        wd = pathlib.Path(job["workdir"] or "")
        try:
            doc = json.loads((wd / "status.json").read_text())
        except (OSError, ValueError):
            return False
        state = str(doc.get("state") or "")
        phase = doc.get("phase")
        if phase and phase != job.get("phase"):
            self.db.set_job(job["id"], phase=phase)
        if state in TERMINAL:
            if job["state"] not in TERMINAL:
                self.db.finish_job(job["id"], state,
                                   doc.get("exit_code"),
                                   str(doc.get("message") or ""))
                self.db.event("finished", job["id"], job["lane"],
                              f"state={state} rc={doc.get('exit_code')}")
                self.log(f"job {job['id']} finished: {state} "
                         f"rc={doc.get('exit_code')} lane={job['lane']}")
            return True
        if job["state"] == "DISPATCHING" and state == "RUNNING":
            self.db.set_job(job["id"], state="RUNNING")
        return False

    # ------------------------------------------------------------------
    def _probe_now(self, ln) -> None:
        """Force one lane's occupancy probe and update the caches."""
        if not self.prober.enabled or not ln.hosts:
            return
        free = self.locks[ln.name].is_free()
        res = self.prober.probe_lane(ln.name, ln.hosts, free)
        self._probe_cache[ln.name] = res.to_dict()
        if res.disposition is LaneDisposition.FOREIGN:
            self._foreign_since.setdefault(ln.name, time.time())
            self.log(f"lane {ln.name}: FOREIGN after a crashed runner -- "
                     f"{res.detail}. The lock is free but the FPGA is not; "
                     f"this lane will not be scheduled until it is reclaimed "
                     f"(`fq reclaim {ln.name}`).")
        else:
            self._foreign_since.pop(ln.name, None)

    def probe_lanes(self, force: bool = False) -> None:
        """Refresh occupancy for lanes that are not ours, on an interval."""
        interval = float(self.cfg.probe.get("interval_s", 60))
        now = time.time()
        for ln in self.cfg.lanes:
            cached = self._probe_cache.get(ln.name)
            if (not force and cached
                    and now - cached["checked_at"] < interval):
                continue
            free = self.locks[ln.name].is_free()
            res = self.prober.probe_lane(ln.name, ln.hosts, free)
            self._probe_cache[ln.name] = res.to_dict()
            if res.disposition is LaneDisposition.FOREIGN:
                self._foreign_since.setdefault(ln.name, now)
                if cached and cached.get("disposition") != "FOREIGN":
                    self.log(f"lane {ln.name}: FOREIGN -- {res.detail}")
            else:
                if ln.name in self._foreign_since:
                    self.log(f"lane {ln.name}: no longer foreign "
                             f"({res.disposition.value})")
                self._foreign_since.pop(ln.name, None)

            if should_auto_reclaim(self.cfg.reclaim_policy,
                                   self._foreign_since.get(ln.name),
                                   self.cfg.reclaim_grace_s, now):
                self._reclaim(ln.name, "auto policy after grace period")

    def _reclaim(self, lane_name: str, why: str,
                 dry_run: bool = False) -> dict:
        """Kill orphaned sims on a lane.  Takes the lane lock first.

        Taking the flock is not a formality: it is what makes reclamation
        safe against a job being dispatched onto this lane at the same
        moment.  If we cannot take it, an fq job owns the lane and there is
        by definition nothing orphaned to reclaim.
        """
        ln = self.cfg.lane(lane_name)
        if ln is None:
            raise ValueError(f"no such lane: {lane_name}")
        if self.cfg.reclaim_policy == "never":
            return {"lane": lane_name, "reclaimed": False,
                    "detail": "reclaim.policy is 'never'"}
        lock = LaneLock(self.cfg.locks_dir, lane_name)
        if not lock.acquire(job_id=None, user="fq-reclaim"):
            return {"lane": lane_name, "reclaimed": False,
                    "detail": "lane is held by a live fq job -- nothing to "
                              "reclaim (its simulation is not an orphan)"}
        try:
            res = self.prober.probe_lane(lane_name, ln.hosts, lock_free=True)
            if res.disposition is not LaneDisposition.FOREIGN:
                return {"lane": lane_name, "reclaimed": False,
                        "detail": f"lane is {res.disposition.value}, not "
                                  f"FOREIGN; nothing to do"}
                # (lock released in finally)
            if dry_run:
                return {"lane": lane_name, "reclaimed": False,
                        "dry_run": True,
                        "would_kill": res.sims_running,
                        "detail": res.detail}
            self.log(f"RECLAIMING lane {lane_name} ({why}): {res.detail}")
            ok, detail = self.prober.reclaim_lane(ln.hosts)
            self.db.event("reclaim", lane=lane_name,
                          detail=f"{why}: ok={ok} {detail}")
            self._foreign_since.pop(lane_name, None)
            self._probe_cache.pop(lane_name, None)
            self.log(f"lane {lane_name} reclaim {'ok' if ok else 'FAILED'}: "
                     f"{detail}")
            return {"lane": lane_name, "reclaimed": ok, "detail": detail}
        finally:
            lock.release()

    # ==================================================================
    # Dispatch
    # ==================================================================
    def lane_views(self) -> list[LaneView]:
        views = []
        for ln in self.cfg.lanes:
            row = self.db.lane_row(ln.name)
            pr = self._probe_cache.get(ln.name)
            disp = pr["disposition"] if pr else "UNKNOWN"
            # A lane counts as free only if the kernel says the lock is free
            # AND the probe found no foreign simulation.  UNKNOWN is treated
            # as free when probing is disabled, and as not-free otherwise --
            # we would rather leave an FPGA idle than double-book it.
            lock_free = self.locks[ln.name].is_free()
            if not self.prober.enabled or not ln.hosts:
                usable = lock_free
            else:
                usable = lock_free and disp == "FREE"
            views.append(LaneView(
                name=ln.name,
                capacity=ln.capacity,
                free=usable,
                enabled=ln.enabled,
                drain=bool(ln.drain or row.get("drain")),
                trees=tuple(os.path.realpath(t) for t in ln.trees),
                last_tree=row.get("last_tree"),
                last_hw_config=row.get("last_hw_config"),
            ))
        return views

    def job_views(self, rows: list[dict]) -> list[JobView]:
        out = []
        for r in rows:
            spec = json.loads(r["spec_json"] or "{}")
            out.append(JobView(
                id=r["id"], user=r["user"], priority=r["priority"],
                submitted_at=r["submitted_at"],
                num_fpgas=r["num_fpgas"],
                tree=(os.path.realpath(r["tree"]) if r["tree"] else None),
                hw_config=r["hw_config"],
                lane_hint=spec.get("lane_hint"),
            ))
        return out

    def dispatch(self) -> int:
        queued = self.db.queued_jobs()
        if not queued:
            return 0
        p = plan_jobs(
            self.job_views(queued), self.lane_views(), time.time(),
            running_fpgas_by_user=self.db.running_fpgas_by_user(),
            reserve_after_s=self.cfg.reserve_after_s,
            max_fpgas_per_user=self.cfg.max_fpgas_per_user)

        for jid, reason in p.blocked.items():
            self.db.set_job(jid, blocked_reason=reason)

        started = 0
        for pl in p.placements:
            job = self.db.get_job(pl.job_id)
            if not job or job["state"] != "QUEUED":
                continue
            if self._start(job, pl.lane, pl.reason):
                started += 1
        return started

    def _start(self, job: dict, lane_name: str, reason: str) -> bool:
        ln = self.cfg.lane(lane_name)
        assert ln is not None
        lock = LaneLock(self.cfg.locks_dir, lane_name)
        # Take the lock BEFORE touching anything.  If this fails, the plan
        # raced with reality; skip and re-plan next iteration.
        if not lock.acquire(job_id=job["id"], user=job["user"]):
            self.vlog(f"lane {lane_name} taken between plan and dispatch; "
                      f"job {job['id']} stays queued")
            return False
        try:
            spec = json.loads(job["spec_json"] or "{}")
            workdir = pathlib.Path(job["workdir"])
            workdir.mkdir(parents=True, exist_ok=True)
            for stale in ("cancel", "done", "status.json"):
                with contextlib.suppress(OSError):
                    (workdir / stale).unlink()

            runner_plan = {
                "job_id": job["id"],
                "user": job["user"],
                "uid": job["uid"],
                "gid": job["gid"],
                "spec": spec,
                "lane": {
                    "name": ln.name, "mode": ln.mode, "tag": ln.tag,
                    "capacity": ln.capacity, "hosts": list(ln.hosts),
                    "instance_type": ln.instance_type,
                    "manage_hosts": ln.manage_hosts,
                },
                "backend": self.cfg.backend,
                "backend_options": self.cfg.backend_options,
                "timeout_s": job["timeout_s"],
                "infrasetup_timeout_s": int(
                    self.cfg.backend_options.get("infrasetup_timeout_s", 7200)),
                "results_dir": spec.get("results_dir"),
            }
            (workdir / "runner.json").write_text(
                json.dumps(runner_plan, indent=2) + "\n")
            if self.am_root:
                with contextlib.suppress(OSError):
                    os.chown(workdir / "runner.json", job["uid"],
                             job["gid"] if job["gid"] is not None else -1)

            preexec = self._make_preexec(job)
            argv = [sys.executable, "-m", "fq.runner",
                    "--workdir", str(workdir),
                    "--lock-fd", str(lock.fd)]
            env = dict(os.environ)
            env["PYTHONPATH"] = (str(pathlib.Path(__file__).resolve().parent.parent)
                                 + os.pathsep + env.get("PYTHONPATH", ""))
            if self.am_root:
                try:
                    pw = pwd.getpwuid(job["uid"])
                    env.update({"HOME": pw.pw_dir, "USER": pw.pw_name,
                                "LOGNAME": pw.pw_name})
                except KeyError:
                    pass
            proc = subprocess.Popen(
                argv, cwd=str(workdir), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                pass_fds=(lock.fd,) if lock.fd is not None else (),
                preexec_fn=preexec, start_new_session=True)
        except Exception as exc:  # noqa: BLE001
            lock.release()
            self.db.finish_job(job["id"], "FAILED", -1,
                               f"failed to launch runner: {exc}")
            self.log(f"job {job['id']}: failed to launch runner: "
                     f"{traceback.format_exc()}")
            return False

        # Record the runner as the lock's owner, then drop OUR copy of the fd.
        # From here the runner is the sole holder: the lock now lives exactly
        # as long as the job, independent of this daemon.
        from .locks import LockOwner
        lock.write_owner(LockOwner(
            pid=proc.pid, start_ticks=proc_start_ticks(proc.pid),
            job_id=job["id"], user=job["user"], acquired_at=time.time()))
        lock.detach()

        self.db.set_job(job["id"], state="DISPATCHING", lane=lane_name,
                        started_at=time.time(), runner_pid=proc.pid,
                        runner_ticks=proc_start_ticks(proc.pid),
                        phase="STAGE", blocked_reason=None)
        self.db.mark_lane_used(lane_name, job["tree"], job["hw_config"],
                               job["id"])
        self.db.event("dispatched", job["id"], lane_name, reason)
        self.log(f"job {job['id']} -> lane {lane_name} "
                 f"(pid {proc.pid}) [{reason}]")
        return True

    def _make_preexec(self, job: dict):
        """Drop privileges to the submitter, if we are root.

        This is the "Option A" model from firesim-queue: the daemon runs as
        root and each job executes with its submitter's uid, groups, HOME, ssh
        keys and filesystem access.  It means a job can reach the submitter's
        chipyard tree and their ~/firesim.pem, and cannot reach anyone else's.
        """
        if not self.am_root:
            return None
        uid = int(job["uid"])
        if uid == 0:
            return None
        try:
            pw = pwd.getpwuid(uid)
        except KeyError:
            return None
        gid = pw.pw_gid
        name = pw.pw_name

        def _drop() -> None:
            os.setgid(gid)
            os.initgroups(name, gid)
            os.setuid(uid)
            os.umask(0o022)
        return _drop

    # ==================================================================
    def run(self) -> int:
        self._singleton_lock()
        self.log(f"fq daemon starting: pool={self.cfg.source_path} "
                 f"backend={self.cfg.backend} lanes={len(self.cfg.lanes)} "
                 f"root={self.am_root}")
        if not self.am_root:
            self.log("NOTE: not running as root -- jobs will execute as "
                     f"{getpass.getuser()}, and foreign submitters are "
                     "refused unless allow_foreign_uid_without_root is set.")
        self.cfg.pid_path.write_text(f"{os.getpid()}\n")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.stop.set())

        api = threading.Thread(target=self.serve_socket, daemon=True)
        api.start()

        # First pass: adopt/reconcile before anything is dispatched.
        self.adopt()
        self.reconcile()
        self.probe_lanes(force=True)

        last_probe = time.time()
        while not self.stop.is_set():
            try:
                self.reconcile()
                if time.time() - last_probe >= float(
                        self.cfg.probe.get("interval_s", 60)):
                    self.probe_lanes()
                    last_probe = time.time()
                self.dispatch()
            except Exception:  # noqa: BLE001
                # One bad iteration must never take the daemon down: the
                # daemon dying is what strands FPGAs.
                self.log(f"dispatch loop error:\n{traceback.format_exc()}")
            if not api.is_alive() and not self.stop.is_set():
                # Same reasoning as above: a dead API thread is invisible from
                # the outside because dispatch keeps working. Bring it back.
                self.log("api thread is dead; restarting listener")
                api = threading.Thread(target=self.serve_socket, daemon=True)
                api.start()
            self.stop.wait(self.cfg.poll_interval_s)

        self.log("draining: daemon stopping. Running jobs keep their lanes "
                 "and are re-adopted on restart.")
        with contextlib.suppress(OSError):
            self.cfg.pid_path.unlink()
        if self.sock:
            with contextlib.suppress(OSError):
                self.sock.close()
        return 0

    def _singleton_lock(self) -> None:
        """Refuse to start a second daemon on the same state dir."""
        import fcntl
        path = self.cfg.state_dir / "daemon.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise SystemExit(
                f"another fq daemon is already running on {self.cfg.state_dir} "
                f"(see {self.cfg.pid_path})")
        os.write(fd, f"{os.getpid()}\n".encode())
        # Held for the process lifetime; intentionally never closed.
        self._singleton_fd = fd
