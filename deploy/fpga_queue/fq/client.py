"""Client library.

Any independent process can drive the queue with this -- a shell script via
the ``fq`` CLI, a Python harness via ``FqClient``, or a CI job.  Nothing here
needs privileges beyond connect access to the socket.

    from fq.client import FqClient, JobSpec

    c = FqClient()
    jid = c.submit(JobSpec(
        tree="/home/ubuntu/chipyard-rose",
        hw_config="f2_saturn_refv256d128_singlecore_60mhz-fast",
        elf="/path/to/zephyr.elf",
        timeout_s=1800,
        done_regex=r"Hello World",
        results_dir="/scratch/me/results/run1",
    ))
    job = c.wait(jid)
    print(job["state"], job["exit_code"])
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import socket
import time
from typing import Any, Callable, Optional

from . import proto


class FqError(Exception):
    """Daemon rejected a request, or the daemon is not reachable."""


DEFAULT_SOCKETS = (
    "/var/run/fq/fq.sock",
    "/var/lib/fq/fq.sock",
)


def default_socket_path() -> str:
    env = os.environ.get("FQ_SOCKET")
    if env:
        return env
    for cand in DEFAULT_SOCKETS:
        if pathlib.Path(cand).exists():
            return cand
    return DEFAULT_SOCKETS[0]


@dataclasses.dataclass
class JobSpec:
    """Everything needed to run one FireSim job.

    ``tree`` is not optional-by-accident: a FireSim bitstream must be driven
    from the chipyard tree that built it, because ``infrasetup`` compiles the
    simulation driver from that tree and the driver must match the bitstream's
    deploy quintuplet.  Tracking only the AGFI would let the queue produce
    silently-wrong runs, so the tree is part of the job's identity.
    """

    # --- hardware -------------------------------------------------------
    tree: str                                  # chipyard root that owns the bitstream
    hw_config: Optional[str] = None            # hwdb key in that tree
    agfi: Optional[str] = None                 # optional explicit AGFI (cross-checked)
    num_fpgas: int = 1

    # --- guest ----------------------------------------------------------
    elf: Optional[str] = None                  # guest ELF staged into the workload dir
    workload: Optional[str] = None             # workload name; default derived from job id
    rootfs: Optional[str] = None               # None => bare metal

    # --- runtime --------------------------------------------------------
    runtime_args: dict[str, Any] = dataclasses.field(default_factory=dict)
    timeout_s: int = 0                         # 0 => pool default
    # How this job ends.  Default is "exit": the guest terminates the
    # simulation itself (Zephyr sys_reboot -> HTIF exit -> runworkload
    # returns).  Use "sentinel" with a regex for guests that cannot exit;
    # use "timeout" for a deliberately fixed-duration run.
    #     {"mode": "exit"|"sentinel"|"timeout",
    #      "sentinel_regex": str, "fail_regex": str, "timeout_s": int}
    # The timeout is always armed as a backstop.  See fq/completion.py.
    completion: Optional[dict[str, Any]] = None
    # Flat shorthands, accepted for convenience; completion{} wins.
    done_regex: Optional[str] = None
    fail_regex: Optional[str] = None

    # --- queueing -------------------------------------------------------
    priority: int = 5
    lane_hint: Optional[str] = None
    project: Optional[str] = None
    comment: Optional[str] = None

    # --- output ---------------------------------------------------------
    results_dir: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v is not None}


class FqClient:
    def __init__(self, socket_path: Optional[str] = None, timeout: float = 30.0):
        self.socket_path = socket_path or default_socket_path()
        self.timeout = timeout

    # -- transport --------------------------------------------------------
    def _call(self, op: str, **kwargs: Any) -> Any:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect(self.socket_path)
        except OSError as exc:
            raise FqError(
                f"cannot reach the fq daemon at {self.socket_path}: {exc}. "
                f"Is it running? (`fq daemon --pool <pool.yaml>`)") from exc
        try:
            proto.send_msg(s, {"op": op, **kwargs})
            resp = proto.recv_msg(s)
        finally:
            s.close()
        if resp is None:
            raise FqError("daemon closed the connection without replying")
        if not resp.get("ok"):
            raise FqError(resp.get("error", "unknown error"))
        return resp.get("result")

    # -- operations -------------------------------------------------------
    def ping(self) -> dict:
        return self._call("ping")

    def submit(self, spec: JobSpec | dict) -> int:
        d = spec.to_dict() if isinstance(spec, JobSpec) else dict(spec)
        # cwd travels with the spec so the daemon can resolve relative paths
        # and infer a project tag the same way for every client.
        d.setdefault("cwd", os.getcwd())
        return int(self._call("submit", spec=d)["job_id"])

    def get(self, job_id: int) -> dict:
        return self._call("get", job_id=job_id)

    def list(self, *, user: Optional[str] = None, state: Optional[str] = None,
             all: bool = False, limit: int = 200) -> list[dict]:
        return self._call("list", user=user, state=state, all=all, limit=limit)

    def cancel(self, job_id: int) -> dict:
        return self._call("cancel", job_id=job_id)

    def signal_done(self, job_id: int) -> dict:
        """Tell a running job it is finished.

        The explicit completion signal, for harnesses that know when their
        work is done and whose guest cannot exit on its own.
        """
        return self._call("signal_done", job_id=job_id)

    def lanes(self) -> list[dict]:
        return self._call("lanes")

    def logs(self, job_id: int, stream: str = "stdout", offset: int = 0,
             max_bytes: int = 262144) -> dict:
        return self._call("logs", job_id=job_id, stream=stream,
                          offset=offset, max_bytes=max_bytes)

    def events(self, job_id: int) -> list[dict]:
        return self._call("events", job_id=job_id)

    def drain(self, lane: str, on: bool = True) -> dict:
        return self._call("drain", lane=lane, on=on)

    # -- convenience ------------------------------------------------------
    def wait(self, job_id: int, poll: float = 2.0,
             timeout: Optional[float] = None,
             on_update: Optional[Callable[[dict], None]] = None) -> dict:
        """Block until the job reaches a terminal state; return the job row."""
        deadline = (time.time() + timeout) if timeout else None
        last = None
        while True:
            job = self.get(job_id)
            key = (job.get("state"), job.get("phase"))
            if on_update and key != last:
                on_update(job)
            last = key
            if job.get("state") in ("DONE", "FAILED", "CANCELLED", "TIMEOUT"):
                return job
            if deadline and time.time() > deadline:
                raise FqError(f"timed out waiting for job {job_id}")
            time.sleep(poll)

    def submit_and_wait(self, spec: JobSpec | dict, **kw: Any) -> dict:
        return self.wait(self.submit(spec), **kw)
