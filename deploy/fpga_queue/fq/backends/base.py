"""Backend interface."""

from __future__ import annotations

import dataclasses
import hashlib
import uuid
import enum
import pathlib
from typing import Any, Optional


class Phase(str, enum.Enum):
    """Job lifecycle phases, in order.

    ``KILL`` and ``COLLECT`` are always executed, including after a failure,
    a cancel or a timeout -- freeing the hardware is not conditional on the
    job having gone well.
    """

    STAGE = "STAGE"
    LAUNCH = "LAUNCH"          # optional: bring up the run farm
    INFRASETUP = "INFRASETUP"  # build driver + program FPGAs (the slow one)
    RUN = "RUN"                # firesim runworkload
    KILL = "KILL"              # always
    COLLECT = "COLLECT"        # always
    TERMINATE = "TERMINATE"    # optional: tear the run farm down


@dataclasses.dataclass
class LaneInfo:
    """The slice of lane config a backend needs."""

    name: str
    mode: str = "hosts"            # "hosts" (ExternallyProvisioned) | "tagged"
    tag: Optional[str] = None
    capacity: int = 1
    hosts: tuple[str, ...] = ()
    instance_type: Optional[str] = None
    manage_hosts: bool = False


@dataclasses.dataclass
class JobContext:
    """Everything a backend needs to build commands for one job."""

    job_id: int
    user: str
    uid: int
    gid: Optional[int]
    spec: dict[str, Any]
    lane: LaneInfo
    workdir: pathlib.Path          # per-job artifact dir (owned by submitter)
    results_dir: Optional[pathlib.Path] = None

    # -- convenience accessors -------------------------------------------
    @property
    def tree(self) -> pathlib.Path:
        return pathlib.Path(self.spec["tree"])

    @property
    def workload(self) -> str:
        return self.spec.get("workload") or f"fq-job-{self.job_id}"

    @property
    def run_tag(self) -> str:
        """Globally-unique, STABLE suffix for this job's results directory.

        ``job_id`` alone is not unique: ids are reused after a daemon restart,
        and FireSim names results dirs ``<ts>-<workload>-<suffix_tag>``.  Two
        jobs sharing an id then collide, and COLLECT can pick up another job's
        results -- silently handing one agent another agent's profile.

        The tag is PERSISTED rather than derived from the workdir's stat: an
        earlier attempt hashed ``st_ctime_ns``, which changes every time a file
        is written into the directory, so setup() and results_globs() computed
        different tags and the COLLECT glob matched nothing.

        If the token cannot be written we fall back to the plain job id --
        no worse than the historical behaviour, and stable, which is what
        actually matters for the glob.
        """
        path = self.workdir / "run_tag"
        try:
            existing = path.read_text().strip()
            if existing:
                return existing
        except OSError:
            pass
        tag = f"fq{self.job_id}x{uuid.uuid4().hex[:8]}"
        try:
            path.write_text(tag + "\n")
        except OSError:
            return f"fq{self.job_id}"
        return tag

    @property
    def hw_config(self) -> Optional[str]:
        return self.spec.get("hw_config")

    @property
    def num_fpgas(self) -> int:
        return int(self.spec.get("num_fpgas", 1))

    @property
    def elf(self) -> Optional[str]:
        return self.spec.get("elf")

    @property
    def runtime_yaml(self) -> pathlib.Path:
        return self.workdir / "config_runtime.yaml"


class Backend:
    """Base class.  Subclasses override what they need."""

    name = "base"

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = dict(options or {})

    # -- submit-time validation ------------------------------------------
    def validate(self, spec: dict, cfg: Any) -> list[str]:
        """Return a list of problems.  Empty list == acceptable spec.

        Run by the daemon *before* the job is enqueued, so that a bad spec
        fails at the submitting terminal instead of an hour later on a lane.
        """
        return []

    # -- execution --------------------------------------------------------
    def stage(self, ctx: JobContext) -> None:
        """Prepare on-disk inputs.  Runs as the submitter, inside the runner."""

    def argv(self, ctx: JobContext, phase: Phase) -> Optional[list[str]]:
        """Command for `phase`, or None if this backend skips that phase."""
        return None

    def uartlog_argv(self, ctx: JobContext) -> Optional[list[str]]:
        """Command that prints the current UART output, for completion
        detection.  None disables regex-based completion for this backend."""
        return None

    def describe(self) -> dict:
        return {"backend": self.name, "options": self.options}
