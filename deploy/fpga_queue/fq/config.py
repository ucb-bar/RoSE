"""Pool configuration: lanes, defaults, discovery.

A *lane* is the unit of scheduling: an exclusively-lockable set of FPGAs that
one job gets to itself.  Choosing what a lane *is* is the central design
decision of this daemon, and FireSim constrains it hard.

FireSim's manager does not address individual FPGAs.  It addresses a **run
farm**, described by a run-farm recipe.  There are two recipes, and they give
two very different lane models:

``mode: hosts``  (recipe ``externally_provisioned.yaml``, the DEFAULT)
    ``run_farm_type: ExternallyProvisioned`` takes an explicit list of
    hostnames/IPs -- ``run_farm_hosts_to_use: - "192.168.0.8": one_fpga_spec``
    -- and **no tag at all**.  It never launches and never terminates
    anything; the recipe's own comment is "Unmanaged list of run farm hosts.
    Assumed that they are pre-setup to run simulations."

    This is the mode to use for a persistent pool, and it is the default here,
    for two reasons:

      1. It makes the tag-collision hazard *structurally impossible*.  There
         is no ``run_farm_tag``, so there is no way for one job to release
         another job's hosts.  Our eight F2 instances all carry the same
         ``fsimcluster=rosef2run`` tag; under the tagged model they would be
         one indivisible eight-FPGA resource, and any second concurrent job
         could terminate the first one's instances.  Under this model they
         split cleanly into eight independent single-host lanes.
      2. It cannot spend money or destroy hardware.  A bug in this daemon
         cannot terminate an instance, because the code path does not exist.

``mode: tagged``  (recipe ``aws_ec2.yaml``, opt-in)
    ``run_farm_type: AWSEC2F2`` launches and terminates hosts *by tag*.  A
    lane in this mode owns one tag exclusively and may manage its own fleet.
    Two lanes sharing a tag is the single most destructive misconfiguration
    available, so it is rejected at config load time, not at dispatch.

A lane's ``capacity`` is its FPGA count, which bounds the ``no_net_num_nodes``
of any job placed on it.

Config is YAML (or JSON -- chosen by file extension) so it can be checked into
a repo and reviewed.  See ``examples/pool.example.yaml``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any, Optional

# Tags that must never be used by a lane.  These are the fleets that
# non-queue humans drive by hand; if the queue ever launched or terminated
# hosts under one of them it would take out somebody's manual run.
DEFAULT_FORBIDDEN_TAGS = ("mainrunfarm", "buildfarm", "default")


class ConfigError(Exception):
    """Raised for a pool config that cannot be safely used."""


@dataclasses.dataclass
class Lane:
    """One exclusively-lockable slice of the FPGA pool."""

    name: str
    # "hosts"  -> ExternallyProvisioned, explicit host list, no tag, never
    #             launches/terminates. Safe default for a persistent pool.
    # "tagged" -> AWSEC2F2, owns run_farm_tag exclusively, may manage a fleet.
    mode: str = "hosts"
    tag: Optional[str] = None
    capacity: int = 1
    # Explicit run-farm hosts (IPs or hostnames). Required in "hosts" mode.
    hosts: tuple[str, ...] = ()
    # Chipyard trees whose bitstreams may run here.  ``None``/empty means
    # "any tree".  This exists because a bitstream must be driven from the
    # tree that built it (driver/bitstream deploy quintuplet must match).
    trees: tuple[str, ...] = ()
    instance_type: Optional[str] = None
    # Only meaningful in "tagged" mode: may this lane launch/terminate its
    # own EC2 fleet?  Off by default -- a persistent pool is operator-owned.
    manage_hosts: bool = False
    enabled: bool = True
    # Operator switch: a draining lane finishes its current job and accepts
    # nothing new.  Used to take hardware out of service without killing work.
    drain: bool = False
    notes: str = ""

    def accepts_tree(self, tree: Optional[str]) -> bool:
        if not self.trees:
            return True
        if tree is None:
            return False
        return os.path.realpath(tree) in {os.path.realpath(t) for t in self.trees}


@dataclasses.dataclass
class PoolConfig:
    state_dir: pathlib.Path
    socket_path: pathlib.Path
    lanes: tuple[Lane, ...]
    backend: str = "firesim"
    backend_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    admins: tuple[str, ...] = ()
    # Scheduling knobs -- see fq/scheduler.py for exactly how each is used.
    default_priority: int = 5
    default_timeout_s: int = 3600
    default_fail_regex: Optional[str] = None
    max_timeout_s: int = 24 * 3600
    reserve_after_s: int = 900
    max_fpgas_per_user: int = 0          # 0 = uncapped
    max_queued_per_user: int = 200
    poll_interval_s: float = 5.0
    # Multi-tenancy.  When the daemon runs as root it drops privileges to the
    # submitter for every job (the "Option A" model inherited from
    # firesim-queue).  When it does not run as root it can only run jobs as
    # itself, and refuses foreign submitters unless this is set.
    allow_foreign_uid_without_root: bool = False
    forbidden_tags: tuple[str, ...] = DEFAULT_FORBIDDEN_TAGS
    discovery: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Occupancy probing + orphan reclamation -- see fq/occupancy.py.
    probe: dict[str, Any] = dataclasses.field(default_factory=dict)
    reclaim_policy: str = "manual"      # manual | auto | never
    reclaim_grace_s: int = 1800
    source_path: Optional[pathlib.Path] = None

    def lane(self, name: str) -> Optional[Lane]:
        for ln in self.lanes:
            if ln.name == name:
                return ln
        return None

    @property
    def jobs_dir(self) -> pathlib.Path:
        return self.state_dir / "jobs"

    @property
    def locks_dir(self) -> pathlib.Path:
        return self.state_dir / "lanes"

    @property
    def db_path(self) -> pathlib.Path:
        return self.state_dir / "queue.db"

    @property
    def log_path(self) -> pathlib.Path:
        return self.state_dir / "daemon.log"

    @property
    def pid_path(self) -> pathlib.Path:
        return self.state_dir / "daemon.pid"


def _load_raw(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".json",):
        return json.loads(text)
    try:
        import yaml  # noqa: PLC0415  (optional dependency)
    except ImportError as exc:  # pragma: no cover - depends on host
        raise ConfigError(
            f"{path} is YAML but PyYAML is not installed; either install it or "
            f"write the pool config as JSON") from exc
    data = yaml.safe_load(text)
    if data is None:
        raise ConfigError(f"{path} is empty")
    return data


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def load_pool(path: str | os.PathLike[str]) -> PoolConfig:
    """Load and validate a pool config.

    Validation is deliberately strict and happens at load time rather than at
    dispatch time: a misconfigured tag is the one mistake in this system that
    can destroy someone else's running work, so it must be impossible to start
    the daemon with one.
    """
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise ConfigError(f"pool config not found: {p}")
    raw = _load_raw(p)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")

    state_dir = raw.get("state_dir")
    if not state_dir:
        raise ConfigError(f"{p}: 'state_dir' is required")
    state_dir = pathlib.Path(os.path.expanduser(str(state_dir))).resolve()

    sock = raw.get("socket_path") or (state_dir / "fq.sock")
    sock = pathlib.Path(os.path.expanduser(str(sock)))

    lanes_raw = raw.get("lanes") or []
    if not isinstance(lanes_raw, list) or not lanes_raw:
        raise ConfigError(f"{p}: 'lanes' must be a non-empty list")

    forbidden = _as_tuple(raw.get("forbidden_tags")) or DEFAULT_FORBIDDEN_TAGS

    lanes: list[Lane] = []
    seen_names: set[str] = set()
    seen_tags: dict[str, str] = {}
    seen_hosts: dict[str, str] = {}
    for i, ln in enumerate(lanes_raw):
        if not isinstance(ln, dict):
            raise ConfigError(f"{p}: lanes[{i}] must be a mapping")
        name = str(ln.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{p}: lanes[{i}] has no 'name'")
        if name in seen_names:
            raise ConfigError(f"{p}: duplicate lane name {name!r}")
        mode = str(ln.get("mode") or "hosts").strip()
        if mode not in ("hosts", "tagged"):
            raise ConfigError(
                f"{p}: lane {name!r} has mode {mode!r}; must be 'hosts' "
                f"(externally_provisioned) or 'tagged' (aws_ec2)")
        tag = ln.get("tag")
        tag = str(tag).strip() if tag else None
        hosts = _as_tuple(ln.get("hosts"))

        if mode == "tagged":
            if not tag:
                raise ConfigError(
                    f"{p}: lane {name!r} is mode 'tagged' but has no 'tag'")
            # THE destructive misconfiguration.  FireSim's AWSEC2F2 run farm
            # binds hosts by `tag:fsimcluster=<run_farm_tag>` and takes the
            # first N sorted by private IP -- so two lanes on one tag select
            # the SAME instances, stomp the same sim_slot_0 / screen fsim0 /
            # FPGA, and `terminaterunfarm` on either kills both.
            if tag in seen_tags:
                raise ConfigError(
                    f"{p}: lanes {seen_tags[tag]!r} and {name!r} share "
                    f"run_farm_tag {tag!r}. Tags must be unique per lane -- "
                    f"FireSim binds AND terminates run-farm hosts by tag, so "
                    f"a shared tag lets one job destroy another's instances "
                    f"and run on its FPGA.")
            if tag in forbidden:
                raise ConfigError(
                    f"{p}: lane {name!r} uses reserved tag {tag!r}; that tag "
                    f"is on the forbidden list (hand-driven fleets live there)")
            seen_tags[tag] = name
        else:  # hosts
            if not hosts:
                raise ConfigError(
                    f"{p}: lane {name!r} is mode 'hosts' but lists no 'hosts'. "
                    f"ExternallyProvisioned needs explicit hostnames/IPs.")

        # Host disjointness -- required in BOTH modes, and for a reason that
        # is easy to get wrong:
        #
        #   * `firesim kill` runs `pkill -SIGKILL FireSim-f2` on the host.
        #     That is host-wide: it kills every slot's driver, not just ours.
        #   * `infrasetup` reflashes every slot up to the host's num_fpgas,
        #     writing a dummy AGFI into slots it is not using.
        #
        # So a job on a host is inherently destructive to every other job on
        # that host.  The lane granularity therefore has to be the whole host;
        # two lanes may never name the same one.
        for h in hosts:
            if h in seen_hosts:
                raise ConfigError(
                    f"{p}: host {h!r} appears in both lane {seen_hosts[h]!r} "
                    f"and lane {name!r}. Hosts must belong to exactly one "
                    f"lane: `firesim kill` pkills FireSim-f2 host-wide and "
                    f"`infrasetup` reflashes every slot on the host, so two "
                    f"jobs sharing a host will destroy each other.")
            seen_hosts[h] = name

        cap = int(ln.get("capacity", len(hosts) or 1))
        if cap < 1:
            raise ConfigError(f"{p}: lane {name!r} capacity must be >= 1")
        seen_names.add(name)
        lanes.append(Lane(
            name=name,
            mode=mode,
            tag=tag,
            capacity=cap,
            hosts=hosts,
            trees=_as_tuple(ln.get("trees")),
            instance_type=ln.get("instance_type"),
            manage_hosts=bool(ln.get("manage_hosts", False)),
            enabled=bool(ln.get("enabled", True)),
            drain=bool(ln.get("drain", False)),
            notes=str(ln.get("notes") or ""),
        ))
        if lanes[-1].manage_hosts and mode != "tagged":
            raise ConfigError(
                f"{p}: lane {name!r} sets manage_hosts but is mode 'hosts'; "
                f"ExternallyProvisioned cannot launch or terminate instances "
                f"(launch_run_farm/terminate_run_farm are no-ops).")

    defaults = raw.get("defaults") or {}
    sched = raw.get("scheduling") or {}

    cfg = PoolConfig(
        state_dir=state_dir,
        socket_path=sock,
        lanes=tuple(lanes),
        backend=str(raw.get("backend", "firesim")),
        backend_options=dict(raw.get("backend_options") or {}),
        admins=_as_tuple(raw.get("admins")),
        default_priority=int(defaults.get("priority", 5)),
        default_timeout_s=int(defaults.get("timeout_s", 3600)),
        default_fail_regex=defaults.get("fail_regex") or None,
        max_timeout_s=int(defaults.get("max_timeout_s", 24 * 3600)),
        reserve_after_s=int(sched.get("reserve_after_s", 900)),
        max_fpgas_per_user=int(sched.get("max_fpgas_per_user", 0)),
        max_queued_per_user=int(sched.get("max_queued_per_user", 200)),
        poll_interval_s=float(sched.get("poll_interval_s", 5.0)),
        allow_foreign_uid_without_root=bool(
            raw.get("allow_foreign_uid_without_root", False)),
        forbidden_tags=forbidden,
        discovery=dict(raw.get("discovery") or {}),
        probe=dict(raw.get("probe") or {}),
        reclaim_policy=str((raw.get("reclaim") or {}).get("policy", "manual")),
        reclaim_grace_s=int((raw.get("reclaim") or {}).get("grace_s", 1800)),
        source_path=p,
    )

    if cfg.reclaim_policy not in ("manual", "auto", "never"):
        raise ConfigError(
            f"{p}: reclaim.policy must be one of manual|auto|never, "
            f"got {cfg.reclaim_policy!r}")

    total = sum(ln.capacity for ln in cfg.lanes if ln.enabled)
    quota = int(raw.get("fpga_quota", 0))
    if quota and total > quota:
        raise ConfigError(
            f"{p}: lanes declare {total} FPGAs but fpga_quota is {quota}. "
            f"AWS will refuse the instances; fix the pool rather than "
            f"discovering it at launch time.")
    return cfg


def default_pool_path() -> Optional[pathlib.Path]:
    """Where to look for a pool config when ``--pool`` is not given."""
    env = os.environ.get("FQ_POOL")
    if env:
        return pathlib.Path(env).expanduser()
    for cand in (
        pathlib.Path("/etc/fq/pool.yaml"),
        pathlib.Path.home() / ".config" / "fq" / "pool.yaml",
    ):
        if cand.exists():
            return cand
    return None
