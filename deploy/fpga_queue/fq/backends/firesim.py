"""The real FireSim backend.

Everything this backend does to a chipyard tree is either a read, or a write
confined to files it created for this one job.  That is deliberate: the trees
are shared with humans running FireSim by hand, and a queue that edits
``config_runtime.yaml`` or appends to ``config_hwdb.yaml`` under them would
corrupt manual work and race with itself.

Two FireSim CLI flags make that possible:

  ``firesim -c <per-job config_runtime.yaml>``   (``--runtimeconfigfile``)
  ``firesim -a <per-job config_hwdb.yaml>``      (``--hwdbconfigfile``)

so each job runs against a private copy of both files, rendered into its own
job directory.  Notably this removes the unlocked, non-atomic
``cat built-hwdb-entries/x >> config_hwdb.yaml`` append that the manual flow
uses -- a job that supplies a raw AGFI gets a synthetic hwdb entry in its
private copy instead of mutating the shared one.

Run farm
--------
Lanes in ``hosts`` mode render ``base_recipe:
run-farm-recipes/externally_provisioned.yaml`` with the lane's own explicit
host list.  ``launchrunfarm``, ``terminaterunfarm`` and
``terminate_on_completion`` are all no-ops under that recipe, so this backend
physically cannot terminate an EC2 instance.  Lanes in ``tagged`` mode render
``aws_ec2.yaml`` with the lane's exclusive ``run_farm_tag``.

Completion
----------
``runworkload`` writes one file per finished job into
``results-workload/<UTC-ts>-<workload>-<suffix_tag>/.monitoring-dir/``.  That
directory is the cleanest completion signal available, so the runner watches
it.  For bare-metal guests that never finish, the uartlog sentinel regex and
the wall-clock timeout are the real terminators; see fq/runner.py.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
from typing import Any, Optional

from .base import Backend, JobContext, Phase

# Maps an FPGA count to the spec name that externally_provisioned.yaml
# already defines.  The recipe ships specs for 1..16 FPGAs.
_SPEC_WORDS = {
    1: "one_fpga_spec", 2: "two_fpgas_spec", 3: "three_fpgas_spec",
    4: "four_fpgas_spec", 5: "five_fpgas_spec", 6: "six_fpgas_spec",
    7: "seven_fpgas_spec", 8: "eight_fpgas_spec",
}


def _quote_glob(pattern: str) -> str:
    """Quote a glob so the shell can still expand it.

    ``shlex.quote()`` wraps the whole string in single quotes, which turns a
    wildcard into a literal: the loop then matches nothing and COLLECT
    silently copies no results, leaving ``cp: cannot stat '...*-fq-job-N'``
    in every job log -- successes included.  Quote the directory portion and
    leave the wildcard segment bare.
    """
    directory, sep, base = pattern.rpartition("/")
    if not sep:
        return base
    return f"{shlex.quote(directory)}/{base}"


class FiresimBackend(Backend):
    name = "firesim"

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        # How to enter a chipyard tree's manager environment.  Sourced with
        # conda on PATH first and WITHOUT a pipe -- a pipe subshells env.sh so
        # PATH never applies and sbt then runs on the system JDK and dies.
        self.conda_path: str = self.options.get(
            "conda_path", "/home/ubuntu/miniconda3/bin:"
                          "/home/ubuntu/miniconda3/condabin")
        self.default_platform: str = self.options.get(
            "default_platform", "EC2InstanceDeployManager")
        self.simulation_dir: str = self.options.get(
            "simulation_dir", "/home/ubuntu")
        self.fpga_db: str = self.options.get("fpga_db", "/opt/firesim-db.json")

    # ------------------------------------------------------------------
    # submit-time validation
    # ------------------------------------------------------------------
    def validate(self, spec: dict, cfg: Any) -> list[str]:
        errs: list[str] = []
        tree = spec.get("tree")
        if not tree:
            errs.append("'tree' is required: a bitstream must be driven from "
                        "the chipyard tree that built it")
            return errs
        troot = pathlib.Path(tree)
        deploy = troot / "sims" / "firesim" / "deploy"
        if not (troot / "env.sh").exists():
            errs.append(f"{troot} does not look like a chipyard tree "
                        f"(no env.sh)")
        if not deploy.is_dir():
            errs.append(f"{deploy} does not exist")
            return errs

        hw = spec.get("hw_config")
        agfi = spec.get("agfi")
        if not hw and not agfi:
            errs.append("one of 'hw_config' (an hwdb key) or 'agfi' is required")

        if hw:
            entry = self._read_hwdb_entry(deploy, hw)
            if entry is None:
                errs.append(
                    f"hw_config {hw!r} is not in "
                    f"{deploy/'config_hwdb.yaml'}. An AGFI being 'available' "
                    f"in EC2 is not enough -- it also needs an hwdb entry. "
                    f"Check {deploy/'built-hwdb-entries'} for a fragment to "
                    f"merge.")
            elif agfi and entry.get("agfi") and entry["agfi"] != agfi:
                # A mismatch here means the caller believes it is running one
                # bitstream and the tree would run another.  Never guess.
                errs.append(
                    f"hw_config {hw!r} maps to {entry['agfi']} in this tree, "
                    f"but the job specifies agfi {agfi}. Refusing: the driver "
                    f"built by infrasetup must match the bitstream.")

        elf = spec.get("elf")
        if elf and not pathlib.Path(elf).exists():
            errs.append(f"elf not found: {elf}")

        n = int(spec.get("num_fpgas", 1))
        if n < 1:
            errs.append("num_fpgas must be >= 1")
        return errs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: pathlib.Path) -> Any:
        import yaml
        with path.open() as fh:
            return yaml.safe_load(fh)

    @staticmethod
    def _dump_yaml(obj: Any, path: pathlib.Path) -> None:
        import yaml
        with path.open("w") as fh:
            yaml.safe_dump(obj, fh, default_flow_style=False, sort_keys=False)

    def _read_hwdb_entry(self, deploy: pathlib.Path,
                         key: str) -> Optional[dict]:
        f = deploy / "config_hwdb.yaml"
        if not f.exists():
            return None
        try:
            data = self._load_yaml(f) or {}
        except Exception:  # noqa: BLE001 - a malformed hwdb is a config error
            return None
        v = data.get(key)
        return v if isinstance(v, dict) else None

    def _runtime_template(self, deploy: pathlib.Path) -> Optional[pathlib.Path]:
        """Pick a complete config_runtime to use as the base.

        FireSim reads every documented runtime field with a bare subscript, so
        a missing section is a KeyError rather than a default.  We therefore
        always start from a real, complete file and patch it, never synthesise
        one from scratch.
        """
        explicit = self.options.get("runtime_template")
        if explicit:
            p = pathlib.Path(explicit)
            if p.exists():
                return p
        for cand in (
            deploy / "config_runtime.yaml",
            deploy / "sample-backup-configs" / "sample_config_runtime.yaml",
        ):
            if cand.exists():
                return cand
        return None

    # ------------------------------------------------------------------
    # staging
    # ------------------------------------------------------------------
    def stage(self, ctx: JobContext) -> None:
        deploy = ctx.tree / "sims" / "firesim" / "deploy"
        workload = ctx.workload

        # --- 1. workload dir + boot binary ------------------------------
        # `common_bootbinary` is resolved as workload_input_base_dir +
        # common_bootbinary, i.e. relative to workloads/<workload_name>/.
        # Prefixing it with the workload name double-nests and infrasetup dies
        # in rsync -- so the value is the bare basename.
        wl_dir = deploy / "workloads" / workload
        wl_dir.mkdir(parents=True, exist_ok=True)
        bootbin = "guest.elf"
        if ctx.elf:
            src = pathlib.Path(ctx.elf)
            bootbin = src.name
            shutil.copy2(src, wl_dir / bootbin)

        wl_json = {
            "benchmark_name": workload,
            "common_bootbinary": bootbin,
            "common_rootfs": ctx.spec.get("rootfs"),   # None => bare metal
            "common_simulation_outputs": ["uartlog"],
            "no_post_run_hook": True,
        }
        (deploy / "workloads" / f"{workload}.json").write_text(
            json.dumps(wl_json, indent=2) + "\n")
        (ctx.workdir / "workload.json").write_text(
            json.dumps(wl_json, indent=2) + "\n")

        # --- 2. private hwdb -------------------------------------------
        # Copy the tree's hwdb, and if the job named a raw AGFI with no hwdb
        # key, synthesise an entry for it.  Never append to the shared file:
        # that append is unlocked and non-atomic in the manual flow.
        hw_key = ctx.hw_config
        src_hwdb = deploy / "config_hwdb.yaml"
        hwdb: dict = {}
        if src_hwdb.exists():
            hwdb = self._load_yaml(src_hwdb) or {}
        if not hw_key:
            hw_key = f"fq_job_{ctx.job_id}"
            hwdb[hw_key] = {
                "agfi": ctx.spec["agfi"],
                "deploy_quintuplet_override": ctx.spec.get(
                    "deploy_quintuplet_override"),
                "custom_runtime_config": None,
            }
            ctx.spec["hw_config"] = hw_key
        # `custom_runtime_config` is read with a bare subscript by
        # RuntimeHWConfig, so an entry missing it raises KeyError.
        ent = hwdb.get(hw_key)
        if isinstance(ent, dict) and "custom_runtime_config" not in ent:
            ent["custom_runtime_config"] = None
        self._dump_yaml(hwdb, ctx.workdir / "config_hwdb.yaml")

        # --- 3. private config_runtime ---------------------------------
        tmpl = self._runtime_template(deploy)
        if tmpl is None:
            raise RuntimeError(
                f"no config_runtime template found under {deploy}; "
                f"cannot render a complete runtime config")
        cfg = self._load_yaml(tmpl) or {}
        cfg.setdefault("target_config", {})
        cfg.setdefault("workload", {})

        lane = ctx.lane
        if lane.mode == "tagged":
            cfg["run_farm"] = {
                "base_recipe": "run-farm-recipes/aws_ec2.yaml",
                "recipe_arg_overrides": {
                    "run_farm_tag": lane.tag,
                    "run_instance_market": "ondemand",
                    "always_expand_run_farm": True,
                    "default_simulation_dir": self.simulation_dir,
                    "run_farm_hosts_to_use": [
                        {lane.instance_type or "f2.6xlarge": len(lane.hosts)
                         or lane.capacity},
                    ],
                },
            }
        else:
            spec_name = _SPEC_WORDS.get(
                self.options.get("fpgas_per_host", 1), "one_fpga_spec")
            cfg["run_farm"] = {
                "base_recipe": "run-farm-recipes/externally_provisioned.yaml",
                "recipe_arg_overrides": {
                    "default_platform": self.default_platform,
                    "default_simulation_dir": self.simulation_dir,
                    "default_fpga_db": self.fpga_db,
                    # Explicit hosts: this lane's, and only this lane's.
                    "run_farm_hosts_to_use": [
                        {h: spec_name} for h in lane.hosts],
                },
            }

        tc = cfg["target_config"]
        tc["topology"] = "no_net_config"
        tc["no_net_num_nodes"] = ctx.num_fpgas
        tc["default_hw_config"] = hw_key
        for k, v in (ctx.spec.get("runtime_args") or {}).items():
            # Dotted paths let a caller reach any runtime field, e.g.
            # "host_debug.zero_out_dram": true
            node = cfg
            parts = str(k).split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = v

        wl = cfg["workload"]
        wl["workload_name"] = f"{ctx.workload}.json"
        # Never true.  Under ExternallyProvisioned it is a no-op anyway, and
        # under a tagged farm it terminates EC2 instances -- instance lifetime
        # is the operator's business, not a per-job side effect.
        wl["terminate_on_completion"] = False
        # Unique results dir per job. Without this, two runs of the same
        # workload starting in the same UTC second collide.
        wl["suffix_tag"] = ctx.run_tag

        self._dump_yaml(cfg, ctx.runtime_yaml)

        # --- 4. paranoia -----------------------------------------------
        # Re-read what we just wrote and assert the run farm really is this
        # lane's. This is the last line of defence against a job touching
        # hardware that belongs to someone else, so it is checked from disk
        # rather than from the in-memory dict.
        back = self._load_yaml(ctx.runtime_yaml)
        ov = back["run_farm"]["recipe_arg_overrides"]
        if lane.mode == "tagged":
            if ov.get("run_farm_tag") != lane.tag:
                raise RuntimeError(
                    f"rendered run_farm_tag {ov.get('run_farm_tag')!r} != "
                    f"lane tag {lane.tag!r}; refusing to run")
        else:
            got = [list(d.keys())[0] for d in ov.get("run_farm_hosts_to_use", [])]
            if sorted(got) != sorted(lane.hosts):
                raise RuntimeError(
                    f"rendered run farm hosts {got} != lane hosts "
                    f"{list(lane.hosts)}; refusing to run")
            if "run_farm_tag" in ov:
                raise RuntimeError(
                    "externally_provisioned run farm must not carry a "
                    "run_farm_tag; refusing to run")

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    def _firesim(self, ctx: JobContext, *subcmd: str) -> list[str]:
        tree = ctx.tree
        firesim_root = tree / "sims" / "firesim"
        deploy = firesim_root / "deploy"
        q = shlex.quote
        inner = (
            "set -o pipefail; "
            # chipyard's env.sh must be sourced with conda already on PATH,
            # and without a pipe -- a pipe subshells it and PATH never applies.
            f"export PATH={q(self.conda_path)}:$PATH; "
            # Drop inherited conda state so the tree's activate is clean.
            "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER "
            "CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA; "
            f"source {q(str(tree / 'env.sh'))}; "
            f"cd {q(str(firesim_root))}; "
            f"source {q(str(firesim_root / 'sourceme-manager.sh'))} "
            "--skip-ssh-setup; "
            f"cd {q(str(deploy))}; "
            f"firesim -c {q(str(ctx.runtime_yaml))} "
            f"-a {q(str(ctx.workdir / 'config_hwdb.yaml'))} "
            + " ".join(q(s) for s in subcmd)
        )
        return ["bash", "-c", inner]

    def argv(self, ctx: JobContext, phase: Phase) -> Optional[list[str]]:
        if phase is Phase.LAUNCH:
            if ctx.lane.mode == "tagged" and ctx.lane.manage_hosts:
                return self._firesim(ctx, "launchrunfarm")
            return None
        if phase is Phase.INFRASETUP:
            return self._firesim(ctx, "infrasetup")
        if phase is Phase.RUN:
            return self._firesim(ctx, "runworkload")
        if phase is Phase.KILL:
            return self._firesim(ctx, "kill")
        if phase is Phase.TERMINATE:
            if ctx.lane.mode == "tagged" and ctx.lane.manage_hosts:
                return self._firesim(ctx, "terminaterunfarm", "--forceterminate")
            return None
        return None

    def uartlog_argv(self, ctx: JobContext) -> Optional[list[str]]:
        """Print the live UART output of every slot in this lane.

        Read from the run hosts, not the manager: `runworkload` only copies
        results back when a job *completes*, and a bare-metal guest never
        does.  Run-farm hosts need ~/firesim.pem, not the manager's default
        key.
        """
        if not ctx.lane.hosts:
            return None
        key = self.options.get("run_host_key", "~/firesim.pem")
        user = self.options.get("run_host_user", "ubuntu")
        simdir = self.simulation_dir
        parts = []
        for h in ctx.lane.hosts:
            parts.append(
                f"ssh -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "
                f"-i {key} {user}@{shlex.quote(h)} "
                f"'for f in {simdir}/sim_slot_*/uartlog; do "
                f"[ -f \"$f\" ] && tr -d \"\\r\" < \"$f\"; done' || true")
        return ["bash", "-c", "; ".join(parts)]

    def argv_clear_uartlog(self, ctx: JobContext) -> Optional[list[str]]:
        """Delete stale uartlogs on this lane's run hosts BEFORE the job runs.

        ``read_uartlog()`` reads from the run host, and
        ``<simdir>/sim_slot_*/uartlog`` SURVIVES between jobs.  So a job whose
        simulation never starts -- a failed infrasetup, an early abort -- hands
        back the PREVIOUS occupant's uartlog, and COLLECT writes it out as this
        job's result.  That is the cross-agent mis-attribution that repeatedly
        handed one agent another agent's profile, and it is invisible because
        the log looks perfectly well-formed; it is simply the wrong model.

        Clearing first converts a silent wrong answer into an honest empty one.
        """
        if not ctx.lane.hosts:
            return None
        key = self.options.get("run_host_key", "~/firesim.pem")
        user = self.options.get("run_host_user", "ubuntu")
        simdir = self.simulation_dir
        parts = []
        for h in ctx.lane.hosts:
            parts.append(
                f"ssh -n -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 "
                f"-i {key} {user}@{shlex.quote(h)} "
                f"'rm -f {simdir}/sim_slot_*/uartlog' || true")
        return ["bash", "-c", "; ".join(parts)]

    def results_globs(self, ctx: JobContext) -> list[str]:
        """Where `runworkload` puts this job's results on the manager."""
        deploy = ctx.tree / "sims" / "firesim" / "deploy"
        return [str(deploy / "results-workload" /
                    f"*-{ctx.workload}-{ctx.run_tag}")]

    def argv_collect(self, ctx: JobContext, dest: pathlib.Path) -> list[str]:
        pats = " ".join(_quote_glob(g) for g in self.results_globs(ctx))
        d = shlex.quote(str(dest))
        return ["bash", "-c",
                f"mkdir -p {d}; shopt -s nullglob; "
                f"for r in {pats}; do cp -a \"$r\" {d}/ || true; done"]
