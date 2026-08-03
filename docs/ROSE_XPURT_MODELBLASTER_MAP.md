# Scheduling models to HW: XPU-RT + ModelBlaster — map & roadmap

How multi-model **schedules are generated (XPU-RT)** and **executed on the RISC-V SoC
(ModelBlaster)**, where the current RoSE flight stack fits, and the concrete gap to the
goal: *run a set of models, schedule them to HW, feed live sensors (camera) in, and emit
outputs down to lower-level control policies.*

> Source caveat: the ModelBlaster runtime lives in a git submodule that is **not checked
> out** in this working tree. `modelblaster/` was extracted to
> `github.com/ucb-bar/ModelBlaster` at commit `0ac2c4ee9e9` ("modelblaster: extract to
> ucb-bar/ModelBlaster, instantiate as submodule"); the `merlin/` and QNN runtimes are
> likewise submodules. To read/modify that source: `git submodule update --init
> soc/sw/xpu-rt/ModelBlaster`, or read the pre-extraction tree with
> `git show 0ac2c4ee9e9^:modelblaster/<path>`. File references below to `modelblaster/…`
> are at `0ac2c4ee9e9^`; `samples/executorch/…` references are on branch `origin/dev`.

## TL;DR

- **XPU-RT** (`soc/sw/xpu-rt/`, a.k.a. FreshScheduler) is the **scheduler frontend**:
  it jointly schedules a *set* of models — a merged DAG of per-op "dispatches" — onto
  **heterogeneous typed cores** (`CPU_P`/`CPU_E` → RVV / Gemmini / scalar), mixing
  **periodic** and **best-effort** tasks, via a MILP (cvxpy+MOSEK) or greedy solvers. It
  emits `schedules/scheduled_*.json`.
- **ModelBlaster** is the **execution + profiling backend**: a PyTorch → per-op C-kernel
  codegen pipeline whose `harness_xpurt` **consumes an XPU-RT schedule** and runs N models
  on M cores in **one Zephyr ELF** on spike / FireSim, with explicit hart pinning,
  data/time dependencies, and time-gated starts.
- The two are joined by two JSON contracts (profiles up, schedule down) in a closed
  feedback loop. This half — *"schedule a set of models to HW"* — is **built and validated**.
- The other half — *live sensor → scheduled models → control policy* — currently exists
  **only as a host-side timing study** (`sims/scripts/play_dronet_mlp_scheduled.py`, torch
  on the host). Running it **on the SoC with RoSE sensors** is the open work; the missing
  edges (camera in, policy out) are exactly what the RoSE bridge + threaded flight
  controller already provide, just not wired into ModelBlaster yet.

## 1. System shape

`soc/sw/xpu-rt/` is itself the XPU-RT / FreshScheduler repo. ModelBlaster is a sibling
under it. The data flow:

```
PyTorch models ──ModelBlaster codegen──► per-op C kernels (scalar / rvv / rvv_opu / gemmini_q31)
       │                                            │
       │  profile on spike / FireSim (rdcycle)      │
       ▼                                            ▼
  results.csv  ─────────────► XPU-RT scheduler ──► scheduled_*.json ──► ModelBlaster
  (per-dispatch mean_time,     (MILP or greedy      (per dispatch:        harness_xpurt:
   IREE-schema)                 over a merged        hardware_target,      ONE Zephyr ELF,
  + *_dispatch_graph.json       multi-model DAG)     start_time, deps)     runs it on the SoC
```

The handoff formats (ModelBlaster ↔ XPU-RT):
- **Up (MB → XPU-RT):** IREE-schema `gen/profile/<backend>/<cpu>/<model>/…/results.csv`
  (`dispatch_id, module_name, mean_time, mean_unit`) + per-network IR
  `*_dispatch_graph.json`. Emitters: `modelblaster/pipeline/profile_writer.py`,
  `emit_dispatch_graph.py`.
- **Down (XPU-RT → MB):** `schedules/scheduled_*.json`, consumed by
  `modelblaster/pipeline/ingest_xpurt_schedule.py`.

The canonical glue doc already in-tree: **`soc/sw/xpu-rt/docs/end_to_end_xpurt_firesim.md`**.

## 2. Schedule generation (XPU-RT)

**Model.** `xpu-rt/xpu-rt/workload.py` — an **Operation** is the atomic schedulable unit
(one dispatch), carrying `processing_times` (one duration per machine-combination),
`predecessors` (DAG edges), and optional time-window bounds `min_start_t`/`max_end_t`. A
**Job** is one model instance; a **Workload** is the merged DAG over all networks plus the
`machines`, `machine_combinations`, and a `transfer_times` matrix. Cores are typed
(`CPU_P` performant, `CPU_E` efficient), each mapped to a backend/accelerator tag.

**Two co-scheduled task classes:**
- **Periodic** — `period` + `window_duration` in the input JSON; expanded into N windowed
  instances (`dronet0, dronet1, …`) pinned to `[start+i·period, start+i·period+window)`.
  Instance count comes from a profile-based horizon `H = S_np / (1 − F_p)`
  (`profile_metrics.py:411`, used by `workload_factory.py`).
- **Best-effort / non-periodic** — no window; these drive the makespan objective.

**Flow** (`scripts/run_xpurt_schedule.py:127`, `schedule_iree_networks`):
1. Parse top-level `data/toplevel/networks_*.json` (`hardware`, `scheduler`, `networks`).
2. Build machines + combinations (`workload_factory.build_machine_combinations`).
3. Load profiled per-op times (`profile_loader.load_profiled_processing_times:198`) —
   strict by default (missing profile CSV → error, not random substitution).
4. Build the Workload, expanding periodic nets (`create_workload_from_network_hierarchy`).
5. **Solve** (`--solver`): `milp` (default, `scheduler.py:295`, cvxpy+MOSEK, flow-shop
   formulation) / `greedy` / `greedy_periodic` (`greedy_scheduler.py` + the periodic
   refinement loop `run_xpurt_schedule.py:339`) / `decomposed`. Optional op **fusion**
   (`fusion.py`) merges sub-threshold dispatches first.
6. Post-process + write artifact (`postprocessing.output_scheduled_json:24`);
   `granularity_advisor.py` runs advisory-only.

**Artifact** (`schedules/scheduled_*.json`) — one entry per dispatch:

```json
"dronet0_dispatch_9": {
  "id": 9, "dependencies": ["dronet0_dispatch_7", "dronet0_dispatch_8"],
  "hardware_target": "CPU_E#0", "start_time": 5.506614, "duration": 0.510602,
  "job_name": "dronet0", "time_dependency": "dronet0_dispatch_6"
}
```

`hardware_target` = assigned core/engine (`+`-joined for multi-core combos);
`dependencies` = data-dep DAG edges; `time_dependency` = the previous dispatch on the
*same* hardware_target (runtime serialization). It is **natively multi-model** — the
`networks` map is merged into one solve. Shipped examples: 2-way (`dronet_yolov8`), 3-way,
unrolled smolVLA.

**Inputs.** The scheduler never reads ONNX/`.pte`; it sees only the per-network
`*_dispatch_graph.json` (op DAG) + profiled `results.csv`. Models originate as PyTorch
(ModelBlaster) or ONNX/MLIR (merlin).

## 3. Schedule execution on HW (ModelBlaster `harness_xpurt`)

This is the real "schedule → HW" path, and the **generated-schedule analog of the
hand-written threaded flight controller** (`docs/ROSE_FLIGHT_CONTROLLER_THREADING.md`).

1. **Ingest** (`ingest_xpurt_schedule.py`): schedule JSON → a flat C
   `xpurt_sched_entry_t[]` table sorted by `start_time`; resolves `CPU_P/CPU_E#n →
   (core, kind, hart)` via `cores/*.json`; remaps IR dispatch_ids → codegen indices.
2. **Codegen** (`generate_xpurt_main.py`): emits `<schedule>_main.c` with **one Zephyr
   pthread worker per core-kind, pinned to a hart** (`pthread_attr_setaffinity_np`). Each
   walks the table and acts only on its kind's entries. Per entry: wait on data-dep
   semaphores → wait on cross-job time-dep → **soft start-gate** (busy-yield until
   `run_t0 + start_time·cycles_per_ms`) → dispatch via a per-backend function table
   (`MODEL_<UMID>_DISPATCH_FNS_{GEMMINI,RVV}[dispatch_id]`) → wall-cycle finalize →
   `k_sem_give` dependents. (Semantics: `modelblaster/notes/xpurt_walker_semantics.md`.)
3. **Build/run** (`modelblaster/examples/xpurt_demo/run.sh`): knobs `SCHEDULE_JSON`,
   `MODELS`, `BACKENDS`, `REGISTRY`, `CPU_P_KIND`/`CPU_E_KIND`, `XPURT_TRACE`. Links every
   model×backend object into one ELF; runs `spike -p4` / FireSim; a runner verifies each
   network vs its PyTorch golden.

**Hardware targets:** RISC-V scalar (Rocket harts), **Saturn RVV** (`rvv`, `rvv_opu`
outer-product), **Gemmini** systolic (`gemmini_q31` int8 RoCC). Curated kernels in
`modelblaster/kernels/<target>/`; backends registered in `pipeline/backends.py::BACKENDS`.
**Validated** (`xpurt_walker_semantics.md §9`): a 242-entry dronet(50 ms)+yolov8_nano
schedule ran on dual-rocket-Saturn-Gemmini at **135.21 ms actual vs 141.63 ms predicted**,
~4× over gemmini-only via heterogeneous op routing.

**Timing/trace:** on FireSim the clock is **mtime** (1 µs/tick at the modeled 1 GHz);
per-dispatch start/end log to the uartlog under `XPURT_TRACE=1`
(`MODELBLASTER_XPURT_TRACE`), plotted by `plot_ros_trace.py`.

### ExecuTorch runner — the baseline (not the scheduling path)

`samples/executorch/` (`origin/dev`) runs **`.pte`** files via **ExecuTorch + XNNPACK**
(RISC-V vector + Gemmini micro-kernels). This is where the `MB_*` knobs + the TACIT
tracing (see `ROSE_COSIM_HANG_INVESTIGATION.md`) live: `MB_XNN_PROFILE`,
`MB_TACIT_TRACE_MODEL`, `MB_MULTI_MODEL`, `MB_EXEC_ITERS`, `MB_XNNPACK_SPIN_THREADPOOL`,
`firesim_1core.conf`, … Entry `riscv_executor_runner.cpp:469` → `run_one_pte():258`. It is
a **baseline/comparison** harness (same `rdcycle` bracketing as MB's codegen): **no
scheduler** (`MB_MULTI_MODEL` runs baked models *sequentially* under one boot) and **no
sensor input** (`prepare_input_tensors` fills tensors with 1s). Use it to compare XNNPACK
against MB codegen, not to schedule a workload.

## 4. The sensor→model→policy loop today — host-side only

`sims/scripts/play_dronet_mlp_scheduled.py` prototypes exactly the target datapath, but as
a **digital twin on the host**, not on the SoC:

- **Input:** FPV camera — `env.unwrapped.scene["fpv_camera"].data.output["rgb"]` →
  `preprocess_camera_frame` (RGB→112×112 CHW) → `dronet(frame)`.
- **Chain:** DroNet `(steer, collision)` → command (`target_yaw_rate`, `target_velocity`)
  → the **trained MLP policy** `mlp_policy.act_inference(obs)` → `env.step(actions)`.
- **Schedule = timing gate only:** each model runs *only inside its scheduled dispatch
  window* (`is_model_active(model, time_in_period, dispatches)`), with **zero-order-hold**
  on its last output between runs; a live Gantt overlay shows the schedule vs. wall time.
  The models run as **torch on the host** — the schedule is used purely to study control
  quality under timing, not to execute models on the SoC.

## 5. Current state vs. the goal — three gaps

**Already there:** joint multi-model scheduling (periodic + best-effort) onto
heterogeneous RISC-V cores, executed as one binary on the SoC with correct
timing/dependencies, in a closed profile→schedule→execute→trace loop. That is the *schedule
a set of models to HW* half.

**Gaps to the goal:**

1. **Live sensor input into SoC-executed models.** `harness_xpurt` inputs are
   **baked/synthetic**; there is no sensor datapath in the runtime, and the **RoSE bridge
   is a separate Zephyr sample family — not wired into ModelBlaster** (no MB↔RoSE
   references in `modelblaster/harness*` or `runtime/`). The camera (DMA) + ToF/IMU
   (reqrsp) plumbing built for the flight controller is exactly what's missing.
2. **Model output → lower-level policy on the SoC.** `harness_xpurt` only verifies outputs
   against a PyTorch golden in-binary. There is no output→policy→actuator chain — which is
   precisely what the **threaded flight controller (estimator + TinyMPC + `send_control`)**
   already is.
3. **Two separate multithreaded worlds.** The flight controller (hand-authored
   io/est/ctrl threads, multi-rate via `ROSE_CTRL_DIV`) and `harness_xpurt` (schedule-driven
   pinned pthread workers) are both Zephyr/RISC-V, but unrelated. Converging them — making
   the control task-set a *scheduled* workload — is the crux.

## 6. Roadmap

The missing edges are the pieces the RoSE stack already provides. Incremental route:

1. **RoSE "source"/"sink" dispatches in ModelBlaster.** A source-op kernel that fills a
   model's input tensor from a live RoSE read (camera via DMA, ToF/IMU via reqrsp) instead
   of baked data; a sink-op kernel that routes a model output out via `send_control` or
   into a shared buffer. Smallest first move — gets one live co-sim camera frame into a
   scheduled model.
2. **Flight-control policy as a scheduled dispatch.** Express estimator + TinyMPC as ops in
   the workload (high-rate periodic), with a vision model (DroNet) as a lower-rate periodic
   producer feeding the policy's setpoint — generalizing the `NAV_MODE` setpoint (see
   `rose_nav_controller`) so its target comes from a scheduled model, not a ramp.
3. **Schedule the combined workload.** Profile the blocks, let XPU-RT co-schedule
   vision + estimator + control, run under `harness_xpurt` — replacing the hand-authored
   threading + `ROSE_CTRL_DIV` with a generated schedule.
4. **Close the loop in the RoSE co-sim.** Run `harness_xpurt` as the SoC guest in the RoSE
   lockstep (the exact setup the flight controller runs in now), so Isaac supplies
   camera + physics and the scheduled models + policy fly the drone — the on-SoC
   realization of what `play_dronet_mlp_scheduled.py` only mocks on the host.

## 7. Reference map

Schedule generation (XPU-RT, in-tree):
- Entry: `soc/sw/xpu-rt/scripts/run_xpurt_schedule.py:127`
- MILP solver: `xpu-rt/xpu-rt/scheduler.py:295`; greedy: `greedy_scheduler.py`
- DAG model / builder: `xpu-rt/xpu-rt/workload.py`, `workload_factory.py:305`
- Profile load: `profile_loader.py:198`; horizon: `profile_metrics.py:411`
- Artifact writer: `postprocessing.py:24`
- Example input/artifact: `data/toplevel/networks_periodic_dronet_yolov8_firesim.json`,
  `schedules/scheduled_networks_periodic_dronet_yolov8_firesim_greedy_profiled.json`

Schedule execution (ModelBlaster, submodule `0ac2c4ee9e9^`):
- `modelblaster/pipeline/ingest_xpurt_schedule.py`, `generate_xpurt_main.py`
- `modelblaster/harness_xpurt/`, `examples/xpurt_demo/run.sh`
- `modelblaster/notes/xpurt_walker_semantics.md` (schedule struct, sync, validation)
- Backends/kernels: `pipeline/backends.py`, `kernels/<target>/`
- Single-model harness entry: `modelblaster/harness/src/main.c`

ExecuTorch baseline (`origin/dev`):
- `samples/executorch/executor_runner/riscv_executor_runner.cpp` (main:469, run_one_pte:258)
- `samples/executorch/executor_runner/CMakeLists.txt` (`MB_*` knobs), `model/gen_pte*.py`

Host-side scheduled sim (in-tree):
- `soc/sw/xpu-rt/sims/scripts/play_dronet_mlp_scheduled.py` (loop + gating: :536-595)
- `sims/scripts/utils/schedule_dispatch.py` (schedule loader)

Glue / end-to-end:
- `soc/sw/xpu-rt/docs/end_to_end_xpurt_firesim.md`
- `soc/sw/xpu-rt/README.md` (Flow A ModelBlaster / Flow B merlin)
