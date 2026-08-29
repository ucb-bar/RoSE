# Running jobs on the AWS F2 FireSim queue

How to get an ELF onto an FPGA and the results back again.

This is the **usage** guide. For why the queue is built the way it is — the lane
model, isolation, scheduling policy, locking — see
[`FPGA_QUEUE_DESIGN.md`](FPGA_QUEUE_DESIGN.md).

---

## 1. Prerequisites

| what | value |
|---|---|
| manager host | `ubuntu@3.88.218.39` (m5.xlarge, AWS account 025690631703, us-east-1) |
| SSH key | `~/.ssh/firesim.pem` — **the only copy**; on the manager it lives at `~/firesim.pem` |
| `fq` CLI | `~/fpga_queue/bin/fq` on the manager |
| daemon socket | `FQ_SOCKET=/var/lib/fq/fq.sock` — export this or every command fails |
| chipyard tree | `/home/ubuntu/chipyard-rose` — must be the tree that **built** the bitstream |

`fq` runs **on the manager**, not locally. From a workstation you drive it over
ssh. The FPGAs themselves are separate `f2.6xlarge` run hosts that the manager
reaches over its own private network; you never log into them directly for
normal work.

Check you can reach the daemon:

```bash
ssh -i ~/.ssh/firesim.pem ubuntu@3.88.218.39 \
  'cd ~/fpga_queue && FQ_SOCKET=/var/lib/fq/fq.sock ./bin/fq ping'
```

---

## 2. Quick start

Build an ELF locally, copy it up, submit, wait, read the uartlog:

```bash
MGR=ubuntu@3.88.218.39
KEY=~/.ssh/firesim.pem
SSH="ssh -o BatchMode=yes -i $KEY"

# 1. build (STOP_AFTER=build so nothing tries to run a simulator locally)
cd $MB
MODEL_NAME=dronet BACKEND=reference TARGET=gemmini_q31 QUANT=int8 \
  OPTIMIZE=0 STOP_AFTER=build RUNNER=firesim \
  bash examples/dronet/run.sh

# 2. ship it
ELF=examples/dronet/int8/build/gemmini_q31_firesim/zephyr/zephyr.elf
scp -i $KEY "$ELF" $MGR:/home/ubuntu/myjob.elf

# 3. submit
$SSH $MGR 'cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock && \
  ./bin/fq submit \
     --tree /home/ubuntu/chipyard-rose \
     --hw-config f2_dual_small_norose_tacit_q31_60mhz \
     --elf /home/ubuntu/myjob.elf \
     --timeout 3000 \
     --results /home/ubuntu/myjob_results'

# 4. wait, then read the uartlog
$SSH $MGR 'cd ~/fpga_queue && FQ_SOCKET=/var/lib/fq/fq.sock ./bin/fq wait <JOB_ID>'
$SSH $MGR 'find /home/ubuntu/myjob_results -name uartlog | head -1 | xargs cat'
```

`experiments/profile_matrix.sh` in this repo is a working, battle-tested
reference for the whole build → submit → collect loop across several models.

---

## 3. The interface

```
fq {submit,status,list,lanes,cancel,signal-done,logs,wait,drain,reclaim,ping,daemon,check}
```

| command | use |
|---|---|
| `submit` | enqueue a job (see flags below) |
| `status` / `list` | queue view, or one job |
| `lanes` | show the FPGA pool and what each lane is doing |
| `wait <id>` | block until a job finishes |
| `logs <id>` | that job's log |
| `cancel <id>` | cancel queued or running job(s) |
| `signal-done <id>` | tell a running job its work is finished |
| `ping` | is the daemon alive |
| `drain` / `reclaim` | **admin**: stop scheduling onto a lane / kill an orphaned sim holding one |
| `check` | validate a pool config and exit |

### `fq submit` flags

| flag | meaning |
|---|---|
| `--tree PATH` | **required.** The chipyard tree that built the bitstream — the driver must match the bitstream. |
| `--hw-config KEY` | hwdb key in that tree's `config_hwdb.yaml` (see §4) |
| `--agfi ID` | explicit AGFI instead; cross-checked against hwdb if `--hw-config` also given |
| `--elf PATH` | guest ELF to boot (path **on the manager**) |
| `--rootfs PATH` | rootfs image; omit for bare metal |
| `--results DIR` | directory to copy artifacts into — **see §5, this is narrower than it sounds** |
| `-t, --timeout N` | wall-clock backstop in seconds (0 = pool default of 3600) |
| `--completion {exit,sentinel,timeout}` | how the job ends; default `exit` |
| `--sentinel RE` | uartlog regex meaning success |
| `--fail-regex RE` | uartlog regex meaning failure |
| `-p, --priority N` | 0 low / 5 normal / 10 high |
| `-n, --num-fpgas N` | FPGAs required |
| `--lane NAME` | pin to a specific lane |
| `--set A.B=C` | patch a `config_runtime` field (repeatable) |
| `-w, --wait` | block until finished; `-q` suppresses log streaming |
| `--project`, `--comment` | bookkeeping |

---

## 4. Available bitstreams

From `/home/ubuntu/chipyard-rose/sims/firesim/deploy/config_hwdb.yaml`. Only the
AGFI-backed entries run on F2 — the rest are U250/Vitis entries pointing at local
paths on `garden` and are unusable from the manager.

| hwdb key | AGFI | what it is |
|---|---|---|
| `f2_dual_small_norose_tacit_q31_60mhz` | `agfi-024f2d6a1c6890cb4` | **the workhorse.** 2× Rocket, Saturn V256/D128, Q0.31 Gemmini 16×16 on hart 0, TACIT encoder |
| `f2_dual_small_norose_tacit_q31_20mhz` | `agfi-0e4a0267313845a9d` | same design at 20 MHz |
| `f2_dual_large_norose_tacit_q31_60mhz` | `agfi-0be54e2cd085c8e5d` | Saturn V512/D256, Gemmini 32×32 accumulator |

Notes worth knowing:

* **`norose`** means no RoSE bridge — these are compute-only bitstreams, not the
  Isaac co-simulation ones.
* **60 MHz vs 20 MHz is host emulation speed only.** The simulated design's
  cycle counts are *identical*; 60 MHz just finishes ~2.99× sooner in wall-clock.
  Always quote cycles, not seconds.
* **`SatGem` = Saturn + Gemmini.** The `q31` in the name is the Gemmini variant
  (`Q31WsGemminiConfig`), whose accumulator scale is Q0.31 fixed point.
  Gemmini attaches to **hart 0 only**; hart 1 is Saturn-RVV-only. That asymmetry
  is what the XPU-RT core registry models as CPU_P / CPU_E.
* **`tacit`** means the TACIT instruction-trace encoder is present. It is, but
  the trace *stream* is currently corrupt on F2 — see
  `experiments/tacit/F2_TACIT_VERDICT.md`.
* Quad-core configs do not fit the device; all eight quad builds failed placement.

---

## 5. Getting data back — read this before relying on `--results`

**`--results DIR` copies back the uartlog and essentially nothing else.** The
FireSim backend hardcodes:

```python
"common_simulation_outputs": ["uartlog"]
```

so any other artifact a run produces — TACIT traces, extra dumps, anything the
guest writes — **stays on the run host and is silently not collected.** If you
need it, scp it off the run host yourself. `fq lanes` gives you the host IP:

```bash
# on the manager
scp -i ~/firesim.pem ubuntu@192.168.0.91:/home/ubuntu/sim_slot_0/tacit0.out .
```

`experiments/tacit/f2_tacit_run.sh` is a worked example of that pattern.

Results land on the manager under
`results-workload/<UTC-ts>-<workload>-<suffix_tag>/`, where `suffix_tag` is the
job's `run_tag`. That tag exists specifically so two runs of the same workload
starting in the same UTC second cannot collide.

### Always verify provenance

Job IDs are reused across daemon restarts, and a stale uartlog looks exactly like
a fresh one. Several wrong conclusions in this project came from analysing the
wrong file. Check **both**:

```bash
grep -o 'myjob\.elf'            uartlog   # the ELF name FireSim embeds in the guest cmdline
grep -o 'harness: model=[a-z0-9_]*' uartlog   # the model banner the harness prints
```

If either disagrees with what you submitted, discard the result.

---

## 6. Completion semantics

`--completion` picks how the job is judged finished:

* **`exit`** (default) — the guest terminates the simulation itself. A Zephyr
  guest calling `sys_reboot` does this. Preferred.
* **`sentinel`** — a `--sentinel` regex appears in the uartlog.
* **`timeout`** — only the wall clock ends it.

Under the hood, `runworkload` writing a file into the results directory's
`.monitoring-dir/` is the cleanest completion signal and is what the runner
watches. For a bare-metal guest that never exits, the sentinel regex and the
timeout are the real terminators.

**`--fail-regex` matters more than it looks.** A guest that crashes and hangs
will otherwise hold its lane for the entire timeout. One early job burned ~100
minutes of a lane because it was submitted with no fail regex. The pool now
carries a default:

```
(mcause:|ZEPHYR FATAL ERROR|Faulting instruction address)
```

but if you expect a *different* failure signature, pass your own — and pass a
short `--timeout` for exploratory runs that may fault.

---

## 7. Pool configuration and defaults

```yaml
state_dir:   /var/lib/fq
socket_path: /var/lib/fq/fq.sock
backend:     firesim
backend_options:
  fpgas_per_host: 1
  run_host_key:   ~/firesim.pem
  infrasetup_timeout_s: 7200
defaults:
  priority:      5
  timeout_s:     3600
  max_timeout_s: 86400
  fail_regex:    (mcause:|ZEPHYR FATAL ERROR|Faulting instruction address)
scheduling:
  reserve_after_s:     900
  max_queued_per_user: 200
  poll_interval_s:     5
probe:
  enabled: true
  interval_s: 60
```

The pool is currently **3 lanes** — `f2-00`, `f2-01`, `f2-02` — one
`f2.6xlarge` host each, at `192.168.0.8`, `.31`, `.91`.

`mode: hosts` (FireSim's `ExternallyProvisioned` run farm) is deliberate and
load-bearing: it binds hosts by explicit IP rather than by EC2 tag, so instances
sharing the `fsimcluster=rosef2run` tag become independent lanes, and its
launch/terminate paths are no-ops — **nothing `fq` does can destroy an instance
or run up a bill.**

One host per lane is required, not stylistic: `firesim kill` runs a host-wide
`pkill FireSim-f2`, and `infrasetup` reflashes every slot on the host. Two jobs
sharing a host would destroy each other.

---

## 8. Gotchas

Hard-won, each of these has cost real time:

1. **`ssh -n` starves heredocs.** If you pipe a heredoc into ssh, do not also
   pass `-n` — the heredoc gets eaten. Prefer staging a script to a file and
   `scp`-ing it. This has broken things at least four separate times, including
   an ssh inside a collection loop consuming the rest of the job list off the
   loop's stdin.
2. **Only the uartlog comes back.** See §5.
3. **Verify the ELF name and model banner** in every uartlog before drawing a
   conclusion. See §5.
4. **`--tree` must be the tree that built the bitstream.** The driver and the
   bitstream are a matched pair; mismatches fail in confusing ways.
5. **A crashing job holds its lane for the full timeout** unless a fail regex
   matches. Use a short `--timeout` when you expect a fault.
6. **Cycles, not seconds.** The 20 MHz and 60 MHz bitstreams are cycle-identical.
7. **Only 3 lanes exist.** Stagger submissions; a same-second dispatch onto lanes
   still tearing down has raced before. `profile_matrix.sh` sleeps 12 s between
   submits for exactly this reason.
8. **The daemon runs as root and setuids to the submitter.** Its key is
   `/root/firesim.pem`, not `~/firesim.pem` — a mismatch there makes every host
   probe fail while the queue still looks healthy.

---

## 9. Troubleshooting

| symptom | likely cause |
|---|---|
| every command errors immediately | `FQ_SOCKET` not exported |
| job queues forever | all 3 lanes busy — `fq lanes`; or a lane is drained |
| job "runs" far longer than expected | guest crashed and no fail regex matched (§6) |
| results dir empty | job did not reach a terminal state, or you wanted a non-uartlog artifact (§5) |
| numbers look impossible | check provenance (§5) — you are probably reading a stale uartlog |
| host probes all fail | daemon key path (§8, item 8) |
| a lane is stuck BUSY with no job | orphaned simulation — `fq reclaim` (admin) |
