# Running Zephyr ELFs on AWS F2 FPGAs

How to get a Zephyr binary executing on F2 (VU47P) FPGAs via FireSim, in AWS account
`025690631703` / `us-east-1`. Written 2026-08-27 from an actual bring-up; every step
below was executed, and the gotchas are ones that actually bit.

Companion docs: `FPGA_GATENAV_REPRODUCE.md` (the RoSE co-sim flow on local U250s),
`ROSE_TACIT_TRACING.md` (TACIT).

---

## 0. Mental model — which tree, and why it matters

The manager instance (`i-03b1e588330725b52`) holds **two** chipyard trees. Picking the
wrong one is the most likely way to waste an hour:

| Tree | chipyard | rose gen | Q31 | Use it for |
|---|---|---|---|---|
| `~/chipyard-fsim` | 1.14.0 | no | no | the older `f2_saturn_refv256d128_*` AGFIs |
| `~/chipyard-rose` | 1.14.0 | **yes** | **yes** | anything built from the RoSE tree |

**Rule: run an AGFI from the tree that built it.** FireSim's `infrasetup` compiles a
simulation driver from the tree you invoke it in, and that driver must match the
bitstream's deploy quintuplet. Cross them and you get a driver/bitstream mismatch —
which does not fail loudly, it fails weirdly.

Garden (`/scratch/dima/rose-infra/RoSE`) is **not** an execution host for any of this.
It is the git source of truth and where you cross-compile the Zephyr ELF.

---

## 1. Reach the manager

```bash
ssh -i ~/.ssh/firesim.pem ubuntu@<manager-public-ip>
```

The public IP changes on every stop/start (no Elastic IP). Recover it with the console,
or from another machine that has AWS credentials. **The only working AWS credentials
live on the manager itself** — so if you stop that instance, you cannot start it again
without console access. Do not stop it casually.

`~/.ssh/firesim.pem` is the sole surviving copy of the `firesim` keypair
(fingerprint `3e:d7:f4:82:71:9a:68:42:9d:72:50:38:50:67:a0:8a:3d:fb:1b:58`).
AWS cannot reissue it. Keep the backup.

---

## 2. Build the Zephyr ELF (on garden)

The board is **`chipyard_riscv64`** — a plain chipyard Rocket target, no RoSE. It is
already in the in-tree Zephyr fork.

```bash
cd /scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
source scripts/activate_conda.sh
source scripts/set_envvars_sdk.sh
west build -p always -b chipyard_riscv64 zephyr_ws/zephyr/samples/hello_world -d /tmp/build_hello
```

Takes about a minute. Output: `/tmp/build_hello/zephyr/zephyr.elf`.

**Sanity-check the ELF before shipping it** — these two properties are what FireSim needs:

```bash
readelf -h /tmp/build_hello/zephyr/zephyr.elf | grep -E 'Entry|Machine'
#   Machine: RISC-V
#   Entry point address: 0x80000000     <-- MUST be the FireSim DRAM base
```

Note the ELF is built `rv64imafdc` — **no V extension**. That is fine: the Saturn vector
unit is present in hardware and simply goes unused. If you want vector code, enable it in
the Zephyr config; the hardware supports it.

Ship it:

```bash
scp -i ~/.ssh/firesim.pem /tmp/build_hello/zephyr/zephyr.elf ubuntu@<manager-ip>:~/zephyr-hello.elf
```

---

## 3. Confirm the AGFI you want is actually usable

An AGFI being `available` is **not** sufficient — it also needs an entry in
`config_hwdb.yaml`, and that merge is easy to miss.

```bash
# what exists, and on which shell
aws ec2 describe-fpga-images --region us-east-1 --owners self \
  --query 'FpgaImages[].{Name:Name,AGFI:FpgaImageGlobalId,Shell:ShellVersion,State:State.Code}' --output text
```

- Shell `0x10212415` = **F2**, runnable.
- Shell `0x04261818` = F1 — **dead**. F1 is no longer offered in us-east-1, so every
  F1-shell AGFI in this account is unrunnable regardless of its `available` state.

Then check the hwdb actually knows about it:

```bash
grep -E '^f2_[a-z0-9_-]+:' ~/chipyard-fsim/sims/firesim/deploy/config_hwdb.yaml
```

If the key is missing but the build produced it, merge from the build artifact:

```bash
D=~/chipyard-fsim/sims/firesim/deploy
cp $D/config_hwdb.yaml $D/config_hwdb.yaml.bak-$(date +%s)
cat $D/built-hwdb-entries/<name> >> $D/config_hwdb.yaml
```

> This exact gap existed for `f2_saturn_refv256d128_singlecore_60mhz-fast`: the AGFI was
> built and `available`, but had never been merged into `config_hwdb.yaml`, so FireSim
> could not run it. Its `built-hwdb-entries` file was intact.

---

## 4. Stage the workload

FireSim bare-metal workloads are a JSON file plus a directory, both under
`deploy/workloads/`.

```bash
D=~/chipyard-fsim/sims/firesim/deploy
mkdir -p $D/workloads/zephyr-hello
cp ~/zephyr-hello.elf $D/workloads/zephyr-hello/zephyr.elf

cat > $D/workloads/zephyr-hello.json <<'JSON'
{
  "benchmark_name": "zephyr-hello",
  "common_bootbinary": "zephyr-hello/zephyr.elf",
  "common_rootfs": null,
  "common_simulation_outputs": ["uartlog"],
  "no_post_run_hook": true
}
JSON
```

`common_rootfs: null` is correct for bare metal — Zephyr is the whole image, there is no
Linux rootfs. `common_bootbinary` is relative to `deploy/workloads/`.

---

## 5. Runtime config

```yaml
# ~/chipyard-fsim/sims/firesim/deploy/config_runtime_f2run.yaml
run_farm:
  base_recipe: run-farm-recipes/aws_ec2.yaml
  recipe_arg_overrides:
    run_farm_tag: rosef2run          # keep DISTINCT from any build fleet tag
    run_instance_market: ondemand
    always_expand_run_farm: true
    run_farm_hosts_to_use:
      - f2.6xlarge: 8
      - f2.12xlarge: 0
      - f2.48xlarge: 0
target_config:
  topology: no_net_config
  no_net_num_nodes: 8                # one sim per FPGA
  default_hw_config: f2_saturn_refv256d128_singlecore_60mhz-fast
  # ... link_latency / net_bandwidth / profile_interval as per the stock template
workload:
  workload_name: zephyr-hello.json
  terminate_on_completion: false
```

Two things that will bite:

- **`run_farm_tag` must be unique per concurrent fleet.** FireSim requests *and releases*
  hosts by tag, so a shared tag lets one job terminate another job's instances. Build
  fleets and run farms must never share one.
- **`workload_name` must reference a file that exists.** The stock config ships pointing
  at `linux-uniform.json`, which is *not* present in this tree — `launchrunfarm` fails on
  `FileNotFoundError` before it launches anything. Valid options here: `bare-base.json`,
  `br-base-uniform.json`, `null.json`, or your own.

---

## 6. Launch, program, run

All three commands must run with the manager env sourced. Note `env.sh` needs conda on
`PATH` *first*, and must be sourced **without a pipe** (a pipe subshells it, `PATH` never
applies, and sbt then runs on the system JDK and dies with "bad constant pool index"):

```bash
export PATH=/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:$PATH
source ~/chipyard-fsim/env.sh
cd ~/chipyard-fsim/sims/firesim
source sourceme-manager.sh --skip-ssh-setup
cd deploy

firesim launchrunfarm  -c config_runtime_f2run.yaml   # boot 8x f2.6xlarge
firesim infrasetup     -c config_runtime_f2run.yaml   # build driver + program FPGAs
firesim runworkload    -c config_runtime_f2run.yaml   # load the ELF and run
```

`infrasetup` is the slow one — it re-elaborates the target config to build a matching
driver before it programs anything. Budget tens of minutes on first run for a config it
has not seen; it is much faster afterwards.

Run these under `nohup`/`tmux`. They outlive an SSH drop and you will lose the run otherwise.

---

## 7. Collect results

```bash
RD=$(ls -td ~/chipyard-fsim/sims/firesim/deploy/results-workload/*zephyr* | head -1)
find $RD -name uartlog | wc -l        # expect one per node
head -20 $RD/*/uartlog
```

Expected Zephyr hello world output:

```
*** Booting Zephyr OS build ... ***
Hello World! chipyard_riscv64
```

### Verified 2026-08-27 — 8/8 FPGAs

Run on `f2_saturn_refv256d128_singlecore_60mhz-fast` (`agfi-057023f4e850e1230`) across
8x f2.6xlarge. Every node printed:

```
*** Booting Zephyr OS build c93d5bdb6078 ***
Hello World! chipyard_riscv64/rocketchip_virt_riscv64
```

**A bare-metal Zephyr sim never terminates.** `hello_world` prints and then idles, so
FireSim reports `8/8 simulations are still running` indefinitely and never copies results
back. Do not wait for `runworkload` to return — read the uartlog live on the run host:

```bash
# from the manager; note the RUN FARM hosts need ~/firesim.pem, not the default key
IPS=$(aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:fsimcluster,Values=rosef2run Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].PrivateIpAddress' --output text)
for ip in $IPS; do
  echo -n "$ip: "
  ssh -i ~/firesim.pem -o StrictHostKeyChecking=no $ip \
    "tr -d '\r' < /home/ubuntu/sim_slot_0/uartlog | grep -a 'Hello World' | head -1"
done
```

Then stop the run: `firesim kill -c config_runtime_f2run.yaml` (or Ctrl-C the manager
process). For a workload that *should* self-terminate, give it an exit mechanism and set
`terminate_on_completion: true`.

---

## 8. Tear down — this is the expensive part

**f2.6xlarge is $1.98/hr each.** Eight of them is **$15.84/hr ≈ $380/day**. They do
*not* stop themselves after a workload; `terminate_on_completion: false` leaves them up
deliberately so you can iterate.

```bash
firesim terminaterunfarm -c config_runtime_f2run.yaml
```

Verify nothing is left behind — belt and braces, since a stale run farm is pure burn:

```bash
aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:fsimcluster,Values=rosef2run Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text
```

---

## 9. Quotas

| Quota | Limit | Notes |
|---|---|---|
| Running On-Demand **F** instances | **384 vCPU** | 16x f2.6xlarge, or 8x f2.12xlarge, or 2x f2.48xlarge — all cap at **16 FPGAs** |
| Running Spot **F** instances | **0** | on-demand only; raising it needs a quota request |
| F Dedicated Hosts | 0 | not a usable path |
| Running On-Demand **Standard** | 496 vCPU | this is what the z1d *build* fleet consumes, not F |

Build fleets and run farms draw on **different** quotas, so they do not contend.

---

## 10. Gotchas, collected

1. **Run an AGFI from the tree that built it** (§0). Driver/bitstream must match.
2. **`available` AGFI ≠ runnable** — it also needs the `config_hwdb.yaml` entry (§3).
3. **F1-shell AGFIs are dead** in us-east-1. 21 of the 23 AGFIs in this account are F1.
4. **Source `env.sh` without a pipe**, with conda already on `PATH` (§6).
5. **Unique `run_farm_tag` per fleet**, or jobs terminate each other (§5).
6. **`workload_name` must exist** — the stock value does not (§5).
7. **ELF entry must be `0x80000000`** to match the FireSim DRAM base (§2).
8. **Manager credentials are irreplaceable in practice** — stopping the manager strands
   it, because the only working AWS credentials are on it (§1).
9. **`[error] Picked up JAVA_TOOL_OPTIONS` is not an error.** sbt tags the JVM's stderr
   line as `[error]`. Likewise `SLF4J: Failed to load class ...` is benign. Do not chase
   either; grep for `Traceback`, `Exception in thread`, `make ... Error N` instead.
10. **`common_bootbinary` is relative to `workloads/<workload_name>/` — do NOT prefix it
    with the workload name.** `bootbinary_path()` is literally
    `workload_input_base_dir + common_bootbinary`. Writing `"zephyr-hello/zephyr.elf"`
    resolves to `workloads/zephyr-hello/zephyr-hello/zephyr.elf` and infrasetup dies with
    `rsync: change_dir ".../zephyr-hello/zephyr-hello" failed`. Correct value: `"zephyr.elf"`.
11. **`runworkload` exit 0 and "Job ... completed!" do NOT mean the workload ran.** If
    infrasetup failed, the driver was never deployed and each node's job script exits
    immediately — you get `RUNWORKLOAD_EXIT=0`, 8 "completed" jobs, and 8 identical
    655-byte uartlogs containing only `sudo: ./FireSim-f2: command not found`.
    **Always gate runworkload on infrasetup, and always verify uartlog CONTENT:**
    ```bash
    firesim infrasetup -c cfg.yaml || { echo "infrasetup failed"; exit 1; }
    firesim runworkload -c cfg.yaml
    ```
12. **Run-farm hosts use `~/firesim.pem`**, not the manager's default SSH key. Plain
    `ssh <private-ip>` gives `Permission denied (publickey)`.
13. **Manager RAM is tight.** chipyard forces `JAVA_TOOL_OPTIONS: -Xmx16G` — a heap
    ceiling *above* the m5.xlarge's 15 GB. A swapfile is load-bearing, not optional:
    ```bash
    sudo fallocate -l 24G /swapfile-rose && sudo chmod 600 /swapfile-rose
    sudo mkswap /swapfile-rose && sudo swapon /swapfile-rose
    ```
    Do not over-allocate — 64 G left only 8.9 GB of disk and would have starved the builds.
