#!/usr/bin/env python3
"""
analyze.py — assemble the RoSE co-sim throughput ABLATION result set.

Inputs
  * data/measured_prior.json  — anchors + model params + measured-prior data (provenanced)
  * run_out/seam_{oldsock,newsock}.csv  — measured-NOW spike barrier per-grant decomposition
                                          (written by lib/spike_barrier.sh; optional)

Outputs
  * results/ablation_results.csv  — one row per config (the ablation matrix)
  * results/waterfall.json        — run-level wall-time waterfall + curve data for report.html

Everything derived is computed from the transparent effective-MHz model:
    grants(step)          = tcyc / step
    wall_s(step, pg_ms)   = compute_floor_s + grants(step) * pg_ms/1000
    effective_MHz         = (tcyc/1e6) / wall_s
The model is validated against the two measured-prior FPGA anchors (@5M, @10M) in
data/measured_prior.json (24.6 / 34.3 MHz) — see the printed self-check.
"""
import csv, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
DATA = os.path.join(BUNDLE, "data", "measured_prior.json")
RUN_OUT = os.path.join(BUNDLE, "run_out")
RESULTS = os.path.join(BUNDLE, "results")

D = json.load(open(DATA))
M = D["model"]
TCYC = M["tcyc_counted"]
FLOOR = M["compute_floor_wall_s"]
TICKS = M["warehouse_control_ticks"]


def grants(step):
    return TCYC / step


def wall_s(step, pg_ms, render_ms=0.0, render_ticks=0):
    """Run-level wall for the whole tcyc workload: compute floor + barrier tax + render tax."""
    return FLOOR + grants(step) * (pg_ms / 1000.0) + render_ticks * (render_ms / 1000.0)


def eff_mhz(wall):
    return (TCYC / 1e6) / wall


# ---------------------------------------------------------------------------
# 1) fold in the measured-NOW spike barrier decomposition if present
# ---------------------------------------------------------------------------
def read_seam(tag):
    p = os.path.join(RUN_OUT, f"seam_{tag}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.reader(open(p)))[1:]
    rows = rows[3:]  # skip 3 warmup grants
    if not rows:
        return None
    col = lambda i: statistics.median(float(r[i]) for r in rows if len(r) >= 9)
    return {"total": col(8), "send": col(5), "rtt": col(6), "pickup": col(7), "n": len(rows)}


spike_now = {"oldsock": read_seam("oldsock"), "newsock": read_seam("newsock")}

# spike per-grant: prefer measured-now, else measured-prior constants
PG = D["per_grant_ms"]
if spike_now["oldsock"]:
    spike_old = {"total": spike_now["oldsock"]["total"], "send": spike_now["oldsock"]["send"],
                 "rtt": spike_now["oldsock"]["rtt"], "n": spike_now["oldsock"]["n"], "source": "measured-now"}
else:
    spike_old = {"total": PG["spike_old_socket"]["total"], "send": PG["spike_old_socket"]["send"],
                 "rtt": PG["spike_old_socket"]["rtt"], "n": None, "source": "measured-prior"}
if spike_now["newsock"]:
    spike_new = {"total": spike_now["newsock"]["total"], "send": spike_now["newsock"]["send"],
                 "rtt": spike_now["newsock"]["rtt"], "n": spike_now["newsock"]["n"], "source": "measured-now"}
else:
    spike_new = {"total": PG["spike_new_socket"]["total"], "send": PG["spike_new_socket"]["send"],
                 "rtt": PG["spike_new_socket"]["rtt"], "n": None, "source": "measured-prior"}

PG_FPGA_OLD = PG["fpga_old_socket"]["total"]      # 123 ms measured-prior
PG_FPGA_NEW = PG["fpga_new_socket"]["total"]      # 24 ms computed
ENV = D["env_step_ms"]
R_OLD = ENV["render_every_step"]["mean"]          # 31.0
R_NEW = ENV["render_lazy_10hz"]["mean"]           # 8.1


# ---------------------------------------------------------------------------
# 2) build the ablation matrix
# ---------------------------------------------------------------------------
COLS = ["config_id", "group", "sync_mode", "environment", "guest", "socket_timeout_s",
        "render_hz", "isaac_camera", "firesim_step", "bitstream",
        "per_grant_ms", "effective_MHz", "ticks_per_s", "speedup_vs_baseline",
        "source", "notes"]
rows = []


def row(**kw):
    r = {c: kw.get(c, "") for c in COLS}
    rows.append(r)


# -- Group SILICON: FPGA effective-MHz (sync_mode + granularity + datapath) --
FMHZ = D["effective_mhz_fpga"]
# free-running ceiling (measured-prior)
row(config_id="ceiling_free_running", group="silicon", sync_mode="free", environment="PatternEnv-v0",
    guest="bench:spin", socket_timeout_s="n/a", render_hz="n/a", isaac_camera="no",
    firesim_step="continuous", bitstream="MMIO-Saturn-60MHz",
    per_grant_ms=0.0, effective_MHz=round(FMHZ["free_ceiling"]["mhz"], 1),
    ticks_per_s="", speedup_vs_baseline="",
    source="measured-prior", notes="FMR 1.0 = 60 MHz host clock; no per-grant barrier (minimal_sync continuous grant). THROUGHPUT CEILING.")

# granularity sweep @ old socket (5M/10M measured-prior; rest computed; all FPGA-run-later)
for step in D["levers"]["sync_granularity"]["sweep_cycles"]:
    w = wall_s(step, PG_FPGA_OLD)
    e = eff_mhz(w)
    if step == 5000000:
        src, e, note = "measured-prior", FMHZ["barrier_5M_oldsock"]["mhz"], "recon_sweep.csv @5M (165 grants, 33.7 s). BASELINE granularity."
    elif step == 10000000:
        src, e, note = "measured-prior", FMHZ["barrier_10M_oldsock"]["mhz"], "recon_sweep.csv @10M (82 grants, 24.1 s)."
    else:
        src, note = "computed(FPGA-run-later)", f"model wall=13.8+{grants(step):.0f}x0.123 s; DEFERRED FPGA (busy w/ capstone flight)."
    row(config_id=f"gran_{step//1000000}M_oldsock", group="silicon", sync_mode="barrier",
        environment="PatternEnv-v0", guest="bench:spin", socket_timeout_s=0.1, render_hz="n/a",
        isaac_camera="no", firesim_step=step, bitstream="MMIO-Saturn-60MHz",
        per_grant_ms=PG_FPGA_OLD, effective_MHz=round(e, 1),
        ticks_per_s=round(grants(step) / (wall_s(step, PG_FPGA_OLD)), 3),
        speedup_vs_baseline=round(e / FMHZ["barrier_5M_oldsock"]["mhz"], 2),
        source=src, notes=note)

# socket-fixed FPGA @5M (computed from spike-proven send fix) + DMA vs MMIO datapath
for bs, dma_note in (("MMIO-Saturn-60MHz", ""),
                     ("DMA-Saturn (30MHz built / 60MHz building)",
                      "DMA control-plane == MMIO on the num_bytes=0 barrier (~0 ms recovered); DMA value = camera-frame payload bandwidth per-serve (orthogonal axis). ")):
    w = wall_s(5000000, PG_FPGA_NEW)
    e = eff_mhz(w)
    row(config_id=f"gran_5M_newsock_{'dma' if 'DMA' in bs else 'mmio'}", group="silicon",
        sync_mode="barrier", environment="PatternEnv-v0", guest="bench:spin",
        socket_timeout_s=0.001, render_hz="n/a", isaac_camera="no", firesim_step=5000000,
        bitstream=bs, per_grant_ms=PG_FPGA_NEW, effective_MHz=round(e, 1),
        ticks_per_s=round(grants(5000000) / w, 3),
        speedup_vs_baseline=round(e / FMHZ["barrier_5M_oldsock"]["mhz"], 2),
        source="computed(FPGA-run-later)",
        notes=dma_note + "socket fix send 105->6 ms (spike-proven) + 18 ms FPGA freeze floor => ~24 ms/grant. DEFERRED FPGA.")

# -- Group SPIKE: measured-NOW / -prior barrier decomposition (socket-fix axis) --
for tag, sp, to in (("spike_barrier_oldsock", spike_old, 0.1),
                    ("spike_barrier_newsock", spike_new, 0.001)):
    row(config_id=tag, group="spike", sync_mode="barrier", environment="PatternEnv-v0",
        guest="bench:reqrsp", socket_timeout_s=to, render_hz="n/a", isaac_camera="no",
        firesim_step=1000, bitstream="rose_spike_sim (functional bridge)",
        per_grant_ms=round(sp["total"], 2), effective_MHz="",
        ticks_per_s=round(1000.0 / sp["total"], 2),
        speedup_vs_baseline=round(spike_old["total"] / sp["total"], 2),
        source=sp["source"],
        notes=f"send={sp['send']:.1f} rtt={sp['rtt']:.1f} ms"
              + (f" (n={sp['n']} grants, 3 warmup skipped)" if sp['n'] else "")
              + ". Barrier round-trip = the socket-fix lever, isolated on the functional Spike bridge (no FPGA).")

# -- Group ISAAC: env.step render axis (measured-prior; GPU busy w/ capstone flight) --
for cid, e, cam, hz, src_key, extra in (
    ("isaac_render_every_step", R_OLD, "yes", 0, "render_every_step", "OLD: RTX render EVERY 200 Hz step (chase 960x540)."),
    ("isaac_render_lazy_10hz", R_NEW, "no", 10, "render_lazy_10hz", "NEW: render_interval=20 aligned to cam_front 10 Hz, chase off. Guest-equivalent frames."),
    ("isaac_physics_floor", ENV["physics_floor"]["mean"], "no", "n/a", "physics_floor", "reference: all cameras removed (physics only).")):
    row(config_id=cid, group="isaac", sync_mode="barrier", environment="WarehouseThrustEnv-v0",
        guest="fused-nav (RVV)", socket_timeout_s="", render_hz=hz, isaac_camera=cam,
        firesim_step=5000000, bitstream="", per_grant_ms="", effective_MHz="",
        ticks_per_s=round(1000.0 / e, 1),
        speedup_vs_baseline=round(R_OLD / e, 2),
        source=ENV[src_key]["source"],
        notes=f"env.step {e:.1f} ms/step ({round(1000.0/e,1)} Hz). " + extra)

# -- Group COSIM: run-level effective-MHz on the LIVE warehouse (folds render in) --
WH = D["isaac_effective_mhz"]["warehouse_oldrender_5M_oldsock"]["mhz"]
cosim_specs = [
    ("cosim_warehouse_all_old", PG_FPGA_OLD, R_OLD, "measured-prior",
     f"BASELINE live co-sim: barrier {PG_FPGA_OLD:.0f} ms + render {R_OLD:.0f} ms. Anchored to measured 16.2 MHz."),
    ("cosim_warehouse_socketfix", PG_FPGA_NEW, R_OLD, "computed",
     "+socket fix only (barrier 123->24 ms)."),
    ("cosim_warehouse_lazyrender", PG_FPGA_OLD, R_NEW, "computed",
     "+lazy render only (env.step 31->8 ms)."),
    ("cosim_warehouse_all_new", PG_FPGA_NEW, R_NEW, "computed",
     "ALL host/env-side levers: +socket fix +lazy render (+DMA ~0 on barrier)."),
]
# calibrate: baseline model must reproduce the 16.2 MHz anchor
base_wall = wall_s(5000000, PG_FPGA_OLD, R_OLD, TICKS)
for cid, pg, rm, src, note in cosim_specs:
    w = wall_s(5000000, pg, rm, TICKS)
    e = eff_mhz(w)
    if cid == "cosim_warehouse_all_old":
        e = WH  # pin to measured anchor
    row(config_id=cid, group="cosim", sync_mode="barrier", environment="WarehouseThrustEnv-v0",
        guest="fused-nav (RVV)", socket_timeout_s=(0.001 if pg == PG_FPGA_NEW else 0.1),
        render_hz=(10 if rm == R_NEW else 0), isaac_camera=("no" if rm == R_NEW else "yes"),
        firesim_step=5000000, bitstream="MMIO-Saturn-60MHz",
        per_grant_ms=pg, effective_MHz=round(e, 1), ticks_per_s="",
        speedup_vs_baseline=round(e / WH, 2),
        source=src + ("(FPGA-run-later)" if src == "computed" else ""),
        notes=note + f" run-level wall={w:.1f} s (floor 13.8 + barrier {grants(5000000)*pg/1000:.1f} + render {TICKS*rm/1000:.1f}).")

# ---------------------------------------------------------------------------
# 3) write results CSV
# ---------------------------------------------------------------------------
os.makedirs(RESULTS, exist_ok=True)
outcsv = os.path.join(RESULTS, "ablation_results.csv")
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------------------
# 4) waterfall + curve data for the report
# ---------------------------------------------------------------------------
def barrier_tax(step, pg):
    return grants(step) * pg / 1000.0

def render_tax(rm):
    return TICKS * rm / 1000.0

waterfall = {
    "workload": f"{TCYC/1e6:.0f} M target cycles / {TICKS} control ticks @ 5M step",
    "unit": "run-level wall seconds (lower = faster)",
    "steps": [
        {"label": "Baseline (all-old)", "compute": FLOOR,
         "barrier": barrier_tax(5e6, PG_FPGA_OLD), "render": render_tax(R_OLD),
         "delta": None, "eff_mhz": WH},
        {"label": "+ socket fix", "compute": FLOOR,
         "barrier": barrier_tax(5e6, PG_FPGA_NEW), "render": render_tax(R_OLD),
         "delta": barrier_tax(5e6, PG_FPGA_OLD) - barrier_tax(5e6, PG_FPGA_NEW),
         "eff_mhz": round(eff_mhz(wall_s(5e6, PG_FPGA_NEW, R_OLD, TICKS)), 1)},
        {"label": "+ lazy render", "compute": FLOOR,
         "barrier": barrier_tax(5e6, PG_FPGA_NEW), "render": render_tax(R_NEW),
         "delta": render_tax(R_OLD) - render_tax(R_NEW),
         "eff_mhz": round(eff_mhz(wall_s(5e6, PG_FPGA_NEW, R_NEW, TICKS)), 1)},
        {"label": "+ DMA datapath", "compute": FLOOR,
         "barrier": barrier_tax(5e6, PG_FPGA_NEW), "render": render_tax(R_NEW),
         "delta": 0.0,
         "eff_mhz": round(eff_mhz(wall_s(5e6, PG_FPGA_NEW, R_NEW, TICKS)), 1),
         "note": "DMA touches the num_bytes=0 barrier by ~0 ms; it recovers camera-frame PAYLOAD bandwidth (a different axis)."},
        {"label": "Ceiling (free-run)", "compute": FLOOR, "barrier": 0.0, "render": 0.0,
         "delta": barrier_tax(5e6, PG_FPGA_NEW) + render_tax(R_NEW),
         "eff_mhz": FMHZ["free_ceiling"]["mhz"]},
    ],
    "granularity_curve": [
        {"step_M": s // 1000000,
         "eff_mhz_oldsock": round(FMHZ["barrier_5M_oldsock"]["mhz"] if s == 5000000
                                  else FMHZ["barrier_10M_oldsock"]["mhz"] if s == 10000000
                                  else eff_mhz(wall_s(s, PG_FPGA_OLD)), 1),
         "eff_mhz_newsock": round(eff_mhz(wall_s(s, PG_FPGA_NEW)), 1),
         "grants": round(grants(s), 1),
         "measured": s in (5000000, 10000000)}
        for s in D["levers"]["sync_granularity"]["sweep_cycles"]
    ],
    "ceiling_mhz": FMHZ["free_ceiling"]["mhz"],
    "spike_barrier": {"old": spike_old, "new": spike_new},
    "env_step": {"old": R_OLD, "new": R_NEW, "floor": ENV["physics_floor"]["mean"]},
}
json.dump(waterfall, open(os.path.join(RESULTS, "waterfall.json"), "w"), indent=2)

# ---------------------------------------------------------------------------
# 5) self-check + console summary
# ---------------------------------------------------------------------------
chk5 = eff_mhz(wall_s(5e6, PG_FPGA_OLD))
chk10 = eff_mhz(wall_s(10e6, PG_FPGA_OLD))
print("=" * 72)
print("MODEL SELF-CHECK (computed vs measured-prior FPGA anchors):")
print(f"  @5M  old socket: model {chk5:.1f} MHz  vs measured 24.6 MHz")
print(f"  @10M old socket: model {chk10:.1f} MHz  vs measured 34.3 MHz")
print(f"  baseline warehouse model wall {base_wall:.1f}s -> {eff_mhz(base_wall):.1f} MHz vs anchor 16.2 MHz")
print("=" * 72)
print(f"spike barrier  OLD socket: {spike_old['total']:.2f} ms/grant  [{spike_old['source']}]")
print(f"spike barrier  NEW socket: {spike_new['total']:.2f} ms/grant  [{spike_new['source']}]"
      f"   -> {spike_old['total']/spike_new['total']:.1f}x")
print(f"wrote {outcsv}  ({len(rows)} configs)")
print(f"wrote {os.path.join(RESULTS,'waterfall.json')}")
