#!/usr/bin/env python3
"""ARM A: uartlog -> scheduler-format results.csv (+ dispatch graph).

The F2 path runs the ELF through `fq`, not firesim_runner, so nothing writes
the IREE-shape profile tree for us. This does what runner_common.emit_iree_profile
would have, under the armA-private cpu tag.

usage: ingest_uartlog.py <uartlog> <model> <quant> <backend_tag> [cpu_tag]
  backend_tag: gemmini_q31 | V256D128_rvv   (the scheduler-side hw name)
"""
import sys, os, json

XPURT = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"
ZCS = os.path.join(XPURT, "zephyr-chipyard-sw")
MB = os.path.join(ZCS, "modelblaster")
sys.path.insert(0, ZCS)

from modelblaster.validation import runner_common          # noqa: E402
from modelblaster.pipeline import profile_writer           # noqa: E402
from modelblaster.pipeline import emit_dispatch_graph      # noqa: E402

uartlog, model, quant, backend_tag = sys.argv[1:5]
cpu_tag = sys.argv[5] if len(sys.argv) > 5 else "firesim_f2_armA"
graph_json = os.path.join(MB, "examples", f"{model}_armA", quant,
                          "generated", "graph.json")

text = open(uartlog, errors="replace").read()

# ---- provenance ------------------------------------------------------
banner = [l for l in text.splitlines() if "harness: model=" in l]
got_model = banner[0].split("model=")[1].split()[0].strip() if banner else "?"
if got_model != model:
    raise SystemExit(f"PROVENANCE MISMATCH: uartlog model={got_model!r} expected {model!r}")
verify = runner_common.parse_verify(text)
wall = runner_common.parse_wall_cycles(text)
recs = runner_common.parse_profile(text)
if not recs:
    raise SystemExit("no MODELBLASTER_PROFILE block in uartlog")

ir = json.load(open(graph_json))
ir_ops = {o["dispatch_id"]: o for o in ir["ops"]
          if o.get("dispatch_id") is not None and o.get("op") != "view"}
ir_ids = set(ir_ops)
prof_ids = {r["dispatch_id"] for r in recs}
missing = sorted(ir_ids - prof_ids)
extra = sorted(prof_ids - ir_ids)

# Zero-cost ops (chunk*/view) carry a dispatch_id in the IR but emit no
# kernel, so the harness never prints a MODELBLASTER_PROFILE row. The
# scheduler's profile loader is strict: pad them as explicit 0-cycle rows.
for did in missing:
    o = ir_ops[did]
    shape = ";".join(f"{k}={v}" for k, v in (o.get("shape") or {}).items())
    recs.append({"dispatch_id": did, "name": o.get("name", ""),
                 "op": o["op"], "shape": shape, "cycles": 0})
recs.sort(key=lambda r: r["dispatch_id"])

meta = profile_writer.ProfileMeta(
    model=model, quant=quant, backend=backend_tag,
    cores=[0], source="firesim", cpu=cpu_tag, clock_mhz=1000.0)
path = profile_writer.write_profile(recs, meta, os.path.join(XPURT, "gen", "profile"))

graph = emit_dispatch_graph.build_graph(ir)
gp = os.path.join(ZCS, "gen", "vmfb", model, cpu_tag, "gemmini_q31",
                  f"{model}.{quant}", f"{model}.{quant}_dispatch_graph.json")
os.makedirs(os.path.dirname(gp), exist_ok=True)
json.dump(graph, open(gp, "w"), indent=2)

print(json.dumps({
    "uartlog": uartlog, "model_banner": got_model, "backend": backend_tag,
    "n_profile_rows": len(recs), "n_ir_dispatches": len(ir_ids),
    "padded_zero_cost_ids": missing, "unexpected_prof_ids": extra,
    "wall_cycles": wall, "verify": verify,
    "results_csv": path, "dispatch_graph": gp,
    "total_ms": round(sum(int(r["cycles"]) for r in recs) / 1e6, 4),
}, indent=2))
