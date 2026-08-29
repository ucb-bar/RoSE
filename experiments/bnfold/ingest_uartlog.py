#!/usr/bin/env python3
"""uartlog -> scheduler-format results.csv (+ dispatch graph) for the BN A/B.

The F2 path runs the ELF through `fq`, not firesim_runner, so nothing writes
the IREE-shape profile tree for us. This does exactly what
runner_common.emit_iree_profile would have.

usage: ingest_uartlog.py <uartlog> <graph.json> <cpu_tag> <backend_tag> <expect_model>
"""
import sys, os, json

XPURT = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"
ZCS = os.path.join(XPURT, "zephyr-chipyard-sw")
sys.path.insert(0, ZCS)

from modelblaster.validation import runner_common          # noqa: E402
from modelblaster.pipeline import profile_writer           # noqa: E402
from modelblaster.pipeline import emit_dispatch_graph      # noqa: E402

uartlog, graph_json, cpu_tag, backend_tag, expect_model = sys.argv[1:6]
text = open(uartlog, errors="replace").read()

# ---- provenance ------------------------------------------------------
banner = [l for l in text.splitlines() if "harness: model=" in l]
got_model = banner[0].split("model=")[1].split()[0].strip() if banner else "?"
if got_model != expect_model:
    raise SystemExit(f"PROVENANCE MISMATCH: uartlog model={got_model!r} "
                     f"expected {expect_model!r}")
verify = runner_common.parse_verify(text)
wall = runner_common.parse_wall_cycles(text)
recs = runner_common.parse_profile(text)
if not recs:
    raise SystemExit("no MODELBLASTER_PROFILE block in uartlog")

ir = json.load(open(graph_json))
ir_ops = {o["dispatch_id"]: o for o in ir["ops"] if o.get("op") != "view"}
ir_ids = set(ir_ops)
prof_ids = {r["dispatch_id"] for r in recs}
missing = sorted(ir_ids - prof_ids)

# Zero-cost ops (chunk*/view) carry a dispatch_id in the IR but emit no
# kernel, so the harness never prints a MODELBLASTER_PROFILE row for them.
# The scheduler's profile loader is strict=True: a dispatch_id present in
# the dispatch graph but absent from results.csv aborts the run. Pad them
# in as explicit 0-cycle rows (what the existing f2 profiles already do).
for did in missing:
    o = ir_ops[did]
    shape = ";".join(f"{k}={v}" for k, v in (o.get("shape") or {}).items())
    recs.append({"dispatch_id": did, "name": o.get("name", ""),
                 "op": o["op"], "shape": shape, "cycles": 0})
recs.sort(key=lambda r: r["dispatch_id"])
padded = missing

meta = profile_writer.ProfileMeta(
    model="yolov8_nano", quant="int8", backend=backend_tag,
    cores=[0], source="firesim", cpu=cpu_tag, clock_mhz=1000.0)
path = profile_writer.write_profile(recs, meta, os.path.join(XPURT, "gen", "profile"))

# dispatch graph under the same additive tag, model dir "yolov8_nano"
graph = emit_dispatch_graph.build_graph(ir)
gp = os.path.join(ZCS, "gen", "vmfb", "yolov8_nano", cpu_tag, backend_tag,
                  "yolov8_nano.int8", "yolov8_nano.int8_dispatch_graph.json")
os.makedirs(os.path.dirname(gp), exist_ok=True)
json.dump(graph, open(gp, "w"), indent=2)

print(json.dumps({
    "uartlog": uartlog, "model_banner": got_model,
    "n_profile_rows": len(recs), "n_ir_dispatches": len(ir_ids),
    "padded_zero_cost_ids": padded,
    "n_rows_written": len(recs),
    "wall_cycles": wall, "verify": verify,
    "results_csv": path, "dispatch_graph": gp,
    "total_ms": round(sum(r["cycles"] for r in recs) / 1e6, 3),
}, indent=2))
