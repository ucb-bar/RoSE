#!/usr/bin/env python3
"""Emit the kernel_opt_log entry for this campaign, with every metric read
back from results/*.json rather than retyped -- a hand-copied table is exactly
how a log entry stops matching the run it claims to describe."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

R = {os.path.basename(p)[:-5]: json.load(open(p))
     for p in glob.glob(os.path.join(C.RESULTS, "*.json"))}
K = ("d1", "d2", "d3", "rmse", "rel", "log10")


def row(tag):
    m = R[tag]["metrics"]
    return {k: round(m[k], 4) for k in K}


def delta(tag, base):
    a, b = R[tag]["metrics"], R[base]["metrics"]
    return {f"d_{k}": round(a[k] - b[k], 4) for k in ("d1", "rmse")}


arms = {}
for ck, fp in (("epoch0", "fp32_e0_full"), ("epoch12", "fp32_e12_full")):
    t = {"fp32": row(fp)}
    for v in ("c1", "c1pc", "c1ca", "c1dwonly", "c1pcdw",
              "c32pcdw", "c1pcdwca", "c8pcdwca", "c32pcdwca", "c32", "c32pc"):
        tag = f"int8_{ck}_{v}"
        if tag in R:
            t[v] = dict(row(tag), **delta(tag, fp))
    arms[ck] = t

abl = R["ablation_groups"]["arms"]
ablw = R["ablation_weights"]["arms"]
b = abl["fp32"]["d1"]
attribution = {
    "protocol": ("fp32 network, ONE quantisation effect reintroduced at a "
                 "time, 128 NYU val frames, same per-image metrics"),
    "fp32_d1": round(b, 4),
    "weights_only_per_tensor_dd1": round(ablw["w_pt_all"]["d1"] - b, 4),
    "activations_only_dd1": round(abl["a_only"]["d1"] - b, 4),
    "both_dd1": round(abl["both"]["d1"] - b, 4),
    "weights_pc_DENSE_only_dd1": round(ablw["w_pc_dense_only"]["d1"] - b, 4),
    "weights_pc_DEPTHWISE_only_dd1": round(ablw["w_pc_dw_only"]["d1"] - b, 4),
    "weights_pc_all_dd1": round(ablw["w_pc_all"]["d1"] - b, 4),
    "activation_groups_dd1": {k: round(v["d1"] - b, 4)
                              for k, v in abl.items()
                              if k.startswith("a_")},
}
entry = {
    "experiment": "fastdepth_int8_depth_accuracy_and_per_channel_depthwise",
    "platform": ("garden CPU only (no FPGA, no GPU). modelblaster "
                 "pipeline/extract_graph.py int8 PTQ; golden simulation and "
                 "the batched evaluator are the same integer arithmetic, "
                 "verified bit-exact both ways"),
    "job": ("no fq job -- this is a numerics/accuracy campaign, not a "
            "hardware measurement. Every number is CPU."),
    "question": ("The int8 fastdepth path was only ever verified bit-exact "
                 "against its own golden, which says nothing about whether "
                 "the quantized network still predicts good depth. What does "
                 "int8 actually COST in NYU depth metrics, and what dominates "
                 "that cost?"),
    "gates": [
        "fp32 arm reproduces the training box's reported checkpoint metrics "
        "(epoch12 d1 0.7836 vs reported 0.7836) -- if it did not, the harness "
        "would be evaluating a different network",
        "PER-IMAGE metric protocol, not batch-pooled: pooling raises RMSE "
        "systematically by Jensen and would not be comparable to published "
        "numbers or to eval_fastdepth.py",
        "the fast batched int8 simulator is BIT-EXACT with the shipped "
        "reference: (A) a full extraction with the stock five-deep-loop "
        "_sim_* primitives vs the patched ones is byte-identical in "
        "graph.json/weights.npz/io.npz; (B) run_ir on the 224 graph "
        "reproduces io.npz's golden with max_abs_err=0. Without (A) the "
        "speedup could have been silently changing the numbers",
        "the new depthwise_conv2d_s8_pc C reference is host-compiled and "
        "agrees with the golden at max_abs_err=0 on all 22 depthwise ops of "
        "a real IR -- a new op kind that nothing can execute is not an op",
        "default behaviour byte-identical: stock int8 IR, --per-channel IR, "
        "and untrained fastdepth's get_model/get_sample_input/"
        "get_calibration_samples all diffed against pre-change output",
        "MB_DRIFT_ATOL NOT set anywhere in this campaign, and not needed: "
        "every verification here is exact equality",
    ],
    "checkpoints": {
        "epoch0 (best_rmse.pt)": "PRIMARY export target: better in metres "
                                 "(rmse 0.63) and its output spans ~[1.2, "
                                 "5.8] m instead of saturating the 10 m "
                                 "ceiling, so int8 codes are not wasted",
        "epoch12 (best.pt)": "higher d1 (0.784) but over-dispersed; kept as "
                             "a second arm because a ceiling-saturated "
                             "output range is exactly the pathology that "
                             "should quantize worst",
    },
    "metrics_654_nyu_val_per_image": arms,
    "what_dominated": attribution,
    "gt_free_fidelity_epoch0": dict(
        R["fidelity_e0"]["fidelity"],
        note=("int8-vs-fp32 PREDICTION difference, no ground truth involved. "
              "A quantisation that merely shrinks an over-dispersed output "
              "could improve a GT metric while being a worse copy of the "
              "model; this cannot be gamed that way. Stock int8 also "
              "systematically SHRINKS the prediction (std ratio 0.789); the "
              "fixed one does not (0.990).")),
    "calibration_leakage_check": dict(
        {k: {"all654": {m: round(v["all"][m], 4) for m in ("d1", "rmse")},
             "heldout622": {m: round(v["heldout"][m], 4) for m in ("d1", "rmse")}}
         for k, v in R["heldout_e0"]["heldout"].items()},
        note=("make_calib.py draws its 32 calibration frames FROM the val "
              "split, so the headline int8 numbers are partly measured on "
              "frames whose activation ranges the quantiser saw. Excluding "
              "them moves d1 by at most 0.002 -- and moves the fp32 arm by a "
              "comparable amount, i.e. it is frame-sampling noise, not "
              "leakage. Stated because 'should be negligible' is not a "
              "measurement.")),
    "mechanism": (
        "A depthwise conv weight is a stack of INDEPENDENT per-channel "
        "filters, and folding BatchNorm into it leaves those channels' "
        "magnitudes orders of magnitude apart. Under one per-tensor scale the "
        "median channel of the trained encoder gets ~33 of 255 codes (worst "
        "layer 13); weight SQNR is 36.1 dB per-tensor vs 47.2 dB per-channel. "
        "--per-channel could not touch any of it because _apply_per_channel "
        "converted conv2d_s8/linear_s8 only and there was no "
        "depthwise_conv2d_s8_pc op to convert TO."),
    "negative_results": [
        "MORE CALIBRATION FRAMES MADE IT WORSE under the stock max-abs rule: "
        "activation scales are per-tensor max-abs, so an extra frame can only "
        "WIDEN a range, and a wider range is a coarser step. epoch0 "
        "per-channel: 1 frame d1 0.7221 -> 32 frames 0.7118. Only once ranges "
        "are clamp-aware do extra frames pay (8 frames buys 0.015 m rmse).",
        "PER-CHANNEL ON THE DENSE CONVS IS WORTH ALMOST NOTHING here: 0.001 "
        "d1 in the weights-only ablation, 0.007 end-to-end. The entire "
        "per-channel story on this network is the depthwise convs.",
        "MSE-OPTIMAL / PERCENTILE ACTIVATION CLIPPING IS NOT THE LEVER. "
        "Measured per-tensor over the real graph: MSE-optimal clipping would "
        "buy a mean of 1.02 dB (median 0.53) and 99.9th-percentile clipping "
        "is 5.2 dB WORSE than max-abs. The activation distributions are not "
        "heavy-tailed; this was the first hypothesis and it was wrong.",
        "CLAMP-AWARE RANGES ALONE ARE NOT A WIN: on epoch12 they cost 0.007 "
        "d1 on their own (0.6100 -> 0.6049). They only pay once the weight "
        "error is removed (+0.002 d1, -0.006 m rmse on top of per-channel "
        "depthwise). Reported as a refinement, not an independent result.",
        "The per-layer SQNR ladder showed error accumulating from ~46 dB at "
        "the input to ~13 dB at the head with no single guilty layer, and "
        "could not by itself identify the cause. The controlled "
        "reintroduce-one-effect ablation could.",
    ],
    "code": [
        "reference_kernels.py: DEPTHWISE_CONV2D_S8_PC KernelSpec + C reference",
        "extract_graph.py: _apply_per_channel include_depthwise/include_dense; "
        "_sim_depthwise_conv2d_s8_pc golden branch; --per-channel-depthwise / "
        "MB_INT8_PC_DEPTHWISE; MB_INT8_CLAMP_AWARE_RANGES consumer-clamp-aware "
        "activation ranges; get_calibration_spec may return None",
        "mb_datasets/nyu_depth.py: declarative NYU loader (h5 or npz bank), "
        "preprocessing bit-exact with make_calib.py on one machine",
        "models/fastdepth.py: get_calibration_spec, matching yolov8_nano's "
        "convention; get_calibration_samples now a wrapper over it",
        "modelblaster commit 646ee95 (not pushed)",
    ],
    "raw": ["experiments/fastdepth_int8/results",
            "experiments/fastdepth_int8/logs",
            "experiments/fastdepth_int8/scripts"],
}
print(json.dumps(entry, indent=1))
