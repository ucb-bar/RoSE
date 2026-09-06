#!/usr/bin/env python3
"""Depth accuracy of a quantised FastDepth IR on NYU val, PER-IMAGE protocol.

Runs the int8 graph itself (int8_fast.run_ir, proven bit-exact against the
shipped golden by verify_fast_sim.py) -- not a fake-quant stand-in -- so the
numbers describe the network that would actually run on the SoC, including
the int8 input surface and the int8 output surface.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")

_G = {}


def _init(ir_dir):
    import int8_fast
    _G["f"] = int8_fast
    _G["ir"] = json.load(open(os.path.join(ir_dir, "graph.json")))
    _G["W"] = dict(np.load(os.path.join(ir_dir, "weights.npz")))
    t = _G["ir"]["tensors"]
    _G["s_in"] = float(t[_G["ir"]["input"]["tensor"]]["quant"]["scale"])
    _G["s_out"] = float(t[_G["ir"]["output"]["tensors"][0]]["quant"]["scale"])


def _chunk(args):
    """Quantise -> run int8 graph -> dequantise, for one slice of frames."""
    rgb, = args
    q = np.clip(np.round(rgb / _G["s_in"]), -127, 127).astype(np.int8)
    y = _G["f"].run_ir(_G["ir"], _G["W"], q)
    return (y.astype(np.float32) * np.float32(_G["s_out"])).reshape(y.shape[0],
                                                                    *y.shape[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-pred", default=None)
    a = ap.parse_args()

    C.apply_trained_env()
    rgb, gt, names = C.load_val(a.n)
    print(f"[int8:{a.tag}] {len(rgb)} frames from {a.ir}", flush=True)
    batches = [(rgb[s:s + a.batch],) for s in range(0, len(rgb), a.batch)]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init, initargs=(a.ir,)) as pool:
            preds = list(pool.imap(_chunk, batches, chunksize=1))
    else:
        _init(a.ir)
        preds = [_chunk(b) for b in batches]
    pred = np.concatenate(preds)
    rows = [C.per_image(pred[i], gt[i]) for i in range(len(pred))]
    ev = C.aggregate(rows)
    print(C.fmt(ev, f"int8:{a.tag}"))
    print(f"   bands near {ev['rmse_0.5_2']:.3f} mid {ev['rmse_2_5']:.3f} "
          f"far {ev['rmse_5_10']:.3f}   {time.time() - t0:.1f}s")
    print(C.save(a.tag, {"arm": "int8", "ir": a.ir, "n": len(rgb),
                         "metrics": ev}))
    if a.save_pred:
        np.savez(a.save_pred, pred=pred.astype(np.float32), names=names)


if __name__ == "__main__":
    main()
