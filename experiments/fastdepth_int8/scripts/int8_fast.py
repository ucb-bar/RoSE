"""Vectorised, batched re-implementation of ModelBlaster's int8 golden simulator.

WHY this exists. The reference simulator in pipeline/extract_graph.py
(`_sim_conv2d_s8` and friends) is a five-deep Python loop nest. That is the
right shape for a *golden*: it is obviously the same accumulate order as the C
kernel, and it runs once per extraction. It is the wrong shape for an accuracy
study, which needs the quantised network evaluated on hundreds of frames --
FastDepth at 224x224 full-width is ~10^9 MACs, and the reference does them one
numpy row at a time.

So this module recomputes the SAME INTEGER ARITHMETIC with im2col + matmul.
Integer addition is associative and commutative, so reordering the accumulation
is not an approximation: given no int32 overflow (asserted in `_acc`, and
checked exhaustively by scripts/verify_fast_sim.py against the shipped golden),
the result is BIT-IDENTICAL, not merely close. The requantize tail is not
re-derived at all -- `_requantize_int` / `_requantize_int_per_oc` are imported
from extract_graph, so the rounding convention cannot drift.

Two entry points:
  patch()   monkeypatch the fast primitives INTO extract_graph, so a normal
            extract_int8() call produces its usual bit-exact golden in minutes
            instead of hours. Nothing on disk changes.
  run_ir()  execute a written-out graph.json + weights.npz over a BATCH of
            inputs, which is what the accuracy harness needs.
"""
from __future__ import annotations

import numpy as np

from modelblaster.pipeline.extract_graph import (  # noqa: F401
    _requantize_int, _requantize_int_per_oc,
)

_I32_MAX = np.int64(2 ** 31 - 1)


def _pad(in_4d_i32: np.ndarray, PH: int, PW: int, in_off: int) -> np.ndarray:
    """Pad in the RAW int8 domain with 0, then lift by `in_off`.

    The reference writes `in_row[ow] = input_offset` for an out-of-bounds tap
    while writing `in[..] + input_offset` for an in-bounds one, so the padding
    value in the raw domain is exactly 0. Padding after adding the offset
    (with 0) would be a different network.
    """
    if PH or PW:
        in_4d_i32 = np.pad(in_4d_i32, ((0, 0), (0, 0), (PH, PH), (PW, PW)))
    return in_4d_i32 + np.int32(in_off)


def _im2col(x: np.ndarray, KH, KW, SH, SW, OH, OW) -> np.ndarray:
    """(N, C, H, W) -> (N, OH*OW, C*KH*KW) via stride tricks (a view until the
    reshape forces one copy)."""
    N, C = x.shape[0], x.shape[1]
    s = x.strides
    win = np.lib.stride_tricks.as_strided(
        x, shape=(N, C, OH, OW, KH, KW),
        strides=(s[0], s[1], s[2] * SH, s[3] * SW, s[2], s[3]),
        writeable=False)
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N, OH * OW, C * KH * KW)


def _acc(in_4d, w_q, b_q, sh, in_off, filt_off) -> np.ndarray:
    """int32 conv accumulate, batched over N. Returns (N, OC, OH, OW) int32."""
    N = in_4d.shape[0]
    OH, OW, KH, KW = sh["OH"], sh["OW"], sh["KH"], sh["KW"]
    SH, SW, PH, PW = sh["SH"], sh["SW"], sh["PH"], sh["PW"]
    x = _pad(in_4d.astype(np.int32), PH, PW, in_off)
    cols = _im2col(x, KH, KW, SH, SW, OH, OW)                      # (N, L, K)
    w = (w_q.astype(np.int64) + np.int64(filt_off)).reshape(sh["OC"], -1)
    # numpy has no BLAS path for integer matmul (it falls back to a
    # single-threaded loop and is ~40x slower here), so accumulate in float64
    # WHEN THAT IS PROVABLY EXACT: every partial sum is an integer, and float64
    # represents every integer up to 2^53 exactly, so if the worst-case
    # |accumulator| bound stays under 2^53 the float path is not an
    # approximation. Otherwise fall back to exact int64.
    K = w.shape[1]
    bound = (int(np.abs(cols).max(initial=0)) * int(np.abs(w).max(initial=0))
             * K + int(np.abs(b_q).max(initial=0)))
    if bound < (1 << 53):
        acc = (cols.astype(np.float64) @ w.T.astype(np.float64)).astype(np.int64)
    else:
        acc = cols.astype(np.int64) @ w.T                          # (N, L, OC)
    acc += b_q.astype(np.int64)
    if np.abs(acc).max(initial=0) > _I32_MAX:
        # The reference accumulates in int32 and would WRAP here; refuse to
        # silently disagree with it rather than guessing the wrap.
        raise OverflowError("int32 conv accumulator overflow; fast sim would "
                            "not match the reference's wrapping behaviour")
    return acc.astype(np.int32).transpose(0, 2, 1).reshape(N, sh["OC"], OH, OW)


def conv2d_s8(in_4d, sh, q, w_q, b_q) -> np.ndarray:
    acc = _acc(in_4d, w_q, b_q, sh, q["input_offset"], q["filter_offset"])
    v = _requantize_int(acc, q["output_multiplier"], q["output_shift"])
    v = v + q["output_offset"]
    return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)


def conv2d_s8_pc(in_4d, sh, q, w_q, b_q, mult, shift) -> np.ndarray:
    acc = _acc(in_4d, w_q, b_q, sh, q["input_offset"], q["filter_offset"])
    v = _requantize_int_per_oc(acc, mult, shift, oc_axis=1)
    return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)


def depthwise_conv2d_s8_pc(in_4d, sh, q, w_q, b_q, mult, shift) -> np.ndarray:
    acc = _dw_acc(in_4d, w_q, b_q, sh, q["input_offset"], q["filter_offset"])
    v = _requantize_int_per_oc(acc, mult, shift, oc_axis=1)
    v = v + q["output_offset"]
    return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)


def _dw_acc(in_4d, w_q, b_q, sh, in_off, filt_off) -> np.ndarray:
    """Depthwise int32 accumulate: 25 (or 9) shifted adds, no im2col.

    Same per-channel independence as the reference's one-channel-at-a-time
    delegation, but every channel at once.
    """
    C = int(sh.get("groups", sh["OC"]))
    OH, OW, KH, KW = sh["OH"], sh["OW"], sh["KH"], sh["KW"]
    SH, SW, PH, PW = sh["SH"], sh["SW"], sh["PH"], sh["PW"]
    x = _pad(in_4d.astype(np.int32), PH, PW, in_off)
    wv = (w_q.astype(np.int32).reshape(C, KH, KW) + np.int32(filt_off))
    acc = np.broadcast_to(b_q.astype(np.int32).reshape(1, C, 1, 1),
                          (in_4d.shape[0], C, OH, OW)).astype(np.int32).copy()
    for kh in range(KH):
        for kw in range(KW):
            tap = x[:, :, kh:kh + OH * SH:SH, kw:kw + OW * SW:SW]
            acc += tap * wv[:, kh, kw].reshape(1, C, 1, 1)
    return acc


def depthwise_conv2d_s8(in_4d, sh, q, w_q, b_q) -> np.ndarray:
    acc = _dw_acc(in_4d, w_q, b_q, sh, q["input_offset"], q["filter_offset"])
    v = _requantize_int(acc, q["output_multiplier"], q["output_shift"])
    v = v + q["output_offset"]
    return np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)


# ---------------------------------------------------------------------------
# patch(): swap the fast primitives into extract_graph for the duration of a
# process, so extract_int8's own golden simulation runs at a usable speed.
# Signatures/semantics match the originals exactly (flat in_arr in, flat or
# 4-D out as the caller expects).
# ---------------------------------------------------------------------------

def patch() -> None:
    from modelblaster.pipeline import extract_graph as EG

    def _sim_conv2d_s8(in_arr, sh, q, w_q, b_q):
        in_4d = np.asarray(in_arr).reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"])
        return conv2d_s8(in_4d, sh, q, np.asarray(w_q), np.asarray(b_q))

    def _sim_depthwise_conv2d_s8(in_arr, sh, q, w_q, b_q):
        C = int(sh.get("groups", sh["OC"]))
        in_4d = np.asarray(in_arr).reshape(sh["N"], C, sh["IH"], sh["IW"])
        # the reference returns this one FLAT (it reshapes(-1) on the way out)
        return depthwise_conv2d_s8(in_4d, sh, q, np.asarray(w_q),
                                   np.asarray(b_q)).reshape(-1)

    def _sim_depthwise_conv2d_s8_pc(in_arr, sh, q, w_q, b_q, mult, shift):
        C = int(sh.get("groups", sh["OC"]))
        in_4d = np.asarray(in_arr).reshape(sh["N"], C, sh["IH"], sh["IW"])
        return depthwise_conv2d_s8_pc(in_4d, sh, q, np.asarray(w_q),
                                      np.asarray(b_q), mult, shift).reshape(-1)

    def _sim_conv2d_int32_acc(in_4d, w_q, b_q, sh, input_offset, filter_offset):
        return _acc(np.asarray(in_4d), np.asarray(w_q), np.asarray(b_q), sh,
                    input_offset, filter_offset)

    EG._sim_conv2d_s8 = _sim_conv2d_s8
    EG._sim_depthwise_conv2d_s8 = _sim_depthwise_conv2d_s8
    EG._sim_conv2d_int32_acc = _sim_conv2d_int32_acc
    EG._sim_depthwise_conv2d_s8_pc = _sim_depthwise_conv2d_s8_pc


# ---------------------------------------------------------------------------
# run_ir(): batched execution of a written-out int8 IR.
# ---------------------------------------------------------------------------

def _shape4(a, sh, keys):
    N = a.shape[0]
    return a.reshape(N, sh[keys[0]], sh[keys[1]], sh[keys[2]])


def run_ir(ir: dict, W: dict, x_i8: np.ndarray, keep_all: bool = False):
    """Run the int8 graph over a batch. `x_i8` is (N, C, H, W) int8; returns
    the output tensor as int8 with its leading batch dim. With keep_all, also
    returns the whole {tensor_name: int8 activation} map, which is what the
    per-layer error attribution needs."""
    in_names = ir["input"].get("tensors") or [ir["input"]["tensor"]]
    if len(in_names) != 1:
        raise NotImplementedError("run_ir handles single-input graphs only")
    act = {in_names[0]: x_i8}
    N = x_i8.shape[0]
    for op in ir["ops"]:
        k, o = op["op"], op["outputs"][0]
        a0 = act[op["inputs"][0]]
        sh = op.get("shape", {})
        q = op.get("quant", {})
        if k == "conv2d_s8":
            act[o] = conv2d_s8(_shape4(a0, sh, ("IC", "IH", "IW")), sh, q,
                               W[op["weight"]], W[op["bias"]])
        elif k == "conv2d_s8_pc":
            act[o] = conv2d_s8_pc(
                _shape4(a0, sh, ("IC", "IH", "IW")), sh, q,
                W[op["weight"]], W[op["bias"]],
                W[q["output_multiplier_per_oc_key"]],
                W[q["output_shift_per_oc_key"]])
        elif k == "depthwise_conv2d_s8":
            C = int(sh.get("groups", sh["OC"]))
            act[o] = depthwise_conv2d_s8(
                a0.reshape(N, C, sh["IH"], sh["IW"]), sh, q,
                W[op["weight"]], W[op["bias"]])
        elif k == "depthwise_conv2d_s8_pc":
            Cc = int(sh.get("groups", sh["OC"]))
            act[o] = depthwise_conv2d_s8_pc(
                a0.reshape(N, Cc, sh["IH"], sh["IW"]), sh, q,
                W[op["weight"]], W[op["bias"]],
                W[q["output_multiplier_per_oc_key"]],
                W[q["output_shift_per_oc_key"]])
        elif k == "relu_s8":
            act[o] = np.maximum(a0, 0).astype(np.int8)
        elif k == "relu6_s8":
            act[o] = np.clip(a0, 0, op["clamp_max"]).astype(np.int8)
        elif k == "add_s8":
            a = act[op["inputs"][0]].astype(np.float32) * np.float32(q["scale_a"])
            b = act[op["inputs"][1]].astype(np.float32) * np.float32(q["scale_b"])
            v = np.round((a + b) / np.float32(q["scale_out"])).astype(np.int32)
            act[o] = np.clip(v, q["activation_min"],
                             q["activation_max"]).astype(np.int8)
        elif k == "upsample_nearest_s8":
            s = sh["scale"]
            t = _shape4(a0, sh, ("C", "IH", "IW"))
            act[o] = np.repeat(np.repeat(t, s, axis=2), s, axis=3)
        elif k == "batchnorm2d_s8":
            t = _shape4(a0, sh, ("C", "H", "W")).astype(np.float32)
            y = (W[op["weight"]].astype(np.float32)[None, :, None, None]
                 * (t * np.float32(q["scale_in"]))
                 + W[op["bias"]].astype(np.float32)[None, :, None, None])
            v = np.round(y / np.float32(q["scale_out"])).astype(np.int32)
            act[o] = np.clip(v, q["activation_min"],
                             q["activation_max"]).astype(np.int8)
        elif k == "view":
            act[o] = a0
        else:
            raise NotImplementedError(
                f"run_ir: op {k!r} not implemented (extend it rather than "
                f"approximating -- an unmodelled op is a silent accuracy lie)")
    outs = ir["output"].get("tensors") or [ir["output"]["tensor"]]
    if len(outs) != 1:
        raise NotImplementedError("run_ir handles single-output graphs only")
    return (act[outs[0]], act) if keep_all else act[outs[0]]
