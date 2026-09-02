/* source: curated */
/* algorithm: gemmini_tiled_conv_pool */
/* accuracy_class: bit_exact */
/* FIXED 2026-08-28 (un-archived): the previous version of this kernel used
 * a per-channel identity conv with weight=+1 to move data through
 * gemmini's mvout-pool datapath at scale=ACC_SCALE_IDENTITY, and measured
 * max_abs_err=17 in isolation (kernel_opt_log.jsonl id 1100/gemmini-
 * correct-cost) despite declaring accuracy_class=bit_exact — archived to
 * archive/ pending a fix.
 *
 * ROOT CAUSE (this fix): cores/gemmini/include/gemmini_params.h defines
 *   ACC_SCALE_IDENTITY  ((acc_scale_t)1 << 30)
 *   ACC_SCALE(x, scale) = round_half_up((x * scale) + 2^30, >> 31)
 * i.e. ACC_SCALE_IDENTITY represents the fixed-point value 0.5, not 1.0 —
 * a genuine Q31-format limitation, not a typo: acc_scale_t is a SIGNED
 * int32, and the value that would represent an exact 1.0 in this >>31
 * convention is 2^31, which overflows int32 (max representable is
 * 2^31 - 1). Whoever chose this bitstream's ACC_SCALE_IDENTITY picked
 * 2^30 (0.5) rather than 2^31-1 (~0.9999999995); either would have
 * worked at mvout time (see kernels/gemmini/gemmini_conv2d_s8_gemmini_
 * tiled_conv.c's header for the SEPARATE, unrelated single-vs-double-
 * rounding drift that affects a REAL non-identity conv scale — 2^31-1's
 * ~5e-10 relative error is negligible by comparison), but 2^30 also
 * works AS LONG AS THE ACCUMULATOR INPUT IS PRE-DOUBLED: verified
 * exhaustively (all int8 v in [-128,127]) that
 *   ACC_SCALE(2*v, ACC_SCALE_IDENTITY) == v   exactly, for every v
 * while ACC_SCALE(v, ACC_SCALE_IDENTITY) == round(v/2) (max diff 64).
 * The previous version's identity conv used weight=+1, giving
 * accumulator x = v*1 = v (NOT pre-doubled) — hence the ~2x-attenuated,
 * badly wrong per-element values that propagated to max_abs_err=17
 * after the downstream linear layer. FIX: weight=+2 per channel, giving
 * accumulator x = v*2 = 2v, which ACC_SCALE_IDENTITY recovers EXACTLY
 * (provably, not just measured-zero-in-practice — this is why the
 * accuracy_class above is BIT_EXACT, unlike the cat2/3/4_c1_s8 MVIN_SCALE
 * trick which is NUMERIC_DRIFT due to a real ties-to-even-vs-away-from-
 * zero rounding mismatch. Here there is no rounding at all: 2v is exactly
 * representable and exactly recovered for every int8 v). No overflow risk
 * (2*v in [-256,254], acc_t is int32).
 *
 * ISOLATION-VERIFIED (this kernel alone curated via a private
 * GLOBAL_CURATED_DIR, every other op forced to spec.reference_impl, per
 * the campaign's isolation-testing rule): dronet/gemmini_q31 on spike,
 * whole-model max_abs_err=0 max_rel_err=0 (was 17 with weight=+1).
 * kernel_picks.json confirmed this file's algorithm was actually
 * selected (source=curated, algorithm=gemmini_tiled_conv_pool), not a
 * silent fallback. See experiments/kernel_opt_log.jsonl id ~1900s.
 *
 * origin: tiled_conv_dw_auto — gemmini's depthwise-conv path with the
 *         pool tail enabled. We turn the conv into a per-channel
 *         passthrough (kernel_dim=1, stride=1, padding=0,
 *         weights = +2 per channel [see FIX above], bias = NULL, act = 0,
 *         scale = ACC_SCALE_IDENTITY) so the conv produces output[c,h,w]
 *         = input[c,h,w] in the accumulator (exactly, after ACC_SCALE's
 *         halving), and then the mvout pool unit takes a max over each
 *         KH×KW window with stride SH==SW while writing to DRAM. Per
 *         kernel_opt_log id 1303's finding (confirmed by reading
 *         cores/gemmini/include/gemmini.h's tiled_conv pool loop): pool
 *         quantizes each position individually THEN takes the max over
 *         already-quantized int8 values — since our per-position
 *         quantize is now exact (not just "close"), the max over them is
 *         exact too (max of exact values is exact; no rounding is done a
 *         second time after the max).
 *
 *         Why depthwise rather than the full tiled_conv_auto: a full
 *         passthrough conv would need a C×C identity weight tensor,
 *         which is O(C^2) and gets large for yolov8 (C=256: 64KB
 *         versus C=256 for dw). The dw variant treats each channel
 *         independently and only needs C int8 weights.
 *
 *         Constraints (otherwise scalar fallback):
 *           - KH == KW, SH == SW, PH == PW, DH == DW == 1
 *             (gemmini's pool params are scalar; non-square shapes
 *              don't map.)
 *           - PH == 0 (pool padding semantics differ from the spec:
 *              gemmini's pool fills OOB with 0, the spec fills with
 *              INT8_MIN. With PH==0 the OOB path is never taken so
 *              the difference doesn't matter; with PH>0 they can
 *              disagree on inputs that have negative values at the
 *              boundary, which we guard against by falling back.)
 *           - tensor fits the static workspace.
 *
 *         All shapes we care about (LeNet, dronet, yolov8 SPPF-adjacent)
 *         use square windows with PH==0, so they take the gemmini path.
 *         Asymmetric/padded maxpools fall back to scalar (exact either
 *         way — the fallback below is the same direct sliding-window max
 *         the RVV/scalar `direct` algorithm uses, just inlined here so
 *         this file is a complete drop-in replacement).
 */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW)
{
    /* 512 KB covers every maxpool input we encounter:
     *   dronet: C=32, IH=IW=64           ->  128 KB
     *   yolov8_nano backbone:  C=64, 40×40 -> 102 KB
     *   yolov8_nano SPPF:      C=128, 20×20 ->  51 KB
     * Function-scope enum so the symbol stays local — `GEMMINI_WS_BYTES`
     * is also used at file scope by `gemmini_q31_conv2d_s8_gemmini_tiled_conv.c`,
     * and the two would collide once concatenated into kernels.c. */
    enum { GEMMINI_WS_BYTES = 512 * 1024 };
    enum { MAXPOOL_MAX_CHANNELS = 1024 };
    static elem_t ws_input  [GEMMINI_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output [GEMMINI_WS_BYTES] __attribute__((aligned(64)));
    /* Per-channel passthrough weights: +2 for every channel (see the FIX
     * note above -- +1 would silently halve every value through
     * ACC_SCALE_IDENTITY), init once. */
    static elem_t ws_weights[MAXPOOL_MAX_CHANNELS] __attribute__((aligned(64)));
    static int    ws_weights_inited = 0;

    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;

    bool gemmini_ok =
           KH == KW && SH == SW && PH == PW
        && DH == 1 && DW == 1
        && PH == 0
        && C <= MAXPOOL_MAX_CHANNELS
        && (size_t)(N * C * IH * IW) <= GEMMINI_WS_BYTES
        && (size_t)(N * C * OH * OW) <= GEMMINI_WS_BYTES;

    if (!gemmini_ok) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int8_t m = INT8_MIN;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh*SH - PH + kh*DH;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow*SW - PW + kw*DW;
                                if (iw < 0 || iw >= IW) continue;
                                int8_t v = input[((n*C + c)*IH + ih)*IW + iw];
                                if (v > m) m = v;
                            }
                        }
                        output[((n*C + c)*OH + oh)*OW + ow] = m;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    if (!ws_weights_inited) {
        for (int i = 0; i < MAXPOOL_MAX_CHANNELS; i++) {
            ws_weights[i] = 2;
        }
        ws_weights_inited = 1;
    }

    gemmini_flush(0);

    /* NCHW -> NHWC into ws_input. */
    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < C; c++)
                    ws_input[((n*IH + h)*IW + w)*C + c] =
                        input[((n*C + c)*IH + h)*IW + w];

    asm volatile("fence" ::: "memory");

    /* Passthrough conv (kernel_dim=1, stride=1, padding=0) + max-pool tail.
     *   in_row_dim = IH, in_col_dim = IW   (passthrough)
     *   out_row_dim = IH, out_col_dim = IW (passthrough conv output)
     *   pool produces (OH, OW) per the spec formula (DH=DW=1 here). */
    tiled_conv_dw_auto(
        N, IH, IW,
        C, IH, IW,
        /* stride       = */ 1,
        /* padding      = */ 0,
        /* kernel_dim   = */ 1,
        ws_input, ws_weights,
        /* bias         = */ NULL,
        ws_output,
        /* act          = */ 0,             /* NO_ACTIVATION */
        /* scale        = */ ACC_SCALE_IDENTITY,
        /* pool_size    = */ KH,
        /* pool_stride  = */ SH,
        /* pool_padding = */ PH,
        WS
    );

    gemmini_fence();
    gemmini_flush(0);

    /* NHWC -> NCHW. */
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < OH; h++)
                for (int w = 0; w < OW; w++)
                    output[((n*C + c)*OH + h)*OW + w] =
                        ws_output[((n*OH + h)*OW + w)*C + c];
}
