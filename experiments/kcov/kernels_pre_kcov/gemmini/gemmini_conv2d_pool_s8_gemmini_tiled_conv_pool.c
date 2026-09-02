/* source: curated */
/* algorithm: gemmini_tiled_conv_pool */
/* accuracy_class: numeric_drift */
/* Fused conv2d_s8 + maxpool2d_s8 (extended fusion, gated by
 * --enable-fusion / MB_ENABLE_FUSION=1 -- see CONV2D_POOL_S8 in
 * pipeline/reference_kernels.py and the detector in
 * pipeline/extract_graph.py). tiled_conv_auto's native pool_size/
 * pool_stride/pool_padding tail is applied to the SAME accumulator the
 * conv itself produces, so the pre-pool [N,OC,OH,OW] intermediate is
 * NEVER materialized to DRAM or the gemmini scratchpad -- the pooled
 * [N,OC,OHp,OWp] result is what actually gets DMA'd out. This is the
 * "for free" pooling kernel_opt_log id 1303/gemmini-fusion-tradeoff
 * identified: gemmini's mvout-pool loop quantizes each PRE-pool position
 * individually via scale_and_sat, THEN takes the running max over the
 * already-quantized int8 values -- since round is monotonic and commutes
 * with max, this is mathematically identical to "round each position
 * with the reference's own formula, then max" PROVIDED the per-position
 * round itself is exact. It is not, on this Q0.31 config -- see below.
 *
 * ACCURACY (why NUMERIC_DRIFT, and the actual computed bound, not a
 * label): this kernel folds (output_multiplier, output_shift) into ONE
 * Q0.31 hardware round-shift-by-31, exactly like the sibling
 * gemmini_conv2d_s8_gemmini_tiled_conv.c (see that file's header for the
 * full derivation) -- the reference/golden computes TWO independent
 * rounds (round(round(acc*mult,2^31),2^shift)), so pre-folding the scale
 * before the single hardware round is structurally not the same
 * operation. The per-layer worst-case error is NOT "roughly acc*delta"
 * as a loose intuition -- it was computed EXACTLY (exhaustive integer
 * sweep over the full theoretical accumulator range for each of dronet's
 * 10 and yolov8n's 63 real conv layers' actual (mult,shift) pairs, not
 * just estimated): every single layer in both models bounds to EXACTLY
 * 1 LSB pre-saturation/pre-propagation, because the acc-scaling
 * structural term (|acc_max * delta| / 2^31, delta = round(mult,shift)
 * - mult/2^shift, |delta| <= 0.5 by construction) stays 2-4 orders of
 * magnitude below the fixed ~1-unit double-vs-single-rounding floor for
 * any accumulator size a real conv layer produces (would need
 * IC*KH*KW > ~500,000 taps before the acc-scaling term would dominate;
 * the biggest real layer measured, yolov8n's 256x3x3=2304-tap layer, is
 * ~220x below that). The pool stage adds ZERO additional error on top of
 * that per-layer conv bound (round commutes with max, proven above).
 * Measured whole-model max_abs_err on dronet (isolated: this kernel's
 * plain conv-only sibling substituted for ALL 10 conv layers, current
 * tree) = 4, consistent with 9-10 sequential ~1-LSB-bounded layers
 * through 3 residual merges -- see gemmini_conv2d_s8_gemmini_tiled_conv.c
 * and experiments/kernel_opt_log.jsonl id ~1900s for the full
 * measurement + the sweep script.
 *
 * WEIGHT LAYOUT CONTRACT: same as the sibling conv2d_s8 kernels -- weight
 * arrives pre-packed HWIO ([KH*KW*IC, OC]) by
 * generate_skeleton.py::_backend_pack_weight, passed straight to
 * tiled_conv_auto with no runtime transpose.
 *
 * pool_padding must be 0 (gemmini's pool fills OOB with 0; the spec
 * fills with INT8_MIN -- with pool_PH==0 the OOB path is never taken so
 * the difference doesn't matter, matching the standalone maxpool2d_s8
 * kernel's same constraint) and pool must be square
 * (pool_KH==pool_KW, pool_SH==pool_SW) with no dilation
 * (pool_DH==pool_DW==1) -- gemmini's pool params are scalar. Anything
 * else, or a non-square/offset conv, falls back to the portable scalar
 * reference (conv then a separate pool pass, both exact by construction
 * -- see CONV2D_POOL_S8.reference_impl). */

#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

enum { GEMMINI_CONV_POOL_WS_BYTES = 512 * 1024 };

void kernel_conv2d_pool_s8(const int8_t *input, const int8_t *weight,
                           const int32_t *bias, int8_t *output,
                           int N, int IC, int IH, int IW, int OC,
                           int KH, int KW, int SH, int SW, int PH, int PW,
                           int input_offset, int filter_offset, int output_offset,
                           int output_multiplier, int output_shift,
                           int activation_min, int activation_max,
                           int pool_KH, int pool_KW, int pool_SH, int pool_SW,
                           int pool_PH, int pool_PW, int pool_DH, int pool_DW)
{
    static elem_t ws_input  [GEMMINI_CONV_POOL_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output [GEMMINI_CONV_POOL_WS_BYTES] __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int OHp = (OH + 2*pool_PH - pool_DH*(pool_KH-1) - 1) / pool_SH + 1;
    int OWp = (OW + 2*pool_PW - pool_DW*(pool_KW-1) - 1) / pool_SW + 1;

    bool gemmini_ok =
           KH == KW && SH == SW && PH == PW
        && input_offset == 0 && filter_offset == 0 && output_offset == 0
        && output_shift >= 0 && output_shift <= 30
        && pool_KH == pool_KW && pool_SH == pool_SW
        && pool_PH == 0 && pool_PW == 0
        && pool_DH == 1 && pool_DW == 1
        && (size_t)(N * IH * IW * IC) <= GEMMINI_CONV_POOL_WS_BYTES
        && (size_t)(N * OHp * OWp * OC) <= GEMMINI_CONV_POOL_WS_BYTES;

    if (!gemmini_ok) {
        /* Portable scalar fallback: conv2d_s8 math into a static
         * intermediate, then a direct sliding-window max -- exact by
         * construction (mirrors CONV2D_POOL_S8.reference_impl). */
        static int8_t tmp[GEMMINI_CONV_POOL_WS_BYTES];
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        in_v = input_offset;
                                    else
                                        in_v = (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                                             + input_offset;
                                    int32_t w_v = (int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                 + filter_offset;
                                    acc += in_v * w_v;
                                }
                            }
                        }
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        size_t idx = (((size_t)n*OC + oc)*OH + oh)*OW + ow;
                        tmp[idx] = (int8_t)scaled;
                    }
                }
            }
        }
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int ohp = 0; ohp < OHp; ohp++) {
                    for (int owp = 0; owp < OWp; owp++) {
                        int8_t m = INT8_MIN;
                        for (int kh = 0; kh < pool_KH; kh++) {
                            int oh = ohp*pool_SH - pool_PH + kh*pool_DH;
                            if (oh < 0 || oh >= OH) continue;
                            for (int kw = 0; kw < pool_KW; kw++) {
                                int ow = owp*pool_SW - pool_PW + kw*pool_DW;
                                if (ow < 0 || ow >= OW) continue;
                                size_t idx = (((size_t)n*OC + oc)*OH + oh)*OW + ow;
                                int8_t v = tmp[idx];
                                if (v > m) m = v;
                            }
                        }
                        size_t out_idx = (((size_t)n*OC + oc)*OHp + ohp)*OWp + owp;
                        output[out_idx] = m;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    /* Transpose input NCHW -> NHWC into ws_input (same blocked pattern as
     * the plain conv2d_s8/gemmini_tiled_conv sibling). */
    {
        const int TB = 32;
        int HW = IH * IW;
        for (int n = 0; n < N; n++) {
            const int8_t *inb = input   + (size_t)n * IC * HW;
            elem_t       *ob  = ws_input + (size_t)n * HW * IC;
            for (int p0 = 0; p0 < HW; p0 += TB) {
                int pn = HW - p0 < TB ? HW - p0 : TB;
                for (int c0 = 0; c0 < IC; c0 += TB) {
                    int cn = IC - c0 < TB ? IC - c0 : TB;
                    for (int c = 0; c < cn; c++) {
                        const int8_t *s = inb + (size_t)(c0 + c) * HW + p0;
                        elem_t       *d = ob  + (size_t)p0 * IC + (c0 + c);
                        for (int p = 0; p < pn; p++) d[(size_t)p * IC] = s[p];
                    }
                }
            }
        }
    }

    /* weight is already HWIO-packed by the codegen -- pass straight
     * through, no ws_weight copy (same contract as the sibling kernel). */

    int32_t scale_q31 = output_shift == 0
        ? output_multiplier
        : (int32_t)(((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;

    /* RELU (1) clamps to [0, INT8_MAX]; NO_ACTIVATION (0) allows full range. */
    int act_kind = (activation_min == 0) ? 1 : 0;

    asm volatile("fence" ::: "memory");

    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, weight, bias, ws_output,
        act_kind, scale,
        pool_KH, pool_SH, pool_PH,
        WS
    );

    /* tiled_conv_auto's body does NOT end with a gemmini_fence -- without
     * this explicit drain the post-conv NHWC->NCHW read races in-flight
     * mvout DMAs (same reasoning as the sibling conv2d_s8 kernel). */
    gemmini_fence();
    gemmini_flush(0);

    /* Transpose POOLED output NHWC -> NCHW. ws_output holds [N,OHp,OWp,OC]
     * -- the pre-pool [N,OH,OW,OC] intermediate was never written here at
     * all (that's the "for free" part). */
    {
        const int TB = 32;
        int OHWp = OHp * OWp;
        for (int n = 0; n < N; n++) {
            int8_t       *outb = output    + (size_t)n * OC * OHWp;
            const elem_t *ob   = ws_output + (size_t)n * OHWp * OC;
            for (int p0 = 0; p0 < OHWp; p0 += TB) {
                int pn = OHWp - p0 < TB ? OHWp - p0 : TB;
                for (int c0 = 0; c0 < OC; c0 += TB) {
                    int cn = OC - c0 < TB ? OC - c0 : TB;
                    for (int c = 0; c < cn; c++) {
                        int8_t       *d = outb + (size_t)(c0 + c) * OHWp + p0;
                        const elem_t *s = ob   + (size_t)p0 * OC + (c0 + c);
                        for (int p = 0; p < pn; p++) d[p] = s[(size_t)p * OC];
                    }
                }
            }
        }
    }

    /* Post-clamp for activation_max < 127 (gemmini RELU only handles
     * min==0). Applying this AFTER pooling instead of before is exact --
     * clamp is monotonic non-decreasing, so it commutes with max the same
     * way round does (see header). Cheaper too: OHp*OWp*OC elements
     * instead of OH*OW*OC. */
    if (activation_max < 127) {
        int n_out = N * OC * OHp * OWp;
        for (int i = 0; i < n_out; i++) {
            int v = output[i];
            if (v > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
