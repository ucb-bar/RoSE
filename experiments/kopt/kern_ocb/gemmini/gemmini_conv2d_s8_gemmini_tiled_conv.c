/* source: curated */
/* algorithm: gemmini_tiled_conv */
/* accuracy_class: numeric_drift */
/* origin: tiled_conv_auto — gemmini hardware im2col + GEMM + float-scale
 *         requantize in one call.  Validated on Saturn FireSim May 2026.
 *         Square kernels only (KH==KW, SH==SW, PH==PW); asymmetric offsets
 *         must be zero (symmetric per-tensor int8 from extract_int8).
 *         Float-scale introduces ≤1 LSB drift vs Q0.31 golden.
 *
 *         WEIGHT LAYOUT CONTRACT: this kernel expects `weight` to already
 *         be in flat HWIO layout (= `[KH*KW*IC, OC]`, the form
 *         `tiled_conv_auto` consumes directly).  The skeleton emitter at
 *         modelblaster/pipeline/generate_skeleton.py::_backend_pack_weight
 *         applies the OIHW→HWIO permutation at codegen time when
 *         `--backend gemmini`, so the runtime weight-transpose loop that
 *         used to live here (which copied into a per-call ws_weight
 *         workspace) is gone — the kernel passes `weight` straight into
 *         tiled_conv_auto.  Saves ~OC*IC*KH*KW int8 copies per layer
 *         (yolov8's biggest layer: 128 IC × 256 OC × 3×3 = 295 KB of
 *         scalar copy that's now zero). */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

/* 512 KB covers all square conv layers in dronet and yolov8_nano.
 *   dronet max weight: IC=128, KH=3, KW=3, OC=128 →  144 KB
 *   yolov8 max weight: IC=128, KH=3, KW=3, OC=256 →  288 KB
 *   yolov8 max input:  IC=3,   IH=160, IW=160      →   75 KB
 *   yolov8 max output: IC=16,  OH=80,  OW=80        →  100 KB
 * All three fit well within 512 KB. */
enum { GEMMINI_WS_BYTES = 512 * 1024 };

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    /* ws_weight is gone — the weight emitter pre-packs to HWIO at
     * codegen time, so we hand `weight` directly to tiled_conv_auto.
     * ws_input / ws_output are still needed for the activation
     * NCHW↔NHWC transposes on entry / exit. */
    static elem_t ws_input  [GEMMINI_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output [GEMMINI_WS_BYTES] __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    /* Fall back to scalar for non-square kernels, non-zero offsets, tensors
     * that exceed the static workspace, or output_shift outside the foldable
     * Q0.31 range. The earlier B_rows = (OC/DIM)*KH*KW*IC > BANK_NUM*BANK_ROWS/2
     * guard was a misdiagnosis — tiled_conv_auto handles arbitrary B_rows by
     * splitting the spatial loop. The original symptom (mcause=1, mepc=0)
     * came from a missing post-call gemmini_fence(), which we now emit. */
    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0
            || output_offset != 0
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
            || output_shift < 0 || output_shift > 30
#endif
            || (size_t)(N * IH * IW * IC) > GEMMINI_WS_BYTES
            || (size_t)(N * OH * OW * OC)  > GEMMINI_WS_BYTES) {
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
                                    /* weight is HWIO-packed per the
                                     * file-header contract:
                                     * idx = ((kh*KW + kw)*IC + ic)*OC + oc */
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                   + filter_offset);
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
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Transpose input NCHW → NHWC into ws_input.
     * Blocked (TB x TB) so the contiguous-read side and the IC-strided
     * write side both stay resident in L1. The old 4-deep loop nest
     * recomputed ((n*IH+h)*IW+w)*IC+c and ((n*IC+c)*IH+h)*IW+w per
     * element; on dronet conv0 the two transposes move 137,984 bytes and
     * were the single largest term in the layer's 1.75M cycles. */
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

    /* Weight is already in flat HWIO layout ([KH*KW*IC, OC]) — see the
     * file-header WEIGHT LAYOUT CONTRACT.  The OIHW→HWIO permutation
     * happens at codegen time in
     * modelblaster/pipeline/generate_skeleton.py::_backend_pack_weight when
     * --backend=gemmini, so we pass `weight` directly into
     * tiled_conv_auto without going through a ws_weight workspace. */

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    /* Q31 gemmini config: acc_scale_t = SInt(32). HW mvout requantize is
     *   y = sat_int8((acc * scale + (1<<30)) >> 31)
     * The TFLite/modelblaster formula we want is
     *   y = sat_int8((acc * mult + (1<<30)) >> (31 + output_shift))
     * Fold (mult, shift) → single Q0.31 multiplier:
     *   scale = (mult + (1<<(s-1))) >> s   (compile-time round-to-nearest).
     * Differs from two-stage rounding by ≤1 LSB per layer; for deep nets
     * (yolov8) the drift accumulates and can exceed atol — those layers
     * fall back to gemmini_im2col_full_C (CPU im2col + scalar Q0.31). */
    int32_t scale_q31 = output_shift == 0
        ? output_multiplier
        : (int32_t)(((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;
#else
    /* Default (Saturn FireSim) gemmini config: acc_scale_t = float.
     * effective_scale = output_multiplier * 2^(-(31 + output_shift)). f32
     * has ~24 bits of mantissa precision; loses ~1 LSB per layer. */
    float scale = ldexpf((float)output_multiplier, -(31 + output_shift));
#endif

    /* RELU (1) clamps to [0, INT8_MAX]; NO_ACTIVATION (0) allows full range. */
    int act_kind = (activation_min == 0) ? 1 : 0;

    /* Drain CPU store buffer before gemmini mvin reads ws_input/ws_weight. */
    asm volatile("fence" ::: "memory");

    /* tiled_conv_auto: gemmini does im2col + GEMM + requantize in hardware.
     * WS = weight-stationary dataflow. */
    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, weight, bias, ws_output,   /* weight is pre-packed HWIO from codegen */
        act_kind, scale,
        0, 0, 0,
        WS
    );

    /* tiled_conv_auto's body (tiled_conv) does NOT end with a
     * gemmini_fence — unlike tiled_matmul_outer_eigen.  Without this
     * explicit drain, the post-conv NHWC->NCHW read and the next op's
     * gemmini_flush race with in-flight mvout DMAs and corrupt memory
     * (FireSim Saturn: mcause=1, mepc=0). */
    gemmini_fence();
    gemmini_flush(0);

    /* Transpose output NHWC → NCHW (same blocking as the input side). */
    {
        const int TB = 32;
        int OHW = OH * OW;
        for (int n = 0; n < N; n++) {
            int8_t       *outb = output    + (size_t)n * OC * OHW;
            const elem_t *ob   = ws_output + (size_t)n * OHW * OC;
            for (int p0 = 0; p0 < OHW; p0 += TB) {
                int pn = OHW - p0 < TB ? OHW - p0 : TB;
                for (int c0 = 0; c0 < OC; c0 += TB) {
                    int cn = OC - c0 < TB ? OC - c0 : TB;
                    for (int c = 0; c < cn; c++) {
                        int8_t       *d = outb + (size_t)(c0 + c) * OHW + p0;
                        const elem_t *s = ob   + (size_t)p0 * OC + (c0 + c);
                        for (int p = 0; p < pn; p++) d[p] = s[(size_t)p * OC];
                    }
                }
            }
        }
    }

    /* Post-clamp for activation_max < 127 (gemmini RELU only handles min==0). */
    if (activation_max < 127) {
        int n_out = N * OC * OH * OW;
        for (int i = 0; i < n_out; i++) {
            int v = output[i];
            if (v > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
