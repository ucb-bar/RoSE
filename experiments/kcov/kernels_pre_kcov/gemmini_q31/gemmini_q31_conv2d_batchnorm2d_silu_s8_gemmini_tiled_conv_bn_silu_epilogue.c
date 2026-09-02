/* source: curated */
/* algorithm: gemmini_tiled_conv_bn_silu_epilogue */
/* accuracy_class: numeric_drift */
/* origin: FAST fused Conv2d+BatchNorm+SiLU for gemmini_q31. The conv MAC +
 *         HW im2col + HW Q0.31 requantize run on the systolic array via
 *         tiled_conv_auto (same speed as the standalone gemmini_tiled_conv),
 *         producing the int8 conv output; the BatchNorm per-channel affine
 *         and SiLU are then applied as a cheap scalar epilogue during the
 *         NHWC->NCHW write-back. Keeps the expensive MAC/im2col on HW
 *         (single-digit-M cycles per conv) instead of the ~40M-cycle CPU
 *         im2col of the im2col_full_C variant.
 *
 *         ACCURACY: the conv requantize here is Gemmini's SINGLE-STAGE Q0.31
 *         mvout (mult/shift folded to one Q0.31 scale), which differs from
 *         the two-stage golden by <=1 LSB/layer -> numeric_drift, covered by
 *         the gemmini_q31 atol=128 envelope. (Gemmini's HW im2col is
 *         fundamentally coupled to the HW requantize: the conv API exposes
 *         only int8 output, no raw-int32/full_C mode, so a truly bit-exact
 *         HW-im2col conv is not achievable without a Gemmini change. The
 *         bit-exact fallback is the sibling im2col_full_C_bn_silu kernel.)
 *
 *         WEIGHT LAYOUT CONTRACT: `weight` is pre-packed flat HWIO
 *         ([KH*KW*IC, OC]) by generate_skeleton.py::_backend_pack_weight;
 *         handed straight to tiled_conv_auto. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

enum { CBNST_WS_BYTES = 512 * 1024 };

void kernel_conv2d_batchnorm2d_silu_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max,
    float silu_scale_in, float silu_scale_out,
    int silu_activation_min, int silu_activation_max)
{
    static elem_t ws_input  [CBNST_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output [CBNST_WS_BYTES] __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    /* Scalar fallback (BIT-EXACT two-stage conv Q0.31 + bn + silu) for
     * non-square kernels, non-zero offsets, oversized tensors, or a
     * conv_output_shift outside the foldable Q0.31 range. */
    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0 || conv_output_offset != 0
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
            || conv_output_shift < 0 || conv_output_shift > 30
#endif
            || (size_t)(N * IH * IW * IC) > CBNST_WS_BYTES
            || (size_t)(N * OH * OW * OC) > CBNST_WS_BYTES) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                float bn_s = bn_scale[oc], bn_b = bn_bias[oc];
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++)
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        ? input_offset
                                        : (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw] + input_offset;
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc] + filter_offset);
                                }
                            }
                        int64_t prod = (int64_t)acc * (int64_t)conv_output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (conv_output_shift > 0)
                            scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (conv_output_shift-1))) >> conv_output_shift);
                        else if (conv_output_shift < 0)
                            scaled <<= (-conv_output_shift);
                        scaled += conv_output_offset;
                        if (scaled < conv_activation_min) scaled = conv_activation_min;
                        if (scaled > conv_activation_max) scaled = conv_activation_max;
                        float fv = (float)(int8_t)scaled * bn_scale_in;
                        float y = bn_s * fv + bn_b;
                        int32_t bv = (int32_t)roundf(y / bn_scale_out);
                        if (bv < bn_activation_min) bv = bn_activation_min;
                        if (bv > bn_activation_max) bv = bn_activation_max;
                        float fbv = (float)(int8_t)bv * silu_scale_in;
                        float sy = fbv / (1.0f + expf(-fbv));
                        int32_t v = (int32_t)roundf(sy / silu_scale_out);
                        if (v < silu_activation_min) v = silu_activation_min;
                        if (v > silu_activation_max) v = silu_activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)v;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    /* Transpose input NCHW -> NHWC into ws_input. */
    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    /* Q31 config: fold (mult, shift) -> single Q0.31 scale for HW mvout. */
    int32_t scale_q31 = conv_output_shift == 0
        ? conv_output_multiplier
        : (int32_t)(((int64_t)conv_output_multiplier + ((int64_t)1 << (conv_output_shift - 1))) >> conv_output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;
#else
    float scale = ldexpf((float)conv_output_multiplier, -(31 + conv_output_shift));
#endif

    asm volatile("fence" ::: "memory");

    /* HW im2col + GEMM + Q0.31 requantize -> int8 conv output (NHWC).
     * NO_ACTIVATION: the conv stage keeps the full int8 range; ReLU/SiLU
     * live in the epilogue below. */
    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, weight, bias, ws_output,
        NO_ACTIVATION, scale,
        0, 0, 0,
        WS
    );

    gemmini_fence();
    gemmini_flush(0);

    /* Epilogue: BN per-channel affine + SiLU on the int8 conv output,
     * NHWC (ws_output) -> NCHW (output). */
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float bn_s = bn_scale[oc], bn_b = bn_bias[oc];
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int8_t conv_int8 = ws_output[((n*OH + oh)*OW + ow)*OC + oc];
                    float fv = (float)conv_int8 * bn_scale_in;
                    float y = bn_s * fv + bn_b;
                    int32_t bv = (int32_t)roundf(y / bn_scale_out);
                    if (bv < bn_activation_min) bv = bn_activation_min;
                    if (bv > bn_activation_max) bv = bn_activation_max;
                    float fbv = (float)(int8_t)bv * silu_scale_in;
                    float sy = fbv / (1.0f + expf(-fbv));
                    int32_t v = (int32_t)roundf(sy / silu_scale_out);
                    if (v < silu_activation_min) v = silu_activation_min;
                    if (v > silu_activation_max) v = silu_activation_max;
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)v;
                }
            }
        }
    }
}
