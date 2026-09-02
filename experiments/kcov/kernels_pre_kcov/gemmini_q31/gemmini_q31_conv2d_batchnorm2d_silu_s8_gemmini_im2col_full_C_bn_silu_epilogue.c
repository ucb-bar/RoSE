/* source: curated */
/* algorithm: gemmini_im2col_full_C_bn_silu_epilogue */
/* accuracy_class: bit_exact */
/* origin: fused Conv2d+BatchNorm+SiLU for gemmini_q31 (yolov8_nano
 *         backbone). Matmul stays on the systolic array: im2col (CPU) →
 *         tiled_matmul_auto(full_C=true) → raw int32 accumulator, then a
 *         SCALAR Q0.31 requantize (bit-exact, like
 *         gemmini_conv2d_s8_gemmini_im2col_full_C.c) followed by the
 *         BatchNorm per-channel affine and the SiLU activation, all applied
 *         IN THE SAME requantize loop over the int8 conv output. Keeps the
 *         heavy MAC on Gemmini (millions of cycles vs the ~hundreds of
 *         millions of the scalar reference fallback) while folding the
 *         bn/silu glue onto the same hart — removing the conv→bn and
 *         bn→silu cross-core ping-pong.
 *
 *         Numerics: bit-exact with conv2d_s8 → batchnorm2d_s8 → silu_s8.
 *         Each stage requantizes to int8 exactly as the unfused chain:
 *           conv_int8 = clamp(Q0.31(acc), conv_act_min, conv_act_max)
 *           fv  = conv_int8 * bn_scale_in
 *           y   = bn_scale[oc] * fv + bn_bias[oc]
 *           bn_int8 = clamp(round(y / bn_scale_out), bn_act_min, bn_act_max)
 *           fbv = bn_int8 * silu_scale_in
 *           sy  = fbv / (1 + exp(-fbv))
 *           out = clamp(round(sy / silu_scale_out), silu_act_min, silu_act_max)
 *
 *         WEIGHT LAYOUT CONTRACT: `weight` is pre-packed flat HWIO
 *         ([KH*KW*IC, OC]) by generate_skeleton.py::_backend_pack_weight. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

enum {
    CBNS_WS_BYTES       = 512 * 1024,
    CBNS_IM2COL_ELEMS   = DIM * 256 * 9,
    CBNS_ACC_ELEMS      = DIM * 256,
};

static inline int8_t mb_conv_bn_silu_epilogue(
        int32_t acc, float bn_s, float bn_b,
        int conv_output_offset, int conv_output_multiplier, int conv_output_shift,
        int conv_activation_min, int conv_activation_max,
        float bn_scale_in, float bn_scale_out,
        int bn_activation_min, int bn_activation_max,
        float silu_scale_in, float silu_scale_out,
        int silu_activation_min, int silu_activation_max)
{
    int64_t prod = (int64_t)acc * (int64_t)conv_output_multiplier;
    prod = (prod + ((int64_t)1 << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (conv_output_shift > 0) {
        scaled = (int32_t)(((int64_t)scaled
            + ((int64_t)1 << (conv_output_shift - 1))) >> conv_output_shift);
    } else if (conv_output_shift < 0) {
        scaled <<= (-conv_output_shift);
    }
    scaled += conv_output_offset;
    if (scaled < conv_activation_min) scaled = conv_activation_min;
    if (scaled > conv_activation_max) scaled = conv_activation_max;
    int8_t conv_int8 = (int8_t)scaled;
    /* BN affine on the int8 conv output. */
    float fv = (float)conv_int8 * bn_scale_in;
    float y = bn_s * fv + bn_b;
    int32_t bv = (int32_t)roundf(y / bn_scale_out);
    if (bv < bn_activation_min) bv = bn_activation_min;
    if (bv > bn_activation_max) bv = bn_activation_max;
    int8_t bn_int8 = (int8_t)bv;
    /* SiLU on the int8 BN output. */
    float fbv = (float)bn_int8 * silu_scale_in;
    float sy = fbv / (1.0f + expf(-fbv));
    int32_t v = (int32_t)roundf(sy / silu_scale_out);
    if (v < silu_activation_min) v = silu_activation_min;
    if (v > silu_activation_max) v = silu_activation_max;
    return (int8_t)v;
}

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
    static elem_t ws_input  [CBNS_WS_BYTES]     __attribute__((aligned(64)));
    static elem_t ws_im2col [CBNS_IM2COL_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[CBNS_ACC_ELEMS]    __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    if (input_offset != 0 || filter_offset != 0
            || (size_t)(N * IH * IW * IC) > CBNS_WS_BYTES
            || (size_t)(K_inner * OC)      > CBNS_WS_BYTES
            || (size_t)(N * OH * OW * OC)  > CBNS_WS_BYTES
            || K_inner * DIM               > CBNS_IM2COL_ELEMS
            || OC * DIM                    > CBNS_ACC_ELEMS) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                float bn_s = bn_scale[oc];
                float bn_b = bn_bias[oc];
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
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                   + filter_offset);
                                }
                            }
                        }
                        output[((n*OC+oc)*OH+oh)*OW+ow] = mb_conv_bn_silu_epilogue(
                            acc, bn_s, bn_b,
                            conv_output_offset, conv_output_multiplier, conv_output_shift,
                            conv_activation_min, conv_activation_max,
                            bn_scale_in, bn_scale_out,
                            bn_activation_min, bn_activation_max,
                            silu_scale_in, silu_scale_out,
                            silu_activation_min, silu_activation_max);
                    }
                }
            }
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

    asm volatile("fence" ::: "memory");

    for (int tile_i = 0; tile_i < total_out; tile_i += DIM) {
        int tile_rows = total_out - tile_i < DIM ? total_out - tile_i : DIM;

        for (int i = 0; i < DIM; i++) {
            elem_t *row = &ws_im2col[i * K_inner];
            if (i >= tile_rows) {
                for (int k = 0; k < K_inner; k++) row[k] = 0;
                continue;
            }
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh_idx * SH - PH + kh;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow_idx * SW - PW + kw;
                    elem_t *cell = row + (kh * KW + kw) * IC;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        const elem_t *src = &ws_input[((n_idx*IH + ih)*IW + iw)*IC];
                        for (int c = 0; c < IC; c++) cell[c] = src[c];
                    } else {
                        for (int c = 0; c < IC; c++) cell[c] = 0;
                    }
                }
            }
        }

        asm volatile("fence" ::: "memory");

        tiled_matmul_auto(
            DIM, OC, K_inner,
            ws_im2col, weight,
            (const void *)bias, (void *)ws_acc_out,
            K_inner, OC, OC, OC,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
            bias != NULL,
            false, false,
            true, false,
            0, WS
        );

        gemmini_fence();
        gemmini_flush(0);

        for (int i = 0; i < tile_rows; i++) {
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            for (int oc = 0; oc < OC; oc++) {
                int32_t acc = ws_acc_out[i * OC + oc];
                output[((n_idx*OC + oc)*OH + oh_idx)*OW + ow_idx] =
                    mb_conv_bn_silu_epilogue(
                        acc, bn_scale[oc], bn_bias[oc],
                        conv_output_offset, conv_output_multiplier, conv_output_shift,
                        conv_activation_min, conv_activation_max,
                        bn_scale_in, bn_scale_out,
                        bn_activation_min, bn_activation_max,
                        silu_scale_in, silu_scale_out,
                        silu_activation_min, silu_activation_max);
            }
        }
    }
}
