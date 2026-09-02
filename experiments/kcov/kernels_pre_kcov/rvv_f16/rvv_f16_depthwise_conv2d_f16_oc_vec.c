/* source: curated */
/* algorithm: oc_vec */
/* origin: RVV+Zvfh depthwise conv2d. Depthwise has no IC reduction (each
 *         output channel reads from exactly one input channel via its
 *         own [1, KH, KW] filter), so vectorizing the IC dim — what we
 *         do for the regular conv2d_f16 — gives a vl of 1. The cheap
 *         win here is to vectorize across OC instead: each lane handles
 *         one channel's (kh, kw) accumulation, and the per-element work
 *         is independent so there's no reduction.
 *
 *         Memory access pattern: input, weight, and output are all
 *         OC-strided in this dispatch (input by IH*IW, weight by KH*KW,
 *         output by OH*OW), so we use vlse16/vsse16 throughout. Strided
 *         loads cost more than unit-stride on V256 but the alternative
 *         requires a NHWC permutation that the codegen doesn't produce.
 *
 *         For EfficientNet's typical depthwise (OC=32..1152, KH=KW=3 or
 *         5, OH*OW from 4 to 32*32): the loop dimensions
 *         (n, oh, ow, kh, kw) are scalar, the inner per-OC compute is
 *         vectorized into ~vlmax_e16m2 lanes per iteration.
 */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_depthwise_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                                 const _Float16 *bias, _Float16 *output,
                                 int N, int IC, int IH, int IW, int OC,
                                 int KH, int KW, int SH, int SW,
                                 int PH, int PW)
{
    (void)IC;   /* depthwise: IC == OC by definition; weight has IC=1 dim. */
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t in_c_stride_bytes  = (ptrdiff_t)IH * IW * sizeof(_Float16);
    const ptrdiff_t w_c_stride_bytes   = (ptrdiff_t)KH * KW * sizeof(_Float16);
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * OW * sizeof(_Float16);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e16m2((size_t)(OC - oc_base));

                    /* Seed the fp32 accumulator with the bias (widened
                     * from fp16) so we don't need a separate add at the
                     * end. NULL bias → zero. */
                    vfloat32m4_t vacc;
                    if (bias != NULL) {
                        vfloat16m2_t vb16 = __riscv_vle16_v_f16m2(
                            bias + oc_base, vl);
                        vacc = __riscv_vfwcvt_f_f_v_f32m4(vb16, vl);
                    } else {
                        vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    }

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;

                            /* input[n, oc_base..oc_base+vl, ih, iw],
                             * stride IH*IW elements per OC. */
                            const _Float16 *in_p = input
                                + ((size_t)n * OC + oc_base) * IH * IW
                                + (size_t)ih * IW + iw;
                            vfloat16m2_t va = __riscv_vlse16_v_f16m2(
                                in_p, in_c_stride_bytes, vl);

                            /* weight[oc_base..oc_base+vl, 0, kh, kw],
                             * stride KH*KW per OC. (depthwise weight
                             * shape is (OC, 1, KH, KW).) */
                            const _Float16 *w_p = weight
                                + (size_t)oc_base * KH * KW
                                + (size_t)kh * KW + kw;
                            vfloat16m2_t vw = __riscv_vlse16_v_f16m2(
                                w_p, w_c_stride_bytes, vl);

                            vacc = __riscv_vfwmacc_vv_f32m4(vacc, va, vw, vl);
                        }
                    }

                    /* Narrow fp32 → fp16 and store with OC stride. */
                    vfloat16m2_t vout = __riscv_vfncvt_f_f_w_f16m2(vacc, vl);
                    _Float16 *out_p = output
                        + ((size_t)n * OC + oc_base) * OH * OW
                        + (size_t)oh * OW + ow;
                    __riscv_vsse16_v_f16m2(out_p, out_c_stride_bytes, vout, vl);

                    oc_base += (int)vl;
                }
            }
        }
    }
}
