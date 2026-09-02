/* source: curated */
/* algorithm: rvv_oc_blocked */
/* accuracy_class: numeric_drift */   /* fp32 accumulation reorder */
/* origin: fp32 depthwise conv2d, OC(=channel)-vectorized. Depthwise has
 *         no IC reduction (each output channel reads from its own single
 *         input channel), so vectorizing the channel dimension gives full
 *         lanes with no cross-lane reduction: each lane c owns one
 *         channel's (kh, kw) accumulation.
 *
 *         Layout: input is NCHW so channels are strided by IH*IW
 *         (vlse32). Weight is packed IHWOC — depthwise [C,1,KH,KW] with
 *         IC=1, OC=C becomes [KH][KW][C] with C contiguous, so the per-
 *         (kh,kw) weight load over channels is UNIT stride (vle32).
 *         Output is NCHW, channels strided by OH*OW (vsse32). */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_conv2d_dw(const float *input, const float *weight, const float *bias,
                      float *output,
                      int N, int C, int IH, int IW,
                      int KH, int KW, int SH, int SW, int PH, int PW)
{
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t in_c_stride_bytes  = (ptrdiff_t)IH * IW * sizeof(float);
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * OW * sizeof(float);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int c_base = 0;
                while (c_base < C) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(C - c_base));

                    vfloat32m4_t vacc;
                    if (bias != NULL)
                        vacc = __riscv_vle32_v_f32m4(bias + c_base, vl);
                    else
                        vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;

                            /* input[n, c_base.., ih, iw], channel stride IH*IW. */
                            const float *in_p = input
                                + ((size_t)n * C + c_base) * IH * IW
                                + (size_t)ih * IW + iw;
                            vfloat32m4_t va = __riscv_vlse32_v_f32m4(
                                in_p, in_c_stride_bytes, vl);

                            /* IHWOC weight [KH][KW][C]: (kh*KW+kw)*C + c,
                             * C contiguous -> unit-stride load. */
                            const float *w_p = weight
                                + ((size_t)kh * KW + kw) * C + c_base;
                            vfloat32m4_t vw = __riscv_vle32_v_f32m4(w_p, vl);

                            vacc = __riscv_vfmacc_vv_f32m4(vacc, va, vw, vl);
                        }
                    }

                    float *out_p = output
                        + ((size_t)n * C + c_base) * OH * OW
                        + (size_t)oh * OW + ow;
                    __riscv_vsse32_v_f32m4(out_p, out_c_stride_bytes, vacc, vl);

                    c_base += (int)vl;
                }
            }
        }
    }
}
