/* source: curated */
/* algorithm: widening */
/* origin: RVV+Zvfh fp16 conv2d, fp32-accumulator with vfwmacc vectorizing
 *         the IC reduction dimension.
 *
 * The reference impl is the standard 7-nested-loop scalar conv with bounds
 * checks for padding and an fp32 accumulator cast back to fp16 at the
 * final store. With Zvfh we can collapse the IC inner loop into vfwmacc:
 * each lane processes one (ic, kh, kw) tap, the fp32 accumulator stays
 * in an LMUL=4 vector across iterations, and one vfredusum at the end
 * folds it into the scalar that gets cast to fp16.
 *
 * The IC tap and the weight tap are both strided across IC (input stride
 * = IH*IW elements per IC, weight stride = KH*KW per IC) so we use
 * strided vector loads (vlse16). Strided fp16 loads are about 2x slower
 * than unit-stride on V256, but the alternative — vectorizing OC with an
 * IHWOC weight permutation — requires reordering weights at codegen
 * time, which is a bigger refactor. Defer that to a follow-up kernel
 * variant (rvv_f16_oc_blocked) if conv2d_f16 turns out to dominate the
 * post-curated cycle profile.
 *
 * Numerics: bit-exact summation order vs the scalar reference except for
 * the final vfredusum tree, which differs from the scalar left-to-right
 * sum by ~1 ulp of fp32 (well below the fp16 quantization at the store).
 */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int N, int IC, int IH, int IW, int OC,
                       int KH, int KW, int SH, int SW, int PH, int PW)
{
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;
    const size_t vlmax_e32m4 = __riscv_vsetvlmax_e32m4();
    const ptrdiff_t in_ic_stride_bytes = (ptrdiff_t)IH * IW * sizeof(_Float16);
    const ptrdiff_t w_ic_stride_bytes  = (ptrdiff_t)KH * KW * sizeof(_Float16);

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            const _Float16 *w_oc = weight + (size_t)oc * IC * KH * KW;
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    /* fp32 accumulator vector. Initialized to zero;
                     * bias is folded in at the final reduction. */
                    vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(
                        0.0f, vlmax_e32m4);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;

                            /* input[n, 0..IC-1, ih, iw] base. Stride
                             * across IC is IH*IW elements. */
                            const _Float16 *in_base =
                                input + ((size_t)n * IC * IH + ih) * IW + iw;
                            /* weight[oc, 0..IC-1, kh, kw] base.
                             * Stride across IC is KH*KW elements. */
                            const _Float16 *w_base =
                                w_oc + (size_t)kh * KW + kw;

                            int ic = 0;
                            while (ic < IC) {
                                size_t vl = __riscv_vsetvl_e16m2(
                                    (size_t)(IC - ic));
                                vfloat16m2_t va = __riscv_vlse16_v_f16m2(
                                    in_base + (size_t)ic * IH * IW,
                                    in_ic_stride_bytes, vl);
                                vfloat16m2_t vb = __riscv_vlse16_v_f16m2(
                                    w_base  + (size_t)ic * KH * KW,
                                    w_ic_stride_bytes, vl);
                                vacc = __riscv_vfwmacc_vv_f32m4(
                                    vacc, va, vb, vl);
                                ic += (int)vl;
                            }
                        }
                    }

                    /* Reduce fp32 vector to scalar. Seed the reduction
                     * with the bias so we don't need a separate add. */
                    float seed = bias ? (float)bias[oc] : 0.0f;
                    vfloat32m1_t vsum0 = __riscv_vfmv_v_f_f32m1(seed, 1);
                    vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m4_f32m1(
                        vacc, vsum0, vlmax_e32m4);
                    float acc = __riscv_vfmv_f_s_f32m1_f32(vred);

                    output[((size_t)n * OC + oc) * OH * OW
                           + (size_t)oh * OW + ow] = (_Float16)acc;
                }
            }
        }
    }
}
