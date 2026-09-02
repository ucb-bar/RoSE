/* source: curated */
/* algorithm: rvv_oc_blocked */
/* accuracy_class: numeric_drift */   /* fp32 accumulation reorder */
/* origin: hand-written RVV port of the conv_transpose2d gather reference.
 *
 *   Transposed conv, gather form: each OUTPUT pixel (oc, oh, ow) collects
 *   the input pixels that scatter onto it. For kernel taps (kh, kw) the
 *   contributing input pixel is
 *       ih = (oh + PH - kh*DH) / SH,   iw = (ow + PW - kw*DW) / SW
 *   and only contributes when those divisions are exact and in range.
 *
 *   Vectorization: across the OUTPUT-CHANNEL dimension (within a group).
 *   For each (n, g, oh, ow) we hold a vl-wide vector of fp32 accumulators,
 *   one per OC lane, and fold each (icg, kh, kw) contribution with a single
 *   vfmacc — the input pixel is a scalar broadcast.
 *
 *   WEIGHTS: the pipeline packs ConvTranspose weight [IC][OCpG][KH][KW] with
 *   perm (1,2,3,0) -> [OCpG][KH][KW][IC], i.e. IC-innermost / OC strided by
 *   KH*KW*IC. Loading the OC slab from THAT directly needs a large-stride
 *   vector load, which the Saturn V256D128 vector unit miscomputes; the old
 *   kernel worked around it by repacking the OC slab into a stack buffer with
 *   a scalar loop INSIDE the (oh,ow,kh,kw,icg) loops — an O(OH*OW*KH*KW*ICpG*OC)
 *   redundant repack that dominated runtime (~0.2 FLOP/cyc, 20-35x behind
 *   XNNPACK). Instead we now transpose the whole weight ONCE at entry into an
 *   OC-innermost scratch buffer  wT[g][icg][kh][kw][ocg]  (O(weights), a one-
 *   time cost), so the inner loop is a plain UNIT-STRIDE OC load exactly like
 *   the fast conv2d kernel. Output is NCHW so the OC-lane store is strided. */

#include <stddef.h>
#include <stdlib.h>
#include <riscv_vector.h>

#define KCONVT2D_VLMAX 256

void kernel_conv_transpose2d(const float *input, const float *weight,
                             const float *bias, float *output,
                             int N, int IC, int IH, int IW,
                             int OC, int OH, int OW,
                             int KH, int KW, int SH, int SW,
                             int PH, int PW, int DH, int DW, int G)
{
    int ICpG = IC / G;
    int OCpG = OC / G;
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * OW * sizeof(float);

    /* --- Pre-transpose weights to OC-innermost, ONCE ---------------------
     * src (IHWOC-packed):  weight[((ocg*KH + kh)*KW + kw)*IC + (g*ICpG+icg)]
     * dst (OC-contiguous): wT[ (((g*ICpG+icg)*KH + kh)*KW + kw)*OCpG + ocg ]
     * so a fixed (g,icg,kh,kw) has its OCpG lanes contiguous -> unit stride. */
    const size_t wT_elems = (size_t)IC * KH * KW * OCpG;   /* = G*ICpG*KH*KW*OCpG */
    float *wT = (float *)malloc(wT_elems * sizeof(float));

    if (wT != NULL) {
        for (int g = 0; g < G; g++) {
            for (int icg = 0; icg < ICpG; icg++) {
                int ic = g * ICpG + icg;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        const float *sp = weight + ((size_t)kh * KW + kw) * IC + ic;
                        float *dp = wT + (((size_t)ic * KH + kh) * KW + kw) * OCpG;
                        for (int ocg = 0; ocg < OCpG; ocg++)
                            dp[ocg] = sp[(size_t)ocg * KH * KW * IC];
                    }
                }
            }
        }
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int ocg = 0;
                    while (ocg < OCpG) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(OCpG - ocg));
                        int oc = g * OCpG + ocg;

                        vfloat32m4_t vacc;
                        if (bias != NULL)
                            vacc = __riscv_vle32_v_f32m4(bias + oc, vl);
                        else
                            vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                        for (int kh = 0; kh < KH; kh++) {
                            int ihs = oh + PH - kh * DH;
                            if (ihs < 0 || (ihs % SH) != 0) continue;
                            int ih = ihs / SH;
                            if (ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iws = ow + PW - kw * DW;
                                if (iws < 0 || (iws % SW) != 0) continue;
                                int iw = iws / SW;
                                if (iw >= IW) continue;
                                for (int icg = 0; icg < ICpG; icg++) {
                                    int ic = g * ICpG + icg;
                                    float in_v = input[
                                        ((size_t)(n * IC + ic) * IH + ih)
                                        * (size_t)IW + iw];
                                    vfloat32m4_t vw;
                                    if (wT != NULL) {
                                        /* fast path: unit-stride OC load */
                                        const float *wp = wT
                                            + (((size_t)ic * KH + kh) * KW + kw)
                                                * OCpG + ocg;
                                        vw = __riscv_vle32_v_f32m4(wp, vl);
                                    } else {
                                        /* fallback (malloc failed): scalar repack
                                         * of the OC-strided slab, correct but slow. */
                                        const float *wp = weight
                                            + ((size_t)(ocg * KH + kh) * KW + kw)
                                                * (size_t)IC + ic;
                                        float wtmp[KCONVT2D_VLMAX];
                                        const ptrdiff_t st = (ptrdiff_t)KH * KW * IC;
                                        for (size_t l = 0; l < vl; l++)
                                            wtmp[l] = wp[(size_t)l * st];
                                        vw = __riscv_vle32_v_f32m4(wtmp, vl);
                                    }
                                    vacc = __riscv_vfmacc_vf_f32m4(
                                        vacc, in_v, vw, vl);
                                }
                            }
                        }

                        float *op = output
                            + ((size_t)(n * OC + oc) * OH + oh)
                                * (size_t)OW + ow;
                        __riscv_vsse32_v_f32m4(op, out_c_stride_bytes, vacc, vl);

                        ocg += (int)vl;
                    }
                }
            }
        }
    }

    if (wT != NULL) free(wT);
}
