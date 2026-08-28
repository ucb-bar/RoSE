/* source: curated */
/* algorithm: oc_blocked */
/* origin: RVV+Zvfh fp16 conv2d, OC-blocked outer loop for input reuse.
 *
 * Companion to rvv_f16_conv2d_f16_widening.c, optimized for the layer
 * shapes where the widening kernel is bandwidth-bound on input reads:
 * 1×1 convolutions where the same input slice across IC is consumed
 * by every (oc) iteration. The widening kernel reloads input for each
 * (oc, oh, ow) tuple — for 1×1 with OC>=32 that's an OC× redundancy
 * in input bandwidth.
 *
 * Algorithm
 * ─────────
 *
 *   for n, oh, ow:                        // outer spatial loop
 *       // ONE pass through the IC dim — accumulate vlmax oc-channels
 *       // per inner iteration into a parallel-OC accumulator vector.
 *       for oc_base in [0, OC) step VL_OC:
 *           VL_OC = min(OC - oc_base, vlmax_e32m4)
 *           vacc <- bias broadcast (or 0)
 *           for kh, kw in window (with padding check):
 *               for ic in [0, IC):                  // scalar over IC
 *                   x      = input[n, ic, ih, iw]   // ONE fp16 element
 *                   x_f32  = (float)x
 *                   v_w    = vlse16 weight[oc_base..+VL_OC, ic, kh, kw]
 *                            stride = IC*KH*KW (per-oc step in OIHW)
 *                   v_w32  = vfwcvt(v_w)
 *                   vacc   = vfmacc.vf vacc, x_f32, v_w32  // SIMD over OC
 *           store v_out = vfncvt(vacc)               // fp32 -> fp16
 *           vse16 output[n, oc_base..+VL_OC, oh, ow], stride = OH*OW
 *
 * Why this helps
 * ──────────────
 *
 * For 1×1 convs (dominant in ViNT's EfficientNet body + MBConv
 * project/expand layers, where IC and OC are 16..1280 but the spatial
 * dims are tiny):
 *
 *   - widening kernel reads input[(IC)] OC times per output pixel.
 *   - oc_blocked reads input[(IC)] ONCE per output pixel; weights
 *     are read once total per output pixel (vs once-per-IC-tap as
 *     before, since the inner SIMD lane is now OC, not IC).
 *
 * On V128 with VL_OC=16 (vlmax_e32m4 = 32 elts but we run at vsew=32),
 * one 1×1 conv2d call sees a 16× reduction in input loads and
 * comparable work. On real cache hierarchies (FireSim L1+L2), the
 * difference is dominated by reduced refills.
 *
 * For 3×3 convs the input slice is shared across kh×kw=9 macc'd taps
 * per output element, so the input-reload reduction is less impactful;
 * the widening kernel may still win on those. The modelblaster picker
 * keeps both registered as algorithm candidates and uses the
 * fastest-verifying option per spec — leave the widening kernel as
 * the queue's first entry so it's tried first; the oc_blocked file
 * here lands as a curated alternative.
 *
 * Numerics
 * ────────
 *
 * Identical fp32 accumulator + final cast-to-fp16, modulo summation
 * order: this kernel sums (ic, kh, kw) in scalar order; widening sums
 * across vector lanes via vfredusum. Both pathways stay in fp32 the
 * whole time, so the worst-case drift at the fp16 cast is ~1 ulp.
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
    const ptrdiff_t w_oc_stride_bytes = (ptrdiff_t)IC * KH * KW * sizeof(_Float16);
    const ptrdiff_t out_oc_stride_bytes = (ptrdiff_t)OH * OW * sizeof(_Float16);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));

                    /* Seed accumulator with bias (widened to fp32). */
                    vfloat32m4_t vacc;
                    if (bias != NULL) {
                        /* Strided fp16 load of bias[oc_base..+vl] is
                         * actually unit-stride. Use the e16 load
                         * then widen. */
                        size_t vl16 = __riscv_vsetvl_e16m2(vl);
                        vfloat16m2_t vb16 = __riscv_vle16_v_f16m2(
                            bias + oc_base, vl16);
                        __riscv_vsetvl_e32m4(vl);
                        vacc = __riscv_vfwcvt_f_f_v_f32m4(vb16, vl);
                    } else {
                        vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    }

                    /* Sweep the kh × kw × ic window. For each (ic, kh, kw)
                     * tap, broadcast the single input scalar and macc-add
                     * vl weight lanes (one per oc in the current block). */
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;
                            for (int ic = 0; ic < IC; ic++) {
                                /* Scalar input element shared across
                                 * all vl oc channels. */
                                _Float16 x = input[((size_t)n * IC + ic) * IH * IW
                                                   + (size_t)ih * IW + iw];
                                float xf = (float)x;
                                /* Vector of vl weights at fixed
                                 * (ic, kh, kw), one per oc in the
                                 * current block. Stride between oc's
                                 * weight is IC*KH*KW elements. */
                                size_t vl16 = __riscv_vsetvl_e16m2(vl);
                                const _Float16 *w_ptr = weight
                                    + (size_t)oc_base * IC * KH * KW
                                    + ((size_t)ic * KH + kh) * KW + kw;
                                vfloat16m2_t vw = __riscv_vlse16_v_f16m2(
                                    w_ptr, w_oc_stride_bytes, vl16);
                                vfloat32m4_t vw32 =
                                    __riscv_vfwcvt_f_f_v_f32m4(vw, vl);
                                /* SIMD MAC: vacc[lane] += xf * vw32[lane]
                                 * for lane in [0, vl). */
                                vacc = __riscv_vfmacc_vf_f32m4(
                                    vacc, xf, vw32, vl);
                            }
                        }
                    }

                    /* Narrow fp32 -> fp16 and store with OC stride. */
                    vfloat16m2_t vout = __riscv_vfncvt_f_f_w_f16m2(vacc, vl);
                    _Float16 *out_p = output
                        + ((size_t)n * OC + oc_base) * OH * OW
                        + (size_t)oh * OW + ow;
                    size_t vl16 = __riscv_vsetvl_e16m2(vl);
                    __riscv_vsse16_v_f16m2(out_p, out_oc_stride_bytes,
                                           vout, vl16);

                    oc_base += (int)vl;
                }
            }
        }
    }
}
