/* source: curated */
/* algorithm: rvv_vsmul_vnclip */
/* origin: pure-integer RVV requantize via vsmul (Q0.31 multiply) +
 *         vnclip (shift+saturating narrow); no FP ops, runs on Zve32x.
 *         LMUL=4 gives 2x more OC/iter than the LMUL=2 rvv_widening_oc. */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/*
 * LMUL alignment (element counts identical across all three types):
 *   i32m4 — accumulator    VLMAX = 4*VLEN/32
 *   i16m2 — requant mid    VLMAX = 2*VLEN/16  = same
 *   i8m1  — weights/out    VLMAX = VLEN/8     = same
 *
 * Output is NCHW [N, OC, OH, OW].  OC elements at a fixed (n,oh,ow) are
 * separated by stride OH*OW, so we use a strided store (vsse8).
 */
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));

                    /* Init accumulator from bias or zero. */
                    vint32m4_t vacc;
                    if (bias != NULL)
                        vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else
                        vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    /* Inner reduction over (IC, KH, KW). */
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int8_t in_byte = 0;
                                if (row_in && iw >= 0 && iw < IW)
                                    in_byte = input[((n*IC + ic)*IH + ih)*IW + iw];
                                int32_t in_v = (int32_t)in_byte + input_offset;

                                /* IHWOC: weight[ic][kh][kw][oc] — OC contiguous */
                                const int8_t *wp = weight
                                    + ((size_t)ic * KH * KW + (size_t)kh * KW + kw) * OC
                                    + oc_base;
                                vint8m1_t vw8 = __riscv_vle8_v_i8m1(wp, vl);

                                /* Sign-extend i8m1 → i16m2, fold filter_offset. */
                                vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                    vw8, (int16_t)filter_offset, vl);

                                /* Widening MAC: i32m4 += i16m2 * scalar. */
                                vacc = __riscv_vwmacc_vx_i32m4(
                                    vacc, (int16_t)in_v, vw16, vl);
                            }
                        }
                    }

                    /* ---- Pure-integer requantize (no FP) ----
                     *
                     * vsmul: computes (acc * output_multiplier + 2^30) >> 31
                     * bit-exact to the Q0.31 reference scalar formula.
                     */
                    vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
                        vacc, output_multiplier, __RISCV_VXRM_RNU, vl);

                    /* Saturating narrow i32m4 → i16m2 with output_shift.
                     * vnclip handles the rounding at the discarded bits. */
                    vint16m2_t vout16;
                    if (output_shift < 0) {
                        /* Left-shift then saturating narrow. */
                        vint32m4_t vshifted = __riscv_vsll_vx_i32m4(
                            vscaled, (size_t)(-output_shift), vl);
                        vout16 = __riscv_vnclip_wx_i16m2(
                            vshifted, 0, __RISCV_VXRM_RNU, vl);
                    } else if (output_shift < 32) {
                        /* Normal case: shift fits in log2(SEW)=5 bits. */
                        vout16 = __riscv_vnclip_wx_i16m2(
                            vscaled, (size_t)output_shift, __RISCV_VXRM_RNU, vl);
                    } else {
                        /* output_shift >= 32: vnclip masks shift mod 32, so we
                         * split into vsra(31) + vnclip(shift-31).  The remainder
                         * (shift-31) is in [1,31] for shifts in [32,62], which
                         * covers all values seen in practice.  For extreme shifts
                         * we cap the remainder at 31 (result rounds to 0 anyway). */
                        int sa2 = output_shift - 31;
                        if (sa2 > 31) sa2 = 31;
                        vint32m4_t vscaled2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
                        vout16 = __riscv_vnclip_wx_i16m2(
                            vscaled2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
                    }

                    /* Add zero-point, clamp to activation range. */
                    vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
                    vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
                    vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);

                    /* Truncate i16m2 → i8m1 (already clamped, no saturation needed). */
                    vint8m1_t vout8 = __riscv_vnsra_wx_i8m1(vout16, 0, vl);

                    int8_t *op = output
                        + ((size_t)n * OC + oc_base) * OH * OW
                        + (size_t)oh * OW + ow;
                    int8_t _obuf[256];
                    __riscv_vse8_v_i8m1(_obuf, vout8, vl);
                    for (size_t _vi = 0; _vi < vl; _vi++)
                        op[_vi * (ptrdiff_t)(OH * OW)] = _obuf[_vi];

                    oc_base += (int)vl;
                }
            }
        }
    }
}
