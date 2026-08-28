/* source: curated */
/* algorithm: rvv_vsmul_vnclip */
/* origin: pure-integer RVV requantize via vsmul (Q0.31) + vnclip, with a
 *         second, OW-vectorized inner loop for low-IC / wide-spatial shapes.
 *
 * ONE kernel serves every conv2d_s8 instance in a model, so this file keeps
 * the proven OC-blocked path verbatim as the default and only *adds* an
 * alternate reduction order, chosen per-call from the shape.
 *
 * Why a second path at all:
 *   The OC path puts OC on the vector lanes. That is right when OC fills a
 *   register, but the first conv of a vision model is the opposite shape --
 *   tiny OC, tiny reduction depth, big spatial. On fused_full's vision_cnn.0
 *   (IC=1, K=5, OC=16, OW=45) OC=16 fills half of VLMAX=32 at e32m4, so half
 *   of every MAC is thrown away, and the OC path's output store is a SCALAR
 *   loop (output is NCHW, so consecutive OC are OH*OW apart).
 *
 *   Vectorizing over OW instead fixes both: OW lanes are contiguous in NCHW
 *   so the store becomes one vse8, the weight becomes a loop-invariant
 *   scalar, and the per-tap input bounds check amortizes over a whole vector
 *   of output columns instead of being redone per pixel.
 *
 * Arithmetic is UNCHANGED between the two paths: same vwmacc widening MAC
 * (operands swapped -- scalar weight x vector input rather than scalar input
 * x vector weight), same vsmul/vnclip requantize. Both are bit-exact.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Shared requantize tail: i32 accumulator -> saturated, offset, clamped i8.
 * Lifted verbatim out of the original kernel so both paths round identically. */
static inline vint8m1_t mb_requant_i32m4(vint32m4_t vacc,
                                         int output_multiplier, int output_shift,
                                         int output_offset,
                                         int activation_min, int activation_max,
                                         size_t vl)
{
    vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
        vacc, output_multiplier, __RISCV_VXRM_RNU, vl);

    vint16m2_t vout16;
    if (output_shift < 0) {
        vint32m4_t vshifted = __riscv_vsll_vx_i32m4(
            vscaled, (size_t)(-output_shift), vl);
        vout16 = __riscv_vnclip_wx_i16m2(vshifted, 0, __RISCV_VXRM_RNU, vl);
    } else if (output_shift < 32) {
        vout16 = __riscv_vnclip_wx_i16m2(
            vscaled, (size_t)output_shift, __RISCV_VXRM_RNU, vl);
    } else {
        int sa2 = output_shift - 31;
        if (sa2 > 31) sa2 = 31;
        vint32m4_t vscaled2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
        vout16 = __riscv_vnclip_wx_i16m2(vscaled2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
    }

    vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
    vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
    vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);
    return __riscv_vnsra_wx_i8m1(vout16, 0, vl);
}

/* ---- Path A: original OC-blocked, one output pixel (n, oh, ow) ---------- */
static inline void mb_conv_pixel_oc(const int8_t *input, const int8_t *weight,
                                    const int32_t *bias, int8_t *output,
                                    int n, int IC, int IH, int IW, int OC,
                                    int KH, int KW, int SH, int SW, int PH, int PW,
                                    int OH, int OW, int oh, int ow,
                                    int input_offset, int filter_offset,
                                    int output_offset,
                                    int output_multiplier, int output_shift,
                                    int activation_min, int activation_max)
{
    int oc_base = 0;
    while (oc_base < OC) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));

        vint32m4_t vacc;
        if (bias != NULL) vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
        else              vacc = __riscv_vmv_v_x_i32m4(0, vl);

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

                    const int8_t *wp = weight
                        + ((size_t)ic * KH * KW + (size_t)kh * KW + kw) * OC
                        + oc_base;
                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wp, vl);
                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                        vw8, (int16_t)filter_offset, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, (int16_t)in_v, vw16, vl);
                }
            }
        }

        vint8m1_t vout8 = mb_requant_i32m4(vacc, output_multiplier, output_shift,
                                           output_offset, activation_min,
                                           activation_max, vl);
        int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                            + (size_t)oh * OW + ow;
        int8_t _obuf[256];
        __riscv_vse8_v_i8m1(_obuf, vout8, vl);
        for (size_t _vi = 0; _vi < vl; _vi++)
            op[_vi * (ptrdiff_t)(OH * OW)] = _obuf[_vi];

        oc_base += (int)vl;
    }
}

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

    /* Path select. OW-vectorization pays when OC leaves lanes idle and there
     * are enough output columns to fill them instead. MB_OC_FILL_SHIFT picks
     * how underfilled OC has to be before we switch:
     *   1 -> only when OC <= VLMAX/2 (conservative; fused_full vision_cnn.*)
     *   0 -> whenever OC <= VLMAX    (aggressive; also dronet conv_modules.0)
     */
#ifndef MB_OC_FILL_SHIFT
#define MB_OC_FILL_SHIFT 1
#endif
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    int ow_vec = (OC <= (int)(vlmax >> MB_OC_FILL_SHIFT))
              && (OW >= (int)vlmax)
              && (IC * KH * KW <= 64);

    if (!ow_vec) {
        for (int n = 0; n < N; n++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    mb_conv_pixel_oc(input, weight, bias, output,
                                     n, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                                     OH, OW, oh, ow, input_offset, filter_offset,
                                     output_offset, output_multiplier,
                                     output_shift, activation_min, activation_max);
        return;
    }

    /* Interior column range where EVERY tap (kw = 0..KW-1) is in bounds, so
     * the strided input loads need no masking. Edge columns fall back to the
     * proven per-pixel OC path -- they are a few percent of the output. */
    int ow_lo = (PW + SW - 1) / SW;                 /* smallest ow with iw>=0  */
    int ow_hi = (IW + PW - KW) / SW + 1;            /* first ow with iw>=IW    */
    if (ow_lo < 0)  ow_lo = 0;
    if (ow_hi > OW) ow_hi = OW;
    if (ow_hi < ow_lo) ow_hi = ow_lo;

    for (int n = 0; n < N; n++) {
        /* Edge columns, all OC at once. */
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < ow_lo; ow++)
                mb_conv_pixel_oc(input, weight, bias, output,
                                 n, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                                 OH, OW, oh, ow, input_offset, filter_offset,
                                 output_offset, output_multiplier, output_shift,
                                 activation_min, activation_max);
            for (int ow = ow_hi; ow < OW; ow++)
                mb_conv_pixel_oc(input, weight, bias, output,
                                 n, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                                 OH, OW, oh, ow, input_offset, filter_offset,
                                 output_offset, output_multiplier, output_shift,
                                 activation_min, activation_max);
        }

        /* Interior: OW on the lanes, one output channel at a time. */
        for (int oc = 0; oc < OC; oc++) {
            int32_t b = (bias != NULL) ? bias[oc] : 0;
            for (int oh = 0; oh < OH; oh++) {
                int ow = ow_lo;
                while (ow < ow_hi) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(ow_hi - ow));
                    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(b, vl);

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            const int8_t *row =
                                input + ((size_t)(n*IC + ic)*IH + ih)*IW;
                            for (int kw = 0; kw < KW; kw++) {
                                /* iw(j) = (ow + j)*SW - PW + kw, all in bounds. */
                                const int8_t *ip = row + (ow * SW - PW + kw);
                                vint8m1_t vin8 = __riscv_vlse8_v_i8m1(
                                    ip, (ptrdiff_t)SW, vl);
                                vint16m2_t vin16 = __riscv_vwadd_vx_i16m2(
                                    vin8, (int16_t)input_offset, vl);

                                int32_t w = (int32_t)weight[
                                    ((size_t)ic * KH * KW + (size_t)kh * KW + kw)
                                    * OC + oc] + filter_offset;
                                vacc = __riscv_vwmacc_vx_i32m4(
                                    vacc, (int16_t)w, vin16, vl);
                            }
                        }
                    }

                    vint8m1_t vout8 = mb_requant_i32m4(
                        vacc, output_multiplier, output_shift, output_offset,
                        activation_min, activation_max, vl);
                    /* NCHW: consecutive ow are contiguous -> one store. */
                    __riscv_vse8_v_i8m1(
                        output + ((size_t)n * OC + oc) * OH * OW
                               + (size_t)oh * OW + ow,
                        vout8, vl);
                    ow += (int)vl;
                }
            }
        }
    }
}
