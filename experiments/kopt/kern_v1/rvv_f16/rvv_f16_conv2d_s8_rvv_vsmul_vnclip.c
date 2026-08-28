/* source: curated */
/* algorithm: rvv_vsmul_vnclip */
/* origin: pure-integer RVV requantize via vsmul (Q0.31) + vnclip, with
 *         OUTPUT-PIXEL REGISTER TILING over the reduction loop.
 *
 * Why tiling, and not a different vectorization axis:
 *   Measured on FireSim, dronet conv_modules.0 spends ~41 cycles per
 *   reduction tap while the vector work in a tap is only ~14 (vle8 2 +
 *   vwadd 4 + vwmacc 8 at OC=32). The other ~27 are scalar: two index
 *   multiplies, two bounds branches, and a scalar load feeding a
 *   vector-scalar dependency. MAC/cycle is also FLAT in OC across models
 *   (~1.06 at OC=32, 0.84 at OC=256), which rules out vector width as the
 *   binding constraint and points squarely at per-reduction-index overhead.
 *
 *   Tiling amortizes exactly that. For a strip of TILE output pixels
 *   sharing an (oh, oc-block), the weight vector for a tap is IDENTICAL
 *   across the strip, so one vle8 + one vwadd feeds TILE vwmaccs, and the
 *   address math is a pointer walk. Per pixel-tap the fixed cost drops
 *   from (2+4+27) to ((2+4)/TILE + a few), leaving the vwmacc as the
 *   dominant term -- which is the whole point.
 *
 *   An earlier attempt vectorized over OW instead. That was 1.7x SLOWER
 *   (fused_full vision_cnn.0: 1.19M -> 2.02M cycles) because it makes the
 *   INPUT the vector operand, and at SW=2 that is a vlse8 strided load
 *   which this vector unit serialises at ~1 element/cycle. Tiling keeps
 *   every vector load contiguous.
 *
 * Padding semantics are preserved exactly: an out-of-bounds tap is NOT
 * skipped, it contributes input_offset*(w+filter_offset), because the
 * quantized pad value 0 still carries the input zero point. Only the
 * all-in-bounds fast path elides the checks.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_CONV_TILE
#define MB_CONV_TILE 4
#endif

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
        vint32m4_t vsh = __riscv_vsll_vx_i32m4(vscaled, (size_t)(-output_shift), vl);
        vout16 = __riscv_vnclip_wx_i16m2(vsh, 0, __RISCV_VXRM_RNU, vl);
    } else if (output_shift < 32) {
        vout16 = __riscv_vnclip_wx_i16m2(vscaled, (size_t)output_shift,
                                         __RISCV_VXRM_RNU, vl);
    } else {
        int sa2 = output_shift - 31;
        if (sa2 > 31) sa2 = 31;
        vint32m4_t v2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
        vout16 = __riscv_vnclip_wx_i16m2(v2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
    }
    vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
    vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
    vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);
    return __riscv_vnsra_wx_i8m1(vout16, 0, vl);
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

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ow = 0;

            /* ---- tiled strip: MB_CONV_TILE pixels share each weight load ---- */
            for (; ow + MB_CONV_TILE <= OW; ow += MB_CONV_TILE) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));

                    vint32m4_t a0, a1, a2, a3;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb; a2 = vb; a3 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz; a2 = vz; a3 = vz;
                    }

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            int iw_last = iw0 + (MB_CONV_TILE - 1) * SW + (KW - 1);
                            const int8_t *ip = in_ic + (size_t)ih * IW + iw0;

                            if (row_in && iw0 >= 0 && iw_last < IW) {
                                /* all taps of all TILE pixels in bounds */
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    const int8_t *q = ip + kw;
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)q[0]        + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)q[SW]       + input_offset), vw16, vl);
                                    a2 = __riscv_vwmacc_vx_i32m4(a2,
                                        (int16_t)((int32_t)q[2*SW]     + input_offset), vw16, vl);
                                    a3 = __riscv_vwmacc_vx_i32m4(a3,
                                        (int16_t)((int32_t)q[3*SW]     + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            } else {
                                const int8_t *in_row = in_ic + (size_t)ih * IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    int8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                                    if (row_in) {
                                        int w0 = iw0 + kw;
                                        if (w0 >= 0 && w0 < IW)             b0 = in_row[w0];
                                        if (w0+SW >= 0 && w0+SW < IW)       b1 = in_row[w0+SW];
                                        if (w0+2*SW >= 0 && w0+2*SW < IW)   b2 = in_row[w0+2*SW];
                                        if (w0+3*SW >= 0 && w0+3*SW < IW)   b3 = in_row[w0+3*SW];
                                    }
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)b0 + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)b1 + input_offset), vw16, vl);
                                    a2 = __riscv_vwmacc_vx_i32m4(a2,
                                        (int16_t)((int32_t)b2 + input_offset), vw16, vl);
                                    a3 = __riscv_vwmacc_vx_i32m4(a3,
                                        (int16_t)((int32_t)b3 + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            }
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    __riscv_vsse8_v_i8m1(op + 0, st,
                        mb_requant_i32m4(a0, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);
                    __riscv_vsse8_v_i8m1(op + 1, st,
                        mb_requant_i32m4(a1, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);
                    __riscv_vsse8_v_i8m1(op + 2, st,
                        mb_requant_i32m4(a2, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);
                    __riscv_vsse8_v_i8m1(op + 3, st,
                        mb_requant_i32m4(a3, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);

                    oc_base += (int)vl;
                }
            }

            /* ---- half-width strip: 2 pixels per weight load.
             * OW=7 shapes (dronet conv_modules.4/5/6) left 3 of every
             * 7 columns on the untiled path, which is why they gained
             * least from TILE=4 (19-33% of the vwmacc ceiling vs ~50%
             * for OW=56/14). A 2-pixel tier reclaims most of it. */
            for (; ow + 2 <= OW; ow += 2) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));

                    vint32m4_t a0, a1;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz;
                    }

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            int iw_last = iw0 + 1 * SW + (KW - 1);
                            const int8_t *ip = in_ic + (size_t)ih * IW + iw0;

                            if (row_in && iw0 >= 0 && iw_last < IW) {
                                /* all taps of all TILE pixels in bounds */
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    const int8_t *q = ip + kw;
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)q[0]        + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)q[SW]       + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            } else {
                                const int8_t *in_row = in_ic + (size_t)ih * IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    int8_t b0 = 0, b1 = 0;
                                    if (row_in) {
                                        int w0 = iw0 + kw;
                                        if (w0 >= 0 && w0 < IW)             b0 = in_row[w0];
                                        if (w0+SW >= 0 && w0+SW < IW)       b1 = in_row[w0+SW];
                                    }
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)b0 + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)b1 + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            }
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    __riscv_vsse8_v_i8m1(op + 0, st,
                        mb_requant_i32m4(a0, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);
                    __riscv_vsse8_v_i8m1(op + 1, st,
                        mb_requant_i32m4(a1, output_multiplier, output_shift,
                            output_offset, activation_min, activation_max, vl), vl);

                    oc_base += (int)vl;
                }
            }

            /* ---- remainder columns: one pixel at a time ---- */
            for (; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));
                    vint32m4_t vacc;
                    if (bias != NULL) vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else              vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *in_row = in_ic + (size_t)ih * IW;
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = iw0 + kw;
                                int8_t in_byte = 0;
                                if (row_in && iw >= 0 && iw < IW)
                                    in_byte = in_row[iw];
                                vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                    vw8, (int16_t)filter_offset, vl);
                                vacc = __riscv_vwmacc_vx_i32m4(vacc,
                                    (int16_t)((int32_t)in_byte + input_offset),
                                    vw16, vl);
                                wq += OC;
                            }
                        }
                    }

                    vint8m1_t vo = mb_requant_i32m4(vacc, output_multiplier,
                        output_shift, output_offset, activation_min,
                        activation_max, vl);
                    __riscv_vsse8_v_i8m1(
                        output + ((size_t)n * OC + oc_base) * OH * OW
                               + (size_t)oh * OW + ow,
                        (ptrdiff_t)(OH * OW), vo, vl);
                    oc_base += (int)vl;
                }
            }
        }
    }
}
