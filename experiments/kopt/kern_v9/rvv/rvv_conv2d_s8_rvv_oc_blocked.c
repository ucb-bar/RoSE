/* source: curated */
/* algorithm: rvv_oc_blocked */
/* origin: rvv_oc_blocked (OC cache blocking + size_t-hoisted indexing)
 *         WITH output-pixel register tiling cascaded 4 -> 2 -> 1.
 *
 * Why this file exists
 * --------------------
 * Two separate results forced this combination.
 *
 * 1. Tiling is the big win. Amortizing one weight vle8+vwadd over several
 *    output pixels took conv2d_s8 1.64x on dronet and 1.70x on fused_full,
 *    because conv was bound by per-reduction-index scalar overhead, not by
 *    vector width. (Vectorizing over OW instead was 1.7x SLOWER -- it makes
 *    the input the vector operand, and at SW=2 that is a vlse8 gather which
 *    this vector unit serialises.)
 *
 * 2. But tiling was built on rvv_vsmul_vnclip, and that kernel is WRONG on
 *    yolov8n: measured on FPGA, oc_blocked -> err=0, untiled vsmul_vnclip
 *    -> err=3, tiled vsmul_vnclip -> Store/AMO access fault. All three are
 *    bit-exact on spike, which cannot see it. The defect predates all
 *    tiling work.
 *
 * The two kernels have byte-identical MAC and requantize sequences. They
 * differ in exactly three things, and this file takes oc_blocked's side of
 * all three rather than guessing which one matters:
 *   - the oc_outer L1D blocking loop;
 *   - the size_t-hoisted row offset. vsmul_vnclip indexes the input as
 *     input[((n*IC+ic)*IH+ih)*IW+iw], entirely in int. oc_blocked hoists it
 *     to size_t with the comment that 32-bit index arithmetic wraps when a
 *     BSS-placed buffer's low 32 bits cross the int32 sign boundary --
 *     buffers live above 0x80000000, so their low word is negative as an
 *     int32. That is layout-dependent, which is exactly why the symptom
 *     moves between err=1 / err=3 / fault as the binary shifts;
 *   - the store form (V8 took oc_blocked's _obuf round-trip; V9 below
 *     reinstates vsse8 now that the index arithmetic is proven to be the
 *     actual defect).
 *
 * Padding semantics are preserved: an out-of-bounds tap is NOT skipped, it
 * contributes input_offset*(w+filter_offset), because the quantized pad
 * value 0 still carries the input zero point. Only the all-in-bounds fast
 * path elides the per-tap checks.
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

/* Store one output pixel's vl channels.
 *
 * V9: the vsse8 strided store is reinstated. Output is NCHW so a pixel's vl
 * channels sit OH*OW apart; the _obuf form spilled the vector to the stack
 * and then ran vl scalar load/stores (on dronet conv_modules.0 that is
 * 3136 px x 32 ch = 100k pairs). vsse8 expresses the same access pattern as
 * one instruction and measured 5.8% on dronet conv.
 *
 * The V8 port dropped it only to change one variable at a time while
 * chasing correctness. The root cause turned out to be 32-bit index
 * arithmetic wrapping for BSS buffers above 0x80000000 -- the STORE FORM was
 * never implicated, only the index arithmetic, which oc_blocked's size_t
 * hoisting (kept here) fixes. So vsse8 comes back and is re-measured. */
#define MB_STORE_PIX(ACC, OFF)                                                \
    do {                                                                      \
        vint8m1_t _v = mb_requant_i32m4((ACC), output_multiplier,             \
            output_shift, output_offset, activation_min, activation_max, vl); \
        __riscv_vsse8_v_i8m1((op) + (OFF), st, _v, vl);                       \
    } while (0)

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

    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;

    /* OC cache blocking, lifted from rvv_oc_blocked: keep a TILE_OC slab of
     * weights resident in L1D across the whole spatial sweep instead of
     * walking the entire weight tensor per output position. */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const int oc_slab_bytes = (int)oc_stride;
    int TILE_OC;
    if (oc_slab_bytes > 0 && oc_slab_bytes <= L1D_OC_BUDGET_BYTES) {
        TILE_OC = L1D_OC_BUDGET_BYTES / oc_slab_bytes;
        if (TILE_OC > vlmax_oc) TILE_OC = (TILE_OC / vlmax_oc) * vlmax_oc;
        else                    TILE_OC = vlmax_oc;
    } else {
        TILE_OC = vlmax_oc;
    }
    if (TILE_OC > OC) TILE_OC = OC;
    if (TILE_OC <= 0) TILE_OC = OC;

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ow = 0;

            /* ---- tiled strip: MB_CONV_TILE pixels share each weight load ---- */
            for (; ow + MB_CONV_TILE <= OW; ow += MB_CONV_TILE) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

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
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);
                    MB_STORE_PIX(a2, 2);
                    MB_STORE_PIX(a3, 3);

                    oc_base += (int)vl;
                }
            }

            /* ---- half-width strip: 2 pixels per weight load.
             * OW=7 shapes (dronet conv_modules.4/5/6) left 3 of every
             * 7 columns on the untiled path, which is why they gained
             * least from TILE=4 (19-33% of the vwmacc ceiling vs ~50%
             * for OW=56/14). A 2-pixel tier reclaims most of it. */
            for (; ow + 2 <= OW; ow += 2) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

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
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);

                    oc_base += (int)vl;
                }
            }

            /* ---- remainder columns: one pixel at a time ---- */
            for (; ow < OW; ow++) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));
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

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(vacc, 0);
                    oc_base += (int)vl;
                }
            }
        }
    }
    }
}
