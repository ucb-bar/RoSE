/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV maxpool2d_s8 with a DEINTERLEAVED load path for
 *         stride-2 pooling (contiguous vle16 + vnsrl, no strided gather).
 *
 * The problem this fixes
 * ---------------------
 * The previous kernel vectorized over OW and fetched each tap with
 * vlse8 -- a stride-SW gather. On dronet's only maxpool (C=32, 56x56 ->
 * 27x27, K=3, S=2) that is 9 gathers of 27 elements per (channel, row),
 * and it measured ~96 cycles per gather+max, i.e. ~3.5 cycles PER ELEMENT.
 * Memory-indexed access on this vector unit is in the same slow family as
 * vluxei (~11.9 cyc/elem measured elsewhere); contiguous loads and
 * register-internal permutes run ~1.7.
 *
 * The fix
 * -------
 * At SW=2, the taps a pooling window needs are exactly the even and odd
 * byte lanes of a CONTIGUOUS run. Load 2*vl bytes as vl 16-bit elements
 * and split them with vnsrl:
 *     vnsrl(v16, 0) -> bytes at even offsets   (taps with even index)
 *     vnsrl(v16, 8) -> bytes at odd offsets    (taps with odd index)
 * so one contiguous vle16 serves TWO taps, and a K=3 window costs two
 * loads instead of three gathers. Nothing is indexed; every load is unit
 * stride and every split is a register-internal shift.
 *
 * Generality
 * ----------
 * The deinterleave path is taken only when SW==2 && DW==1, the whole
 * chunk is in bounds, and the row base is 2-byte aligned (vle16 needs
 * element alignment). Everything else -- other strides, dilation, edge
 * chunks, unaligned rows -- falls through to the original strided/scalar
 * behaviour, which is preserved exactly. Padded positions are still
 * simply not considered, matching -inf padding against the INT8_MIN
 * accumulator init.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    const int deint_shape = (SW == 2 && DW == 1);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_nc = input + (size_t)(n*C + c)*IH*IW;
            int8_t *out_nc = output + (size_t)(n*C + c)*OH*OW;

            for (int oh = 0; oh < OH; oh++) {
                int ow = 0;
                size_t vl;
                for (; ow < OW; ow += vl) {
                    vl = __riscv_vsetvl_e8m4((size_t)(OW - ow));
                    vint8m4_t vacc = __riscv_vmv_v_x_i8m4((int8_t)(-128), vl);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        const int8_t *in_row = in_nc + (size_t)ih*IW;
                        int iw_base = ow*SW - PW;

                        /* ---- fast path: contiguous load + vnsrl split ---- */
                        if (deint_shape
                            && iw_base >= 0
                            && iw_base + (KW - 1) + 2*(int)vl <= IW
                            && ((((uintptr_t)in_row) & 1u) == 0)) {
                            int last_off = -1;
                            vuint16m8_t v16 = __riscv_vmv_v_x_u16m8(0, vl);
                            for (int kw = 0; kw < KW; kw++) {
                                int idx = iw_base + kw;      /* byte index of tap */
                                int odd = idx & 1;
                                int off = idx - odd;         /* even -> 16b aligned */
                                if (off != last_off) {
                                    v16 = __riscv_vle16_v_u16m8(
                                        (const uint16_t *)(const void *)(in_row + off),
                                        vl);
                                    last_off = off;
                                }
                                vuint8m4_t by = __riscv_vnsrl_wx_u8m4(
                                    v16, (size_t)(odd ? 8 : 0), vl);
                                vacc = __riscv_vmax_vv_i8m4(
                                    vacc, __riscv_vreinterpret_v_u8m4_i8m4(by), vl);
                            }
                            continue;
                        }

                        /* ---- general path: strided when fully in bounds ---- */
                        for (int kw = 0; kw < KW; kw++) {
                            int iw0 = iw_base + kw*DW;
                            int iw_last = iw0 + (int)(vl - 1)*SW;
                            if (iw0 >= 0 && iw_last < IW) {
                                vint8m4_t vd = (SW == 1)
                                    ? __riscv_vle8_v_i8m4(in_row + iw0, vl)
                                    : __riscv_vlse8_v_i8m4(in_row + iw0,
                                          (ptrdiff_t)SW, vl);
                                vacc = __riscv_vmax_vv_i8m4(vacc, vd, vl);
                            } else {
                                /* boundary chunk: per-lane, skipping pads */
                                int8_t tmp[256];
                                __riscv_vse8_v_i8m4(tmp, vacc, vl);
                                for (size_t lane = 0; lane < vl; lane++) {
                                    int iw = iw0 + (int)lane*SW;
                                    if (iw >= 0 && iw < IW) {
                                        int8_t v = in_row[iw];
                                        if (v > tmp[lane]) tmp[lane] = v;
                                    }
                                }
                                vacc = __riscv_vle8_v_i8m4(tmp, vl);
                            }
                        }
                    }

                    __riscv_vse8_v_i8m4(out_nc + (size_t)oh*OW + ow, vacc, vl);
                }
            }
        }
    }
}
