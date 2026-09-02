/* source: curated */
/* algorithm: rvv_ow_lanes_taps */
/* accuracy_class: bit_exact */
/* origin: hand-written. One vector = a run of OUTPUT COLUMNS of one
 *         (n, c, oh), the same shape as rvv_avgpool2d_s8_rvv_ow_lanes.c.
 *
 *   BIT-EXACT BY CONSTRUCTION, and for a better reason than the transformer
 *   kernels have. Everything up to the store is INTEGER: `acc += iv * wv` in
 *   int32, then a fixed-point requantize through int64. Integer addition is
 *   associative, so accumulating the taps in a different order -- which is
 *   what this does, one tap across many columns instead of many taps down one
 *   column -- gives the identical sum. There is no rounding mode to match and
 *   no summation order to preserve.
 *
 *   THE PADDING PROBLEM, AND WHY THIS DOES NOT MASK. For a fixed (kh, kw) the
 *   input row index `ih = oh*SH - PH + kh` is constant, so its bounds check is
 *   scalar and a whole row is skipped or taken. But `iw = ow*SW - PW + kw`
 *   varies per LANE, and the obvious answer -- build a mask per tap -- costs a
 *   comparison chain per tap per tile.
 *
 *   It is not needed. `iw` is monotonic in `ow`, so the columns for which a
 *   tap is in bounds form a CONTIGUOUS RANGE, computable in closed form:
 *
 *       ow >= ceil((PW - kw) / SW)          and    ow <= (IW-1-kw+PW) / SW
 *
 *   So each tap contributes over a sub-range of the tile, and a sub-range is
 *   addressed by moving the pointer and shortening the vector. No mask, no
 *   per-lane compare, and the arithmetic is identical to the reference's
 *   `if (iw < 0 || iw >= IW) continue;`.
 *
 *   THE ACCUMULATOR LIVES IN MEMORY, not a register, and that is what makes
 *   the range trick work. Different taps cover different sub-ranges of the
 *   same tile, so a register accumulator would need the mask this design
 *   exists to avoid. A 32-lane int32 scratch is 128 bytes, L1-resident, and
 *   the extra load/store per tap is far cheaper than a comparison chain.
 *
 *   WHAT IT IS FOR. `depthwise_conv2d_s8` is one of the ops ViNT needs and no
 *   model in this tree currently emits. It is written and verified against the
 *   reference on the board; it has not been exercised by a real model, because
 *   ViNT cannot be built here (its int8 calibration needs the IDSIA stills and
 *   the IsaacLab forest renders, neither of which is present).
 *
 *   VTYPE. Accumulator at e32m4; the int8 source for that element count is
 *   e8m1 and the fixed-point intermediate is e64m8. Every vsetvl is handed the
 *   ELEMENT COUNT, never a previous vsetvl's result.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Upper bound on lanes per tile, so the scratch can be a stack array.
 * vsetvlmax_e32m4() is 32 at VLEN=256 and 64 at VLEN=512; 256 covers any
 * plausible part. A stack array, NOT a static one: the thread pool runs one
 * worker per hart in a single address space. */
#ifndef MB_DW_MAX_LANES
#define MB_DW_MAX_LANES 256
#endif

void kernel_depthwise_conv2d_s8(const int8_t *input, const int8_t *weight,
                                const int32_t *bias, int8_t *output,
                                int N, int C, int IH, int IW,
                                int KH, int KW, int SH, int SW, int PH, int PW,
                                int input_offset, int filter_offset,
                                int output_offset,
                                int output_multiplier, int output_shift,
                                int activation_min, int activation_max)
{
    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;
    if (OH <= 0 || OW <= 0) return;

    const size_t lanes_max = __riscv_vsetvlmax_e32m4();
    const int tile_max = (int)(lanes_max > MB_DW_MAX_LANES
                               ? (size_t)MB_DW_MAX_LANES : lanes_max);
    int32_t accbuf[MB_DW_MAX_LANES];

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *plane = input + (size_t)(n * C + c) * IH * IW;
            const int8_t *wc = weight + (size_t)c * KH * KW;
            const int32_t b = bias ? bias[c] : 0;
            int8_t *oplane = output + (size_t)(n * C + c) * OH * OW;

            for (int oh = 0; oh < OH; oh++) {
                int ow0 = 0;
                while (ow0 < OW) {
                    const int t = (OW - ow0) < tile_max ? (OW - ow0) : tile_max;

                    /* Seed with the bias, exactly as the reference does. */
                    {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)t);
                        __riscv_vse32_v_i32m4(accbuf,
                            __riscv_vmv_v_x_i32m4(b, vl), vl);
                    }

                    for (int kh = 0; kh < KH; kh++) {
                        const int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;   /* scalar per row */
                        const int8_t *row = plane + (size_t)ih * IW;

                        for (int kw = 0; kw < KW; kw++) {
                            /* The contiguous range of ow for which this tap is
                             * in bounds. Closed form, because iw is monotonic
                             * in ow. */
                            const int num_lo = PW - kw;
                            int lo = (num_lo <= 0) ? 0 : (num_lo + SW - 1) / SW;
                            const int num_hi = IW - 1 - kw + PW;
                            if (num_hi < 0) continue;
                            int hi = num_hi / SW;            /* inclusive */
                            if (lo < ow0) lo = ow0;
                            if (hi > ow0 + t - 1) hi = ow0 + t - 1;
                            if (lo > hi) continue;

                            const int cnt = hi - lo + 1;
                            const int32_t wv =
                                (int32_t)wc[kw + kh * KW] + filter_offset;
                            const int8_t *src = row + (size_t)lo * SW + kw - PW;
                            int32_t *dst = accbuf + (lo - ow0);

                            int k = 0;
                            while (k < cnt) {
                                const size_t n_elem = (size_t)(cnt - k);
                                size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
                                vint8m1_t v8 = (SW == 1)
                                    ? __riscv_vle8_v_i8m1(src + k, vl8)
                                    : __riscv_vlse8_v_i8m1(src + (size_t)k * SW,
                                                           (ptrdiff_t)SW, vl8);
                                size_t vl = __riscv_vsetvl_e32m4(n_elem);
                                vint32m4_t iv =
                                    __riscv_vsext_vf4_i32m4(v8, vl);
                                iv = __riscv_vadd_vx_i32m4(iv, input_offset, vl);
                                vint32m4_t a =
                                    __riscv_vle32_v_i32m4(dst + k, vl);
                                a = __riscv_vmacc_vx_i32m4(a, wv, iv, vl);
                                __riscv_vse32_v_i32m4(dst + k, a, vl);
                                k += (int)vl8;
                            }
                        }
                    }

                    /* ---- requantize, the reference's fixed-point tail ---- */
                    {
                        const size_t vl = __riscv_vsetvl_e32m4((size_t)t);
                        vint32m4_t acc = __riscv_vle32_v_i32m4(accbuf, vl);

                        /* ((int64)acc * mult + (1<<30)) >> 31 */
                        vint64m8_t p = __riscv_vwmul_vx_i64m8(
                            acc, (int32_t)output_multiplier, vl);
                        p = __riscv_vadd_vx_i64m8(p, (int64_t)1 << 30, vl);
                        p = __riscv_vsra_vx_i64m8(p, 31, vl);
                        vint32m4_t prod = __riscv_vncvt_x_x_w_i32m4(p, vl);

                        vint32m4_t v;
                        if (output_shift > 0) {
                            const int32_t r = (int32_t)1 << (output_shift - 1);
                            v = __riscv_vadd_vx_i32m4(prod, r, vl);
                            v = __riscv_vsra_vx_i32m4(v, (size_t)output_shift, vl);
                        } else {
                            v = __riscv_vsll_vx_i32m4(prod, (size_t)(-output_shift), vl);
                        }
                        v = __riscv_vadd_vx_i32m4(v, output_offset, vl);
                        v = __riscv_vmax_vx_i32m4(v, activation_min, vl);
                        v = __riscv_vmin_vx_i32m4(v, activation_max, vl);

                        size_t vl16 = __riscv_vsetvl_e16m2((size_t)t);
                        vint16m2_t v16 = __riscv_vncvt_x_x_w_i16m2(v, vl16);
                        size_t vl8 = __riscv_vsetvl_e8m1((size_t)t);
                        vint8m1_t v8 = __riscv_vncvt_x_x_w_i8m1(v16, vl8);
                        __riscv_vse8_v_i8m1(oplane + (size_t)oh * OW + ow0,
                                            v8, vl8);
                    }

                    ow0 += t;
                }
            }
        }
    }
}
