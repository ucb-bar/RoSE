/* source: curated */
/* algorithm: rvv_igemm */
/* weight_layout: ihwoc  (same pre-pack as rvv_oc_blocked / rvv_vsmul_vnclip:
 *  weight[((ic*KH+kh)*KW+kw)*OC+oc], OC innermost/contiguous.) */
/*
 * conv2d_s8 as implicit GEMM (XNNPACK igemm style), adapted to this SoC's
 * hardware constraints and to ModelBlaster's NCHW/planar tensor layout.
 *
 * WHY (diagnosis this file is trying to fix)
 * -------------------------------------------
 * rvv_vsmul_vnclip / rvv_oc_blocked already vectorize over OC (the natural
 * choice: weight is OC-contiguous, so an OC-wide vle8+vwmacc is the MAC),
 * and already register-tile 4 output pixels per weight load (1.64-1.88x
 * win, because that amortizes the weight vle8+vwadd across pixels). What
 * neither does is amortize the PADDING/BOUNDS BOOKKEEPING across the
 * reduction. Their loop nest is:
 *
 *   oc_base loop:
 *     for ic in IC:                      <-- redone per oc_base slice
 *       for kh in KH:                    <-- redone per ic (IC-fold waste)
 *         ih = oh*SH-PH+kh; row_in = ...; iw0 = ...; iw_last = ...
 *         if (row_in && in-bounds) { for kw: MAC }
 *         else                     { for kw: per-pixel-masked MAC }
 *
 * `ih`, `row_in`, `iw0`, `iw_last` do not depend on `ic` at all, yet they
 * are recomputed on every one of the IC iterations, and the whole thing is
 * redone again for every oc_base slice of the OC cache-blocking loop (up
 * to OC/vl times). For a 1x1 conv (KH=KW=1, dronet has one) that bookkeeping
 * IS the entire non-MAC cost of the tap. This is exactly the
 * "per-reduction-index overhead ... address arithmetic and a branch for
 * every vwmacc" diagnosis: the address/branch work already happens at
 * per-K-step granularity (well, per-(ic,kh) granularity after tiling), but
 * K = IC*KH*KW is not the loop the bookkeeping is over.
 *
 * THE FIX: an indirection buffer (XNNPACK's own trick), adapted to NCHW.
 * -------------------------------------------------------------------
 * XNNPACK's igemm builds one pointer per (output pixel, kernel tap) into
 * NHWC input, where a fixed pixel's IC channel values are contiguous, so
 * the K-loop over IC is a `*a0++` walk. Our input is NCHW (channel-planar):
 * a fixed pixel's channels are IH*IW bytes apart, not 1 byte apart. That
 * is fine for a *scalar* load (this kernel never turns it into a vector
 * strided load -- see the "measured and refuted" note below) -- it is
 * exactly as costly as a `*a0++` walk, just with a runtime stride instead
 * of 1. So the same trick applies with an added per-tap step size:
 *
 *   for a tile of MR=4 (then 2, then 1) consecutive output columns, build
 *   ONCE (before the oc_base loop, i.e. shared across the WHOLE OC sweep,
 *   not just the WHOLE IC reduction):
 *     aptr[pixel][tap]  = &input[n, 0, ih(tap), iw(tap,pixel)]   (ic=0 base)
 *                         or &mb_igemm_zero if that tap is out of bounds
 *     astep[pixel][tap] = IH*IW   (real data: walk to the next channel)
 *                         or 0    (padding: keep re-reading the same zero
 *                                  byte every ic step -- branch-free, and
 *                                  still contributes input_offset*(w+
 *                                  filter_offset) per ic, matching the
 *                                  conv2d_s8 spec's OOB semantics exactly)
 *
 *   Then the reduction becomes, per tap, per ic:
 *     vw8  = vle8(weight_tap_base + ic*(KH*KW*OC), vl)   // still OC-contig
 *     vw16 = vwadd_vx(vw8, filter_offset, vl)
 *     a[j] = vwmacc_vx(a[j], *p[j] + input_offset, vw16, vl)   for j in MR
 *     p[j] += astep[j][tap]                                    for j in MR
 *
 *   No branch, no ih/row_in/iw0 recompute, anywhere in that loop. The
 *   bookkeeping drops from O(TILE_OC/vl * IC * KH) occurrences per output
 *   tile down to O(MR * KH*KW) occurrences -- an IC-fold (up to 512x) and
 *   oc_base-fold (up to 8x, OC/vl) reduction, done ONCE per (n,oh,ow_tile)
 *   and reused by every oc_base slice.
 *
 *   The indirection buffer itself is a small stack array (<= MB_CONV_TILE *
 *   KH*KW pointers, e.g. 4*9=36 for a 3x3 -- nowhere near a materialised
 *   im2col buffer, which would be OH*OW*KH*KW*IC bytes). This matches
 *   XNNPACK's own microkernel, which also re-walks the SAME `a[]` pointer
 *   array once per nc-tile (rewinding `a -= ks`) rather than caching input
 *   data across the N/OC sweep -- redundant A-side re-reads across OC
 *   tiles is standard GEMM microkernel practice, not a defect; L1D makes
 *   the re-read cheap. This file does the same (rebuild-not-cache across
 *   oc_base is fine; what's fixed is the address ARITHMETIC, not the data
 *   re-reads).
 *
 * MEASURED AND REFUTED -- do not repeat:
 *   - This file never turns the input access into a *vector* strided load
 *     (no vlse8 on input). Every K-step input read is one scalar load per
 *     accumulator row (exactly as many scalar loads as vsmul_vnclip does
 *     today), because Saturn serialises strided vector loads to ~1
 *     elem/cycle and OW-vectorization already measured 1.7x slower for
 *     exactly that reason.
 *   - oc_blocked (this file's OC slab strategy, unchanged) is kept because
 *     dropping it was 1.66-1.81x slower previously.
 *
 * CORRECTNESS: all pointer/index arithmetic that scales with N, IC, IH, IW
 * or OC is size_t/ptrdiff_t (see rule: 32-bit index arithmetic wraps for
 * BSS buffers above 0x80000000 -- this kernel does MORE pointer arithmetic
 * than its predecessors, via the running `p[j] += astep` walk, so this is
 * enforced even more carefully here than in oc_blocked).
 *
 * mr x nr tile: mr=4 (register-tiled output pixels, cascading 4 -> 2 -> 1
 * for OW remainders) x nr=vlmax_e32m4 (32 output channels at VLEN=256,
 * LMUL=4). mr=4 matches fmaPipeDepth=4 -- four independent i32m4
 * accumulators (16 of 32 vector registers) keep the MAC pipeline full
 * without register pressure spilling the weight/temp registers.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_CONV_TILE
#define MB_CONV_TILE 4
#endif

static inline vint8m1_t mb_igemm_requant_i32m4(vint32m4_t vacc,
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

#define MB_STORE_PIX(ACC, OFF)                                                \
    do {                                                                      \
        vint8m1_t _v = mb_igemm_requant_i32m4((ACC), output_multiplier,       \
            output_shift, output_offset, activation_min, activation_max, vl); \
        __riscv_vsse8_v_i8m1((op) + (OFF), st, _v, vl);                       \
    } while (0)

/* Build the indirection buffer for `mr` consecutive output columns starting
 * at (oh, ow0), for one batch element (in_n0 = input + n*IC*IH*IW, i.e. the
 * ic=0 base of that batch's channel-plane stack). One (pointer, ic-step)
 * pair per (pixel, tap); a tap out of [0,IH)x[0,IW) bounds for a pixel gets
 * the zero sentinel with step=0, so the caller's ic-loop reads the same
 * zero byte every step -- no per-element branch, and it still contributes
 * input_offset*(w+filter_offset) per ic exactly as the conv2d_s8 spec
 * requires for out-of-bounds taps.
 *
 * aptr/astep are flat [mr*taps] arrays, row-major as [pixel][tap] with row
 * stride `taps` (taps = KH*KW, constant for the whole kernel call, so the
 * caller can size/index a single buffer for every mr tier).
 */
static inline void mb_igemm_build_indir(
    const int8_t *restrict in_n0, int IH, int IW,
    int oh, int ow0, int SH, int SW, int PH, int PW, int KH, int KW,
    int mr, int taps,
    const int8_t **restrict aptr, ptrdiff_t *restrict astep)
{
    static const int8_t mb_igemm_zero = 0;
    const ptrdiff_t ic_stride = (ptrdiff_t)IH * (ptrdiff_t)IW;

    for (int j = 0; j < mr; j++) {
        const int ow_j = ow0 + j;
        int t = 0;
        for (int kh = 0; kh < KH; kh++) {
            const int ih = oh * SH - PH + kh;
            const int row_ok = (ih >= 0 && ih < IH);
            for (int kw = 0; kw < KW; kw++, t++) {
                const int iw = ow_j * SW - PW + kw;
                const size_t idx = (size_t)j * (size_t)taps + (size_t)t;
                if (row_ok && iw >= 0 && iw < IW) {
                    aptr[idx]  = in_n0 + (size_t)ih * (size_t)IW + (size_t)iw;
                    astep[idx] = ic_stride;
                } else {
                    aptr[idx]  = &mb_igemm_zero;
                    astep[idx] = 0;
                }
            }
        }
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
    const int taps = KH * KW;

    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;
    const ptrdiff_t w_ic_stride = (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)OC;

    /* Indirection buffer, sized once for the largest tier (MB_CONV_TILE);
     * smaller tiers just use the first mr*taps entries. */
    const int8_t *aptr_buf[MB_CONV_TILE * (taps > 0 ? taps : 1)];
    ptrdiff_t     astep_buf[MB_CONV_TILE * (taps > 0 ? taps : 1)];

    /* OC cache blocking, unchanged from rvv_oc_blocked / rvv_vsmul_vnclip:
     * keep a TILE_OC slab of weights resident in L1D across the whole
     * spatial sweep instead of walking the entire weight tensor per
     * output position. */
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
        const int8_t *in_n0 = input + (size_t)n * (size_t)IC * (size_t)IH * (size_t)IW;

        for (int oh = 0; oh < OH; oh++) {
            int ow = 0;

            /* ---- tiled strip: MB_CONV_TILE pixels share the indirection
             * buffer AND every weight load across the whole oc_base sweep. */
            for (; ow + MB_CONV_TILE <= OW; ow += MB_CONV_TILE) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, MB_CONV_TILE, taps,
                                      aptr_buf, astep_buf);

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

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[0*(size_t)taps + t];
                        const int8_t *p1 = aptr_buf[1*(size_t)taps + t];
                        const int8_t *p2 = aptr_buf[2*(size_t)taps + t];
                        const int8_t *p3 = aptr_buf[3*(size_t)taps + t];
                        const ptrdiff_t s0 = astep_buf[0*(size_t)taps + t];
                        const ptrdiff_t s1 = astep_buf[1*(size_t)taps + t];
                        const ptrdiff_t s2 = astep_buf[2*(size_t)taps + t];
                        const ptrdiff_t s3 = astep_buf[3*(size_t)taps + t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);

                            a0 = __riscv_vwmacc_vx_i32m4(a0,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            a1 = __riscv_vwmacc_vx_i32m4(a1,
                                (int16_t)((int32_t)(*p1) + input_offset), vw16, vl);
                            a2 = __riscv_vwmacc_vx_i32m4(a2,
                                (int16_t)((int32_t)(*p2) + input_offset), vw16, vl);
                            a3 = __riscv_vwmacc_vx_i32m4(a3,
                                (int16_t)((int32_t)(*p3) + input_offset), vw16, vl);

                            p0 += s0; p1 += s1; p2 += s2; p3 += s3;
                            wq += w_ic_stride;
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

            /* ---- half-width strip: 2 pixels. Same indirection scheme,
             * mr=2. Reclaims the OW=7-style shapes (dronet conv_modules
             * .4/5/6) that the 4-wide tile leaves 3-of-7 columns idle on. */
            for (; ow + 2 <= OW; ow += 2) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, 2, taps, aptr_buf, astep_buf);

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

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[0*(size_t)taps + t];
                        const int8_t *p1 = aptr_buf[1*(size_t)taps + t];
                        const ptrdiff_t s0 = astep_buf[0*(size_t)taps + t];
                        const ptrdiff_t s1 = astep_buf[1*(size_t)taps + t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);

                            a0 = __riscv_vwmacc_vx_i32m4(a0,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            a1 = __riscv_vwmacc_vx_i32m4(a1,
                                (int16_t)((int32_t)(*p1) + input_offset), vw16, vl);

                            p0 += s0; p1 += s1;
                            wq += w_ic_stride;
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

            /* ---- remainder columns: one pixel at a time, mr=1. Still goes
             * through the same branch-free indirection scheme -- there is
             * no separate "slow path" in this kernel at all, unlike its
             * predecessors, because the padding sentinel makes the fast
             * and slow cases the same code. */
            for (; ow < OW; ow++) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, 1, taps, aptr_buf, astep_buf);

                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));
                    vint32m4_t vacc;
                    if (bias != NULL) vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else              vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[t];
                        const ptrdiff_t s0 = astep_buf[t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            p0 += s0;
                            wq += w_ic_stride;
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
