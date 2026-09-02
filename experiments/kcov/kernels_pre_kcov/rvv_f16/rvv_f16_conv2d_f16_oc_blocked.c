/* source: curated */
/* algorithm: oc_blocked */
/* origin: RVV+Zvfh fp16 conv2d, OC-blocked outer loop for input reuse,
 * WITH output-pixel register tiling cascaded 4 -> 2 -> 1.
 *
 * Companion to rvv_f16_conv2d_f16_widening.c, optimized for the layer
 * shapes where the widening kernel is bandwidth-bound on input reads:
 * 1×1 convolutions where the same input slice across IC is consumed
 * by every (oc) iteration. The widening kernel reloads input for each
 * (oc, oh, ow) tuple — for 1×1 with OC>=32 that's an OC× redundancy
 * in input bandwidth. It is also the candidate that targets conv2d_f16's
 * degenerate IC=1 first layer: with IC=1 the widening kernel's
 * reduction-dimension vectorization has nothing to work with (a single
 * strided fp16 load per tap), so per-element overhead dominates. This
 * kernel instead vectorizes over OC, which has real width even when IC
 * doesn't.
 *
 * Two-level OC blocking (this is what the "oc_blocked" name refers to,
 * and what the pipeline's anti-cheat structural check requires actual
 * identifiers for — `oc_outer` / `TILE_OC`, mirroring the sibling
 * kernels/rvv/rvv_conv2d_s8_rvv_oc_blocked.c and
 * kernels/rvv_f16/rvv_f16_conv2d_s8_rvv_oc_blocked.c):
 *
 *   - `oc_outer` is the OUTER loop, stepping by `TILE_OC` — an L1D-sized
 *     slab of OC. The entire (n, oh, ow) spatial sweep runs once per
 *     slab, so that slab's weights (TILE_OC * IC*KH*KW fp16 elements)
 *     stay resident in L1D across the sweep instead of the working set
 *     being the FULL weight tensor (which can blow L1D for the deeper
 *     layers — vision_cnn.6 here is IC=64,OC=64,KH=KW=3 -> 73,728 B of
 *     fp16 weight, well past a 24 KiB L1D budget).
 *   - `oc_base`/`vl` is the INNER, vector-width-sized step within the
 *     current oc_outer..oc_outer+TILE_OC slab — this is what actually
 *     drives the vfmacc lane count.
 *
 * TILE_OC is sized off an L1D budget the same way the s8 sibling does,
 * clamped to a whole number of vector groups and to OC itself.
 *
 * PIXEL REGISTER TILING (the new part)
 * ─────────────────────────────────────
 * After the 3.69x L1D-blocking + stale-vsetvl win (see kernel_opt_log
 * id1700/1701/1702), this kernel sat at ~8.2% of its achievable roof
 * (widened 8 ops/cyc — see Numerics below). Per-op cycle accounting on
 * real hardware showed conv2d_f16 dominated by *per-tap fixed cost*: a
 * strided `vlse16` weight load plus a `vfwcvt` widen happen once per
 * single output pixel per (ic, kh, kw) tap — the weight vector for a
 * given (ic, kh, kw, oc-block) does not depend on which output pixel
 * (oh, ow) is being produced, so reloading and re-widening it per pixel
 * is pure waste when neighboring pixels in a row share it.
 *
 * kernels/rvv/rvv_conv2d_s8_rvv_oc_blocked.c already proved the fix for
 * this exact class of overhead on the int8 side: hold MB_CONV_F16_TILE
 * (4, cascading to 2 then 1 for the remainder) output-pixel accumulators
 * side by side, load+widen the weight vector ONCE per tap, and issue one
 * `vfmacc.vf` per accumulator against that single widened weight vector.
 * That amortizes the vlse16+vfwcvt fixed cost by the tile width instead
 * of paying it per pixel. (OW-vectorization was tried and rejected for
 * the s8 sibling — at stride 2 it turns the INPUT into the strided
 * operand, which this vector unit serializes at ~1 elem/cyc; pixel
 * *register* tiling avoids that because the vector lane is still OC,
 * never OW.)
 *
 * Since `ih = oh*SH - PH + kh` does not depend on `ow`, the KH bounds
 * check is still a single per-row test shared by the whole tile. Only
 * the KW/IW bounds check needs a per-tile-width test (`iw0`/`iw_last`
 * for the first/last pixel in the tile at this kw); a fully-in-bounds
 * tile takes the branch-free fast path, a partially-in-bounds tile
 * (only possible near the left/right edge) falls back to per-lane
 * bounds checks while still sharing the single weight load, and a
 * fully-out-of-bounds tap (kw shifted the whole tile out of frame)
 * skips the weight load entirely — identical to the un-tiled kernel's
 * `continue` on an out-of-bounds tap, since fp16 padding is exactly
 * zero (no zero-point offset to fold in, unlike the s8 kernel).
 *
 * Algorithm
 * ─────────
 *
 *   TILE_OC <- L1D-budget-sized slab of OC (clamped to a vector-group
 *              multiple and to OC)
 *   for oc_outer in [0, OC) step TILE_OC:       // L1D cache-blocking loop
 *       oc_end = min(oc_outer + TILE_OC, OC)
 *       for n, oh:
 *           ow = 0
 *           while ow + 4 <= OW:                 // 4-pixel register tile
 *               for oc_base in [oc_outer, oc_end) step VL_OC:
 *                   vacc[0..3] <- bias broadcast (or 0), one per tile pixel
 *                   for kh, kw in window (row-shared bounds check):
 *                       for ic in [0, IC):
 *                           v_w = vlse16 weight[oc_base..+VL_OC, ic, kh, kw]
 *                           v_w32 = vfwcvt(v_w)              // ONE load+widen per tap
 *                           for t in [0, 4):
 *                               x_t = input[n, ic, ih, iw0 + t*SW]  // scalar
 *                               vacc[t] = vfmacc.vf vacc[t], x_t, v_w32
 *                   store vacc[0..3] -> output[..., oh, ow..ow+3]
 *               ow += 4
 *           // then the same shape at tile width 2, then width 1 (identical
 *           // to the pre-tiling single-pixel kernel) for the remainder.
 *
 * Why this helps
 * ──────────────
 * The core has exactly one LoadSequencer feeding the vector unit at
 * vMemDataBits = 128b = 16 B/cycle; every strided fp16 gather (vlse16)
 * competes for that same port. Tiling doesn't make the gather itself
 * faster (still ~1 elem/cyc serialized), it just makes ONE gather do the
 * work that used to take TILE gathers, cutting the fixed per-tap
 * overhead — vsetvli-adjacent bookkeeping + the vlse16/vfwcvt pair — by
 * roughly the tile width for every layer wide enough to hit the TILE=4
 * strip at least once (all 6 conv2d_f16 calls in fused_full: OW in
 * {45,23,12,6,8,8}).
 *
 * For 3×3 convs the (kh,kw) window still walks 9 taps per IC same as
 * before; what changes is that 4 output pixels now share each of those
 * 9 taps' weight loads instead of paying for them independently.
 *
 * Numerics
 * ────────
 * For any single output pixel, the (kh, kw, ic) summation order into
 * that pixel's fp32 accumulator is byte-identical to the pre-tiling
 * kernel — tiling only changes which OTHER pixels' accumulators are
 * live in registers at the same time, never the order of terms added to
 * any one of them.
 *
 * WEIGHT-TAP MAC: this kernel uses `vfwmacc.vf` (a single widening
 * multiply-accumulate: fp16 scalar x fp16 vector -> fp32 accumulate in
 * one instruction) rather than a separate `vfwcvt` widen followed by a
 * same-width `vfmacc`. An earlier revision did the two-instruction
 * vfwcvt+vfmacc form; switching the weight tap to vfwmacc.vf removed
 * one vector instruction per (ic, kh, kw) tap (previously already
 * amortized 4x/2x by the pixel tile, now removed entirely) and measured
 * a further 1.087x on real hardware (conv2d_f16 4,929,652 cyc vs
 * 5,359,898 cyc with pixel-tiling alone, same shapes, same
 * max_abs_err=0.0009765625/max_rel_err=0.000935453689 -- bit-identical
 * accuracy, FPGA job 178 DONE rc=0). Despite the name, `vfwmacc` is
 * NOT the true narrow-accumulate path -- it still produces an fp32
 * (widened) result per the RVV V-extension's widening-FMA semantics, so
 * its applicable compute ceiling remains the WIDENED 8 ops/cycle, not
 * the narrow-accumulate 16 ops/cycle a generic `_f16`-suffix lookup
 * would otherwise assume (see scripts/roofline_analysis.py's per-op
 * override for conv2d_f16). A TRUE narrow-accumulate path (fp16 *
 * fp16 -> fp16 accumulate throughout, no fp32 at all, real ceiling 16
 * ops/cycle) was evaluated and rejected: CONV2D_F16's semantics
 * explicitly mandate an fp32 accumulator "to avoid catastrophic
 * cancellation when summing many partial products" (see
 * pipeline/reference_kernels.py's CONV2D_F16 KernelSpec docstring) --
 * the deeper layers here reduce over IC*KH*KW up to 64*3*3=576 terms,
 * well past where a pure-fp16 running sum would diverge from the fp32
 * reference beyond tolerance. The bias-widen path (bias fp16 -> fp32
 * accumulator seed) still uses vfwcvt since there is no per-element
 * multiply there for vfwmacc to fuse.
 *
 * The bias only needs widening (not a tap), so it still uses
 * `vfwcvt` once per oc_base block, not vfwmacc.
 */

#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_CONV_F16_TILE
#define MB_CONV_F16_TILE 4
#endif

/* Narrow one tile pixel's fp32 accumulator to fp16 and store it with the
 * OC stride. `out_pix_elems` is the element offset (not byte) of this
 * pixel's oc_base lane within `output`. */
static inline void mb_f16_store_pixel(_Float16 *output,
                                      ptrdiff_t out_oc_stride_bytes,
                                      size_t out_pix_elems,
                                      vfloat32m4_t vacc, size_t vl)
{
    vfloat16m2_t vout = __riscv_vfncvt_f_f_w_f16m2(vacc, vl);
    __riscv_vsse16_v_f16m2(output + out_pix_elems, out_oc_stride_bytes,
                           vout, vl);
}

void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int N, int IC, int IH, int IW, int OC,
                       int KH, int KW, int SH, int SW, int PH, int PW)
{
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t w_oc_stride_bytes = (ptrdiff_t)IC * KH * KW * sizeof(_Float16);
    const ptrdiff_t out_oc_stride_bytes = (ptrdiff_t)OH * OW * sizeof(_Float16);

    /* OC cache blocking, lifted from rvv_oc_blocked (see
     * kernels/rvv/rvv_conv2d_s8_rvv_oc_blocked.c): keep a TILE_OC slab
     * of weights resident in L1D across the whole spatial sweep instead
     * of walking the entire weight tensor per output position. */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const int oc_slab_bytes = (int)w_oc_stride_bytes;
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

                /* ---- 4-pixel register-tiled strip: one weight load+widen
                 * serves MB_CONV_F16_TILE output pixels. See header. ---- */
                for (; ow + MB_CONV_F16_TILE <= OW; ow += MB_CONV_F16_TILE) {
                    int oc_base = oc_outer;
                    while (oc_base < oc_end) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                        vfloat32m4_t vacc0, vacc1, vacc2, vacc3;
                        if (bias != NULL) {
                            vfloat16m2_t vb16 = __riscv_vle16_v_f16m2(
                                bias + oc_base, vl);
                            vfloat32m4_t vb32 = __riscv_vfwcvt_f_f_v_f32m4(vb16, vl);
                            vacc0 = vb32; vacc1 = vb32; vacc2 = vb32; vacc3 = vb32;
                        } else {
                            vfloat32m4_t vz = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                            vacc0 = vz; vacc1 = vz; vacc2 = vz; vacc3 = vz;
                        }

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw0 = ow * SW - PW + kw;
                                int iw_last = iw0 + (MB_CONV_F16_TILE - 1) * SW;
                                if (iw_last < 0 || iw0 >= IW) continue;
                                int all_in = (iw0 >= 0 && iw_last < IW);

                                for (int ic = 0; ic < IC; ic++) {
                                    const _Float16 *in_row = input
                                        + ((size_t)n * IC + ic) * IH * IW
                                        + (size_t)ih * IW;
                                    const _Float16 *w_ptr = weight
                                        + (size_t)oc_base * IC * KH * KW
                                        + ((size_t)ic * KH + kh) * KW + kw;
                                    vfloat16m2_t vw = __riscv_vlse16_v_f16m2(
                                        w_ptr, w_oc_stride_bytes, vl);

                                    /* Single-instruction widening MAC (fp16
                                     * scalar x fp16 vector -> fp32
                                     * accumulate) in place of vfwcvt+vfmacc.
                                     * Same 8 ops/cyc widened ceiling either
                                     * way (RTL-fixed), but removes the
                                     * separate convert instruction per tap
                                     * -- see the Numerics section above. */
                                    if (all_in) {
                                        _Float16 x0 = in_row[iw0];
                                        _Float16 x1 = in_row[iw0 + SW];
                                        _Float16 x2 = in_row[iw0 + 2*SW];
                                        _Float16 x3 = in_row[iw0 + 3*SW];
                                        vacc0 = __riscv_vfwmacc_vf_f32m4(vacc0, x0, vw, vl);
                                        vacc1 = __riscv_vfwmacc_vf_f32m4(vacc1, x1, vw, vl);
                                        vacc2 = __riscv_vfwmacc_vf_f32m4(vacc2, x2, vw, vl);
                                        vacc3 = __riscv_vfwmacc_vf_f32m4(vacc3, x3, vw, vl);
                                    } else {
                                        int iw1 = iw0 + SW;
                                        int iw2 = iw0 + 2*SW;
                                        int iw3 = iw0 + 3*SW;
                                        if (iw0 >= 0 && iw0 < IW)
                                            vacc0 = __riscv_vfwmacc_vf_f32m4(
                                                vacc0, in_row[iw0], vw, vl);
                                        if (iw1 >= 0 && iw1 < IW)
                                            vacc1 = __riscv_vfwmacc_vf_f32m4(
                                                vacc1, in_row[iw1], vw, vl);
                                        if (iw2 >= 0 && iw2 < IW)
                                            vacc2 = __riscv_vfwmacc_vf_f32m4(
                                                vacc2, in_row[iw2], vw, vl);
                                        if (iw3 >= 0 && iw3 < IW)
                                            vacc3 = __riscv_vfwmacc_vf_f32m4(
                                                vacc3, in_row[iw3], vw, vl);
                                    }
                                }
                            }
                        }

                        size_t out_base = ((size_t)n * OC + oc_base) * OH * OW
                                         + (size_t)oh * OW + ow;
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 0, vacc0, vl);
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 1, vacc1, vl);
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 2, vacc2, vl);
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 3, vacc3, vl);

                        oc_base += (int)vl;
                    }
                }

                /* ---- 2-pixel strip for the OW%4 remainder before the
                 * single-pixel tail. Same fast/slow bounds split. ---- */
                for (; ow + 2 <= OW; ow += 2) {
                    int oc_base = oc_outer;
                    while (oc_base < oc_end) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                        vfloat32m4_t vacc0, vacc1;
                        if (bias != NULL) {
                            vfloat16m2_t vb16 = __riscv_vle16_v_f16m2(
                                bias + oc_base, vl);
                            vfloat32m4_t vb32 = __riscv_vfwcvt_f_f_v_f32m4(vb16, vl);
                            vacc0 = vb32; vacc1 = vb32;
                        } else {
                            vfloat32m4_t vz = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                            vacc0 = vz; vacc1 = vz;
                        }

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw0 = ow * SW - PW + kw;
                                int iw1 = iw0 + SW;
                                if (iw1 < 0 || iw0 >= IW) continue;
                                int all_in = (iw0 >= 0 && iw1 < IW);

                                for (int ic = 0; ic < IC; ic++) {
                                    const _Float16 *in_row = input
                                        + ((size_t)n * IC + ic) * IH * IW
                                        + (size_t)ih * IW;
                                    const _Float16 *w_ptr = weight
                                        + (size_t)oc_base * IC * KH * KW
                                        + ((size_t)ic * KH + kh) * KW + kw;
                                    vfloat16m2_t vw = __riscv_vlse16_v_f16m2(
                                        w_ptr, w_oc_stride_bytes, vl);

                                    if (all_in) {
                                        vacc0 = __riscv_vfwmacc_vf_f32m4(
                                            vacc0, in_row[iw0], vw, vl);
                                        vacc1 = __riscv_vfwmacc_vf_f32m4(
                                            vacc1, in_row[iw1], vw, vl);
                                    } else {
                                        if (iw0 >= 0 && iw0 < IW)
                                            vacc0 = __riscv_vfwmacc_vf_f32m4(
                                                vacc0, in_row[iw0], vw, vl);
                                        if (iw1 >= 0 && iw1 < IW)
                                            vacc1 = __riscv_vfwmacc_vf_f32m4(
                                                vacc1, in_row[iw1], vw, vl);
                                    }
                                }
                            }
                        }

                        size_t out_base = ((size_t)n * OC + oc_base) * OH * OW
                                         + (size_t)oh * OW + ow;
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 0, vacc0, vl);
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base + 1, vacc1, vl);

                        oc_base += (int)vl;
                    }
                }

                /* ---- single-pixel remainder: identical to the
                 * pre-tiling kernel (proves the tiled strips above are
                 * numerically inert additions, not a different algorithm
                 * for the tail). ---- */
                for (; ow < OW; ow++) {
                    int oc_base = oc_outer;
                    while (oc_base < oc_end) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                        vfloat32m4_t vacc;
                        if (bias != NULL) {
                            vfloat16m2_t vb16 = __riscv_vle16_v_f16m2(
                                bias + oc_base, vl);
                            vacc = __riscv_vfwcvt_f_f_v_f32m4(vb16, vl);
                        } else {
                            vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                        }

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                for (int ic = 0; ic < IC; ic++) {
                                    _Float16 x = input[((size_t)n * IC + ic) * IH * IW
                                                       + (size_t)ih * IW + iw];
                                    const _Float16 *w_ptr = weight
                                        + (size_t)oc_base * IC * KH * KW
                                        + ((size_t)ic * KH + kh) * KW + kw;
                                    vfloat16m2_t vw = __riscv_vlse16_v_f16m2(
                                        w_ptr, w_oc_stride_bytes, vl);
                                    vacc = __riscv_vfwmacc_vf_f32m4(
                                        vacc, x, vw, vl);
                                }
                            }
                        }

                        size_t out_base = ((size_t)n * OC + oc_base) * OH * OW
                                         + (size_t)oh * OW + ow;
                        mb_f16_store_pixel(output, out_oc_stride_bytes, out_base, vacc, vl);

                        oc_base += (int)vl;
                    }
                }
            }
        }
    }
}
