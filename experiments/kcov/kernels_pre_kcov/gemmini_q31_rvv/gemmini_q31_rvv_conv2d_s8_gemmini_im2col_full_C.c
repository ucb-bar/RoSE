/* source: curated */
/* algorithm: gemmini_im2col_full_C */
/* accuracy_class: bit_exact */
/* WEIGHT LAYOUT CONTRACT: like the sibling tiled_conv variant, this
 * kernel expects `weight` already in flat HWIO layout
 * ([KH*KW*IC, OC]) — the form tiled_matmul_auto consumes directly.
 * The skeleton emitter (generate_skeleton.py::_backend_pack_weight)
 * permutes OIHW→HWIO at codegen time when --backend gemmini, so
 * we pass weight straight through without a workspace copy. */
/*
 * origin: im2col -> tiled_matmul_auto(full_C=true) -> RVV Q0.31 requantize.
 *
 * NOTE: this file used to be a symlink shared with
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c (the plain,
 * non-Q31 "gemmini" target, which has no `v` in kernel_cflags). It is now
 * a standalone gemmini_q31-only file so it can use RVV intrinsics
 * (kernel_cflags gained rv64imafdc*v* + <riscv_vector.h> for this target
 * specifically, kernel_opt_log.jsonl id 800) without breaking the plain
 * "gemmini" target's build. If you need to change the shared logic (the
 * im2col / tiled_matmul_auto(full_C=true) structure), check both files.
 *
 * WHY THIS KERNEL EXISTS (root-cause of the gemmini_q31 accuracy bug):
 * The sibling gemmini_tiled_conv algorithm asks gemmini's HW mvout unit
 * to fold (output_multiplier, output_shift) into ONE Q0.31 scale
 * (scale_q31 = round(mult / 2^shift)) and does a SINGLE hardware
 * round-shift-by-31. The reference/golden (the scalar fallback below,
 * kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c, and the PyTorch
 * quantized op it was generated from) all compute a TWO-STAGE rounding:
 *   stage1 = round_half_up(acc * mult,   2^31)
 *   stage2 = round_half_up(stage1,       2^shift)   (shift > 0)
 * Pre-folding mult/shift into one scale BEFORE multiplying by acc changes
 * which values land exactly on a rounding boundary (the pre-rounding of
 * the scale is amplified by acc, instead of a fixed +-0.5 in output
 * units), so a single hardware round cannot reproduce the two-stage
 * golden bit-for-bit -- this is a genuine, unavoidable structural
 * mismatch for that algorithm (see gemmini_q31_conv2d_s8_gemmini_tiled_conv.c
 * for the analysis and its own header comment). It is not a bug in that
 * kernel's code, just a HW datapath that cannot represent double
 * rounding.
 *
 * The fix is this algorithm: bypass gemmini's mvout requantize entirely
 * (tiled_matmul_auto(..., full_C=true) drains the RAW int32 accumulator,
 * no scale/round/saturate in hardware) and do the exact two-stage Q0.31
 * requantize on the CPU with vsmul_vx (stage1) + vnclip_wx (stage2), both
 * __RISCV_VXRM_RNU -- bit-identical to the scalar int64 formula and to
 * kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c's mb_requant_i32m4, which
 * is independently verified bit-exact end-to-end elsewhere in this repo.
 * Verified in ISOLATION on dronet (this kernel alone, every other op
 * forced to scalar reference_impl): max_abs_err=0
 * (experiments/kernel_opt_log.jsonl id 1101). The unvectorized ancestor
 * of this file (identical arithmetic, scalar requantize loop) was
 * likewise measured bit-exact on Saturn RTL FireSim, May 2026 -- see
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c.
 *
 * COST: draining a raw int32 accumulator moves 4 bytes/output instead of
 * gemmini_tiled_conv's 1, and needs a CPU-side requantize pass gemmini's
 * float/Q31-scale mvout does inside the accelerator for free
 * (kernel_opt_log id 302/303/304: the unvectorized version of this
 * kernel was ~4-5x slower than gemmini_tiled_conv on dronet, almost
 * entirely the scalar per-element requantize loop). The RVV
 * vectorization here (vsmul/vnclip over the OC dimension, same rounding
 * mode) plus writing straight into `output`'s NCHW layout via a strided
 * vector store (no NHWC ws_output staging buffer, no separate
 * NHWC->NCHW transpose pass -- same fix as id 304) claws back most of
 * that gap; see experiments/kernel_opt_log.jsonl id 1105+ for the
 * measured trade.
 *
 * Handles non-square kernels, any stride/padding, and large
 * output_shift values (int64/RVV requantize, no UB).
 *
 * -------------------------------------------------------------------------
 * RVV im2col + transpose (kernel_opt_log.jsonl id 2400 = baseline split,
 * id 2401 = this change, id 2403 = the FPGA A/B)
 * -------------------------------------------------------------------------
 * MEASURE ON HARDWARE, NOT ON SPIKE.  An rdcycle-bracketed, numerically-
 * inert copy of this kernel (dronet, all 10 conv2d_s8 layers, curated set,
 * max_abs_err=0) was run BOTH ways.  The two profiles disagree so badly
 * they invert the priority order, so the FPGA column is the real one:
 *
 *                        FPGA f2_dual_small_...q31_60mhz    spike
 *   phase                  BEFORE    %     AFTER    %       BEFORE  %
 *   im2col gather        2,039,505 43.5% 1,090,432 34.3%   3,724,056 71.6%
 *   NCHW->NHWC transpose   997,760 21.3%   419,838 13.2%   1,052,263 20.2%
 *   RVV requantize       1,325,917 28.2% 1,345,898 42.3%     359,322  6.9%
 *   Gemmini GEMM + drain   330,538  7.0%   327,435 10.3%      67,575  1.3%
 *   -------------------------------------------------------------------
 *   sum                  4,693,720       3,183,603  1.47x            4.82x
 *
 * Spike bills a vector instruction at roughly one cycle, so it
 * systematically under-weights phases that are ALREADY vectorized and
 * over-weights scalar ones.  Measured end-to-end on hardware (jobs 184 /
 * 185, un-instrumented, both max_abs_err=0): conv2d_s8 aggregate over all
 * 10 layers 4,703,561 -> 3,150,508 = 1.493x, model 5,391,657 -> 3,853,664
 * = 1.399x, every layer moving 1.29x-1.68x.  The same A/B on spike says
 * 4.80x.  Quote the 1.493x.  (kernel_opt_log id 2403 / 2405.)
 *
 * NOTE for anyone carrying over the pure-gemmini_q31 profile: that profile
 * says "requantize 43.6%", but it was taken on the PURE (non-RVV) sibling
 * where the requantize is a scalar int64 two-stage loop.  Here it was
 * ALREADY vsmul/vnclip-vectorized before this change.  Vectorizing "the
 * requantize" on THIS target recovers nothing -- it is untouched below.
 *
 * Two restructurings, both pure data movement (max_abs_err stays 0 by
 * construction -- the same bytes land in the same places):
 *
 *  (1) im2col: the kw loop steps iw by exactly 1 whatever SW is, and both
 *      ws_input (NHWC) and the im2col row are channel-minor, so for a fixed
 *      kh the KW cells the scalar loop copied one CHANNEL at a time are
 *      contiguous on BOTH sides.  The whole in-bounds kw run becomes ONE
 *      vectorized copy of (kw_hi-kw_lo)*IC bytes with the out-of-bounds
 *      prefix/suffix as two zero-fills -- vector length KW*IC instead of
 *      IC (96B not 32B on dronet's 3x3/IC=32 layers), and KW-1 fewer loop
 *      set-ups per kh.  1.87x on hardware.
 *
 *  (2) transpose: for fixed (n,h) this is an IC x IW -> IW x IC transpose,
 *      so exactly one side must be strided.  Pick the orientation with the
 *      LONGER vector (walk w when IW >= IC, else walk c) to amortize the
 *      per-iteration set-up.  2.38x on hardware.
 *
 * WHAT IS NOT RECOVERED, and why (honest scoreboard):
 *  - The requantize is untouched: it was already RVV.  On hardware it is
 *    now the LARGEST phase (42.3%), and it was already the second largest
 *    (28.2%) BEFORE this change -- spike hid that, reporting it at 6.9%.
 *    Its cost is per-output-row overhead, not arithmetic: three integer
 *    div/mods per row to decompose out_idx into (n,oh,ow), plus
 *    ceil(OC/vlmax) short STRIDED vector stores (stride OH*OW).  That is
 *    the next thing to attack in this kernel.
 *  - conv_modules.0 (IC=3, 112x112) gains least (1.48x on hardware) and is
 *    the single biggest conv.  With IC=3 the fused kw run is only KW*IC=9
 *    bytes, so its im2col is loop-overhead-bound, not bandwidth-bound.
 *    Fixing that needs a different im2col shape for tiny-IC layers (gather
 *    a whole tile column-block at a time), not a wider vector.
 *  - Hoisting the per-row div/mod into an incremental wrap-counter was
 *    NOT attempted here: a previous agent measured it as a consistent
 *    small REGRESSION on spike.  Given the spike-vs-hardware inversion
 *    documented above, that measurement cannot distinguish "no win" from
 *    "win only on real hardware" -- spike bills a hardware divide at
 *    roughly one instruction too.  It needs an FPGA A/B.
 */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>
#include <riscv_vector.h>

/*
 * Static workspace limits.  512 KB covers all square conv layers in
 * dronet and yolov8_nano:
 *   WS_BYTES:     max input  = IC=3,IH=160,IW=160 →  75 KB (yolov8 l0)
 *                 max output = IC=16,OH=80,OW=80  → 100 KB (yolov8 l0)
 *                 (ws_weight is gone — weight is pre-packed HWIO at
 *                  codegen time and passed straight to tiled_matmul_auto)
 *   IM2COL_ELEMS: max K_inner = IC=256,K=3×3     → 2304 (yolov8 detect head)
 *   ACC_ELEMS:    max OC      = 256               (yolov8 l7/l8/l9)
 */
enum {
    WS_BYTES       = 512 * 1024,
    IM2COL_ELEMS   = DIM * 256 * 9,   /* DIM rows × max K_inner (IC=256, 3×3) */
    ACC_ELEMS      = DIM * 256,        /* DIM rows × max OC (256 in yolov8)    */
};

/* ---------------------------------------------------------------------------
 * RVV byte-move helpers (kernel_opt_log id 2401).
 *
 * These replace the two CPU-scalar byte loops this kernel used to spend
 * 92% of its cycles in (im2col gather 71.8% + NCHW->NHWC transpose 20.3%;
 * see the phase table in the file header).  Both are pure data movement --
 * byte-for-byte identical results, so max_abs_err stays 0 by construction.
 * ------------------------------------------------------------------------- */

/* Contiguous int8 copy, vectorized. Equivalent to memcpy(dst, src, n). */
static inline void gq31_vcopy_i8(elem_t *dst, const elem_t *src, size_t n)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m8(n);
        __riscv_vse8_v_i8m8(dst, __riscv_vle8_v_i8m8(src, vl), vl);
        dst += vl; src += vl; n -= vl;
    }
}

/* Contiguous int8 zero-fill, vectorized. Equivalent to memset(dst, 0, n). */
static inline void gq31_vzero_i8(elem_t *dst, size_t n)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m8(n);
        __riscv_vse8_v_i8m8(dst, __riscv_vmv_v_x_i8m8(0, vl), vl);
        dst += vl; n -= vl;
    }
}

/* Two-stage Q0.31 requantize, vectorized over a run of OC channels.
 * Bit-identical to the scalar int64 reference (see file header):
 *   stage1 = round_half_up(acc * output_multiplier, 2^31)
 *   stage2 = round_half_up(stage1, 2^output_shift)          (shift > 0)
 *          = stage1 << (-output_shift)                      (shift < 0, exact)
 * vsmul_vx(..., RNU) performs stage1 (Q0.31 multiply-round, matches the
 * scalar '+ (1<<30) >> 31'); vnclip_wx(..., RNU) performs stage2 (matches
 * the scalar '+ (1<<(shift-1)) >> shift'). Structurally the same
 * construction as kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c's
 * mb_requant_i32m4. */
static inline vint8m1_t gq31_requant_i32m4(vint32m4_t vacc,
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
    /* ws_weight is gone — generate_skeleton.py::_backend_pack_weight
     * pre-packs weights to flat HWIO at codegen time, so we pass
     * `weight` straight into tiled_matmul_auto below. */
    static elem_t ws_input  [WS_BYTES]     __attribute__((aligned(64)));
    static elem_t ws_im2col [IM2COL_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[ACC_ELEMS]    __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    /* Fall back to scalar for configs exceeding workspace or using offsets.
     * offsets != 0 would require zero-point subtraction inside the GEMM, which
     * tiled_matmul_auto does not support; gemmini_im2col_full_C assumes
     * symmetric per-tensor int8 (offsets == 0 from extract_int8). */
    if (input_offset != 0 || filter_offset != 0
            || (size_t)(N * IH * IW * IC) > WS_BYTES
            || (size_t)(K_inner * OC)      > WS_BYTES
            || (size_t)(N * OH * OW * OC)  > WS_BYTES
            || K_inner * DIM               > IM2COL_ELEMS
            || OC * DIM                    > ACC_ELEMS) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        in_v = input_offset;
                                    else
                                        in_v = (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                                             + input_offset;
                                    /* weight is HWIO-packed:
                                     * idx = ((kh*KW + kw)*IC + ic)*OC + oc */
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                   + filter_offset);
                                }
                            }
                        }
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Transpose input NCHW → NHWC into ws_input (RVV).
     *
     * For a fixed (n, h) this is a plain IC x IW -> IW x IC transpose:
     *   src[c][w] = input[((n*IC + c)*IH + h)*IW + w]   (row stride IH*IW)
     *   dst[w][c] = ws_input[((n*IH + h)*IW + w)*IC + c] (row stride IC)
     * Exactly one of the two sides has to be strided. Both orientations move
     * the same number of elements, so pick the one with the LONGER vector so
     * the per-iteration vsetvl/loop overhead is amortized over more work:
     *   IW >= IC -> walk w (unit-stride load, strided store, vl up to IW)
     *   IC >  IW -> walk c (strided load, unit-stride store, vl up to IC)
     * dronet hits the second branch on every layer (IC=32..128 vs IW=4..27),
     * yolov8's wide early layers hit the first. Pure data movement -- the
     * bytes written are identical to the scalar loop this replaces. */
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < IH; h++) {
            const elem_t *in_nh  = &input[((size_t)n*IC*IH + h)*IW];
            elem_t       *out_nh = &ws_input[(((size_t)n*IH + h)*IW)*IC];
            if (IW >= IC) {
                for (int c = 0; c < IC; c++) {
                    const elem_t *src = in_nh  + (size_t)c*IH*IW;
                    elem_t       *dp  = out_nh + c;
                    size_t w = 0, rem = (size_t)IW;
                    while (rem > 0) {
                        size_t vl = __riscv_vsetvl_e8m8(rem);
                        __riscv_vsse8_v_i8m8(dp + w*(size_t)IC, (ptrdiff_t)IC,
                                             __riscv_vle8_v_i8m8(src + w, vl), vl);
                        w += vl; rem -= vl;
                    }
                }
            } else {
                for (int w = 0; w < IW; w++) {
                    const elem_t *src = in_nh  + w;
                    elem_t       *dp  = out_nh + (size_t)w*IC;
                    size_t c = 0, rem = (size_t)IC;
                    while (rem > 0) {
                        size_t vl = __riscv_vsetvl_e8m8(rem);
                        __riscv_vse8_v_i8m8(dp + c,
                            __riscv_vlse8_v_i8m8(src + c*(size_t)IH*IW,
                                                 (ptrdiff_t)IH*IW, vl), vl);
                        c += vl; rem -= vl;
                    }
                }
            }
        }
    }

    /* Weight is already HWIO-packed by the codegen
     * (generate_skeleton.py::_backend_pack_weight, --backend gemmini).
     * Layout = `[KH*KW*IC, OC]` flat — exactly the B-matrix layout
     * tiled_matmul_auto wants — so no ws_weight copy needed; we'll
     * pass `weight` directly to tiled_matmul_auto below. */

    /* Drain CPU store buffer (covers ws_input writes — the weight
     * was a const blob so already coherent, but keep the fence here
     * as gemmini mvin sets up A and B together). */
    asm volatile("fence" ::: "memory");

    /* Process output positions in tiles of DIM rows. */
    for (int tile_i = 0; tile_i < total_out; tile_i += DIM) {
        int tile_rows = total_out - tile_i < DIM ? total_out - tile_i : DIM;

        /* Build im2col A-matrix: DIM rows × K_inner cols.
         * Row i holds the flattened receptive field for output position
         * (tile_i + i).  Rows past tile_rows are zero-padded. */
        for (int i = 0; i < DIM; i++) {
            elem_t *row = &ws_im2col[(size_t)i * K_inner];
            if (i >= tile_rows) {
                gq31_vzero_i8(row, (size_t)K_inner);
                continue;
            }
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            /* iw for kw=0. The kw loop steps iw by exactly 1 regardless of SW,
             * so for a fixed kh the KW cells the scalar loop wrote one channel
             * at a time are CONTIGUOUS on both sides:
             *   src advances by IC per kw (ws_input is NHWC, iw+1 -> +IC)
             *   dst advances by IC per kw (row is [kh][kw][ic])
             * => the whole in-bounds kw run is a single contiguous copy of
             * (kw_hi - kw_lo)*IC bytes, and the out-of-bounds prefix/suffix are
             * two contiguous zero-fills. This both removes KW-1 of every KW
             * loop set-ups and hands the copy a vector length of KW*IC instead
             * of IC (e.g. 96B instead of 32B for dronet's 3x3/IC=32 layers).
             * Byte-for-byte the same im2col matrix as the scalar loop. */
            const int iw0 = ow_idx * SW - PW;
            int kw_lo = -iw0 > 0 ? -iw0 : 0;
            int kw_hi = (IW - iw0) < KW ? (IW - iw0) : KW;
            if (kw_lo > KW) kw_lo = KW;
            if (kw_hi < kw_lo) kw_hi = kw_lo;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh_idx * SH - PH + kh;
                elem_t *base = row + (size_t)kh * KW * IC;
                if (ih < 0 || ih >= IH) {          /* whole row out of bounds */
                    gq31_vzero_i8(base, (size_t)KW * IC);
                    continue;
                }
                if (kw_lo > 0)
                    gq31_vzero_i8(base, (size_t)kw_lo * IC);
                if (kw_hi > kw_lo) {
                    const elem_t *src = &ws_input[
                        ((((size_t)n_idx*IH + ih)*IW) + (size_t)(iw0 + kw_lo))*IC];
                    gq31_vcopy_i8(base + (size_t)kw_lo * IC, src,
                                  (size_t)(kw_hi - kw_lo) * IC);
                }
                if (kw_hi < KW)
                    gq31_vzero_i8(base + (size_t)kw_hi * IC,
                                  (size_t)(KW - kw_hi) * IC);
            }
        }

        /* Drain CPU stores to ws_im2col before gemmini mvin. */
        asm volatile("fence" ::: "memory");

        /* GEMM: ws_im2col [DIM × K_inner] × weight [K_inner × OC] + bias[OC].
         * weight is pre-packed HWIO from codegen — flat [KH*KW*IC, OC] is
         * exactly the B-matrix layout tiled_matmul_auto wants.
         * full_C=true → raw int32 accumulator output (no float-scale mvout). */
        tiled_matmul_auto(
            DIM, OC, K_inner,
            ws_im2col, weight,
            (const void *)bias, (void *)ws_acc_out,
            K_inner, OC, OC, OC,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
            bias != NULL,
            false, false,
            true, false,
            0, WS
        );

        /* Wait for gemmini DMA writes to ws_acc_out to reach L2.
         * gemmini_fence drains the in-flight mvout DMAs; gemmini_flush
         * resets the controller. Both are needed — without the fence,
         * yolov8-scale tiles (DIM * OC * K_inner large) can race with
         * the CPU's subsequent ws_acc_out reads in the requantize loop,
         * which corrupts the stack and surfaces as mcause=1 mepc=0
         * (return-address-zeroed) several frames later. See
         * modelblaster/notes/gemmini_tiled_conv_fence_required.md. */
        gemmini_fence();
        gemmini_flush(0);

        /* RVV Q0.31 requantize: int32 accumulator -> int8, written
         * straight into `output`'s NCHW layout (stride OH*OW elements
         * between consecutive oc) via a strided vector store. No NHWC
         * staging buffer, no separate transpose pass (kernel_opt_log
         * id 304's fix, applied here too). */
        for (int i = 0; i < tile_rows; i++) {
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            int8_t *dst0 = output + (((size_t)n_idx * OC) * OH + oh_idx) * OW + ow_idx;
            ptrdiff_t oc_stride_bytes = (ptrdiff_t)OH * OW;  /* elem_t is 1 byte */
            int oc = 0;
            while (oc < OC) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc));
                vint32m4_t vacc = __riscv_vle32_v_i32m4(&ws_acc_out[i * OC + oc], vl);
                vint8m1_t vout = gq31_requant_i32m4(
                    vacc, output_multiplier, output_shift, output_offset,
                    activation_min, activation_max, vl);
                __riscv_vsse8_v_i8m1(dst0 + (size_t)oc * OH * OW,
                                     oc_stride_bytes, vout, vl);
                oc += (int)vl;
            }
        }
    }
}
