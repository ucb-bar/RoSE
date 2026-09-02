/* source: curated */
/* algorithm: rvv_vsmul_vnclip */
/*
 * ============================================================================
 * SHAPE-CONDITIONAL conv2d_s8: tiled-direct for KH*KW > 1, iGEMM for 1x1.
 * ============================================================================
 *
 * NOTE ON THE FILENAME (unchanged from before, still true): the
 * rvv_vsmul_vnclip ALGORITHM is retired. Measured on FPGA it is wrong on
 * yolov8n (untiled -> max_abs_err=3, tiled -> Store/AMO access fault) while
 * spike reports bit-exact, because it indexes the input with 32-bit int
 * arithmetic that wraps for BSS buffers above 0x80000000. This file carries
 * the production conv2d_s8 under that name because rvv_vsmul_vnclip sorts
 * before rvv_oc_blocked and rvv_igemm in spec.algorithms, so whatever lives
 * here is what every rvv model actually gets. Collapse the names when
 * convenient.
 *
 * WHY THE DISPATCH IS IN THE KERNEL AND NOT IN THE SELECTOR
 * --------------------------------------------------------
 * The kernel selector picks ONE algorithm per op for the whole model. It
 * cannot express "per-layer, by shape". yolov8n calls conv2d_s8 63 times with
 * two very different tap counts (39x 3x3, 24x 1x1), and the two algorithms
 * below have OPPOSITE FPGA verdicts on those two groups, so the choice has to
 * be made per call site -- i.e. here, as a runtime branch on KH/KW.
 *
 * THE MEASUREMENT THAT MOTIVATES IT (real FPGA, f2_dual_small_norose_tacit_
 * q31_60mhz, yolov8n, 63 conv layers; NOT spike -- spike reversed the sign of
 * this result and claimed iGEMM was a 1.58x win overall):
 *
 *     3x3 and larger : tiled-direct wins decisively (iGEMM 6-9x slower on the
 *                      small-channel C2f bottleneck layers). The indirection
 *                      buffer costs up to KH*KW=9 scattered pointer gathers
 *                      per tap, a DRAM/cache latency spike does not model.
 *     1x1            : the spatial gather disappears entirely (one tap, one
 *                      pointer, one ic-step per output pixel), leaving only
 *                      iGEMM's overhead-amortization benefit: the direct path
 *                      recomputes in_ic/w_ic/ih/row_in/iw0/iw_last and takes a
 *                      bounds branch on EVERY ic step, where iGEMM does one
 *                      pointer add.
 *
 * So: KH==1 && KW==1 -> iGEMM; everything else -> tiled-direct.
 *
 * The 1x1 iGEMM path is fully general in stride and padding (an out-of-bounds
 * tap gets the zero sentinel with ic-step 0, which still contributes
 * input_offset*(w+filter_offset) exactly as the conv2d_s8 spec requires), and
 * there is no dilation in this kernel's ABI, so no shape reaches it that it
 * cannot handle. Both paths are bit-exact against the scalar reference, and
 * they share one requantize/store sequence, so the dispatch changes no numeric
 * result whichever way it goes -- max_abs_err must stay 0.
 *
 * !!! THE 1x1 DISPATCH IS OFF BY DEFAULT AND MUST STAY OFF UNTIL THE NUMERIC
 * !!! DIVERGENCE BELOW IS ROOT-CAUSED. It is a real and large speedup that is
 * !!! NOT bit-exact on this FPGA. Build knobs:
 *
 *   (default)            tiled-direct for every shape. Compiles to code that
 *                        is instruction-for-instruction identical to the
 *                        pre-dispatch production kernel. max_abs_err = 0.
 *   -DMB_IGEMM_1X1=1     enable the 1x1 -> iGEMM dispatch. FAST (see table)
 *                        but max_abs_err = 5 on yolov8n on real hardware.
 *   -DMB_NO_IGEMM_1X1    hard opt-out; compiles the iGEMM body out entirely.
 *                        Same behaviour as the default, smaller binary.
 *   -DMB_FORCE_IGEMM     iGEMM for every shape (research arm; 1.55x on conv,
 *                        also not bit-exact -- max_abs_err = 1).
 *
 * WHAT WAS MEASURED (real FPGA, f2_dual_small_norose_tacit_q31_60mhz,
 * yolov8n, 63 conv2d_s8 layers, one matched IR + one matched curated kernel
 * set + one bitstream, all five arms submitted from the same session):
 *
 *   arm                              conv2d_s8 cycles   vs baseline  max_abs_err
 *   tiled-direct (baseline, job 188)      155,388,816       1.0000x        0
 *   1x1 dispatch (job 193)                135,300,478       1.1485x        5   <-- FAILS
 *   1x1 dispatch, GCC-cloned (job 189/190)134,714,913       1.1535x        6   <-- FAILS
 *   iGEMM everywhere (job 192)            100,031,882       1.5534x        1   <-- FAILS
 *   layout control (job 196)              155,395,165       1.0000x        0
 *
 *   split for the 1x1 dispatch arm: the 24 pointwise layers go 39,201,597 ->
 *   19,073,621 = 2.0553x and EVERY ONE of the 24 gets faster; the 39 non-1x1
 *   layers are 0.9997x, i.e. untouched, as the codegen identity guarantees.
 *
 * WHY IT IS OFF: the dispatch is not numerically neutral ON HARDWARE, and it
 * has to be. The two implementations are provably the same integer arithmetic
 * in the same order, spike agrees bit-for-bit at 27 shapes (every stride /
 * padding / OW-remainder / batch / no-bias combination plus the seven real
 * yolov8n pointwise shapes), and the whole-model spike verify is
 * max_abs_err=0 -- but yolov8n on the FPGA comes back max_abs_err=5,
 * deterministically (byte-identical reruns).
 *
 * WHAT THAT IS *NOT*: job 196 is the control that rules out the obvious
 * alternative. It links the iGEMM body in, so its .text/.rodata/BSS layout
 * and every model buffer address are identical to the enabled build's, but a
 * `volatile` gate keeps the dispatch from ever firing. It returns
 * max_abs_err=0 and baseline cycles. So the divergence is not the binary
 * moving and waking a latent layout-sensitive bug somewhere else in the
 * model -- it is EXECUTING the iGEMM body on the pointwise layers.
 *
 * WHERE TO LOOK NEXT: the iGEMM's indirection walk is the only thing the
 * direct path does not do. Note also that iGEMM-everywhere gives err=1 while
 * iGEMM-on-1x1-only gives err=5, so the size of the error is not monotone in
 * how much iGEMM runs -- consistent with a per-layer divergence whose effect
 * on the final tensor depends on which layers diverge. An isolated
 * kernel-vs-kernel unit test ON HARDWARE is the missing experiment
 * (the spike version of it is clean; the FPGA version hung in the harness
 * and was cancelled, jobs 191 and 195).
 *
 * The two implementations below are the two previously-shipped kernels moved
 * verbatim into static functions -- see their own doc blocks, which are kept
 * with them. NOTE: PATH B is a FORK of kernels/rvv/rvv_conv2d_s8_rvv_igemm.c
 * as of the point this file was written; that file is being evolved
 * independently (it has since been restructured to an ic-outer/tap-inner loop
 * for weight locality). For KH==KW==1 the two forms address exactly the same
 * weights and inputs, so the fork is behaviourally current for the only shape
 * class this file dispatches to it -- but re-sync before enabling
 * MB_FORCE_IGEMM for real work.
 *
 * WHY BOTH PATHS ARE __attribute__((noinline))
 * -------------------------------------------
 * It makes the A/B honest. With noinline, mb_conv2d_s8_tiled_direct compiles
 * to code that is instruction-for-instruction IDENTICAL to the previous
 * production kernel_conv2d_s8 (verified: same 96 basic blocks, same 1236
 * instructions, zero diffs), so every non-1x1 layer is bit-identical AND
 * cycle-identical to the baseline and the only measured delta is the 1x1
 * dispatch itself. Without it GCC re-schedules and re-allocates the merged
 * function and the baseline drifts by a couple of percent for no reason.
 * The wrapper it costs is 34 instructions, executed once per conv layer (63
 * times for the whole of yolov8n) against millions of cycles of inner loop.
 *
 * WHY THE iGEMM PATH IS ALSO __attribute__((noipa)) -- DO NOT REMOVE
 * ------------------------------------------------------------------
 * noipa blocks interprocedural constant propagation into the iGEMM body.
 * That looks like it throws away a free win (the dispatch condition proves
 * KH==KW==1, so GCC will happily clone the function as
 * mb_conv2d_s8_igemm.constprop.0 with taps folded to 1, the tap loop gone
 * and the indirection buffer demoted from a VLA to a fixed 4-entry array).
 * The clone is MEASURABLY WRONG ON HARDWARE.
 *
 * Measured, not theorised: with the clone enabled, yolov8n on
 * f2_dual_small_norose_tacit_q31_60mhz returns max_abs_err=6 against the
 * golden -- twice, cycle-for-cycle deterministic (fq jobs 189 and 190) --
 * while spike reports max_abs_err=0 for the same binary. With noipa the
 * same source is bit-exact on the same FPGA.
 *
 * What changes in the codegen is exactly one thing, and it is visible with
 * objdump: in the cloned function GCC schedules the bias load as
 *
 *     vsetvli zero,zero,e8,m1,ta,ma
 *     vle32.v v4,(t1)            <-- EEW=32 while SEW=8, so EMUL=4
 *
 * i.e. it reuses the e8/m1 vtype the following vle8 wants and relies on the
 * load's encoded EEW to imply EMUL=4. That is architecturally legal (vl and
 * VLMAX agree between e8/m1 and EEW=32/EMUL=4, and v4 is EMUL-aligned) and
 * spike executes it correctly. Every other build of this file -- the
 * tiled-direct path, -DMB_FORCE_IGEMM, -DMB_NO_IGEMM_1X1, and the original
 * standalone kernels/rvv/rvv_conv2d_s8_rvv_igemm.c -- emits the bias load
 * under a plain e32/m4 vtype and has zero such sites; the dispatch clone was
 * the only configuration that produced them, and the only one that failed.
 * Treat mixed-EEW unit-stride loads as a spike-invisible hazard on this
 * vector unit until the RTL is checked.
 *
 * Cost of the fix: the iGEMM body stays generic (a one-iteration tap loop
 * and a VLA of 4 pointers). Measured on FPGA that is still a 2.06x win on
 * the pointwise layers -- see the header table.
 */

/* Default-off gate. Unless the caller asks for the 1x1 iGEMM dispatch with
 * -DMB_IGEMM_1X1=1, this file behaves exactly like -DMB_NO_IGEMM_1X1, i.e.
 * exactly like the pre-dispatch production kernel. See the header for the
 * FPGA measurement that put it behind this gate. */
#if !defined(MB_FORCE_IGEMM) && !defined(MB_NO_IGEMM_1X1)
#  if !defined(MB_IGEMM_1X1) || (MB_IGEMM_1X1 == 0)
#    define MB_NO_IGEMM_1X1 1
#  endif
#endif

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


/* ==========================================================================
 * PATH A -- tiled-direct (production default; used for every KH*KW > 1 shape)
 * Verbatim body of the previous kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c
 * kernel_conv2d_s8, now a static function. Its original doc block follows.
 * ==========================================================================
 */
/* NOTE: the rvv_vsmul_vnclip ALGORITHM is retired. Measured on FPGA it is
 * wrong on yolov8n (untiled -> max_abs_err=3, tiled -> Store/AMO access fault)
 * while spike reports bit-exact, because it indexes the input with 32-bit int
 * arithmetic that wraps for BSS buffers above 0x80000000. This file now carries
 * the rvv_oc_blocked-derived tiled port so that EVERY selector path is safe:
 * rvv_vsmul_vnclip sorts before rvv_oc_blocked in spec.algorithms, so leaving
 * the old code here would keep handing models the broken kernel. Collapse the
 * two names when convenient.
 */
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

static __attribute__((noinline)) void mb_conv2d_s8_tiled_direct(const int8_t *input, const int8_t *weight,
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

#if !defined(MB_NO_IGEMM_1X1)

/* ==========================================================================
 * PATH B -- implicit GEMM with an indirection buffer (used for KH==KW==1).
 * Verbatim body of kernels/rvv/rvv_conv2d_s8_rvv_igemm.c's kernel_conv2d_s8,
 * now a static function. Its original doc block follows.
 * ==========================================================================
 */
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

static __attribute__((noinline, noipa)) void mb_conv2d_s8_igemm(const int8_t *input, const int8_t *weight,
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

#endif /* !MB_NO_IGEMM_1X1 */


/* ==========================================================================
 * DISPATCH
 * ========================================================================== */
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
#if defined(MB_FORCE_IGEMM)
    const int use_igemm = 1;
#elif defined(MB_NO_IGEMM_1X1)
    const int use_igemm = 0;
#else
    /* Pointwise only. Stride/padding are handled correctly by the iGEMM
     * indirection buffer (OOB taps take the zero sentinel with ic-step 0), so
     * no further shape guard is needed -- but a degenerate shape that would
     * make OH/OW non-positive is sent to the direct path, which is the one
     * that has always handled it. */
    const int use_igemm = (KH == 1 && KW == 1 && SH > 0 && SW > 0 &&
                           IH > 0 && IW > 0 && IC > 0 && OC > 0);
#endif

#if !defined(MB_NO_IGEMM_1X1)
    if (use_igemm) {
        mb_conv2d_s8_igemm(input, weight, bias, output,
                           N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                           input_offset, filter_offset, output_offset,
                           output_multiplier, output_shift,
                           activation_min, activation_max);
        return;
    }
#else
    (void)use_igemm;
#endif

    mb_conv2d_s8_tiled_direct(input, weight, bias, output,
                              N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                              input_offset, filter_offset, output_offset,
                              output_multiplier, output_shift,
                              activation_min, activation_max);
}
