/* source: curated */
/* algorithm: gemmini_tiled_conv */
/* accuracy_class: numeric_drift */
/* origin: tiled_conv_auto — gemmini hardware im2col + GEMM + float-scale
 *         requantize in one call.  Validated on Saturn FireSim May 2026.
 *         Square kernels only (KH==KW, SH==SW, PH==PW); asymmetric offsets
 *         must be zero (symmetric per-tensor int8 from extract_int8).
 *         Float-scale introduces ≤1 LSB drift vs Q0.31 golden.
 *
 *         ROOT CAUSE of the drift, characterised 2026-08-28 (experiments/
 *         kernel_opt_log.jsonl id 1100-1103): the golden reference (scalar
 *         fallback below, kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c,
 *         gemmini_im2col_full_C's CPU requantize) all compute a TWO-STAGE
 *         rounding: stage1 = round_half_up(acc*mult, 2^31), THEN
 *         stage2 = round_half_up(stage1, 2^output_shift). gemmini's HW
 *         mvout unit only implements ONE hardware round-shift-by-31
 *         (cores/gemmini/include/gemmini_params.h::ACC_SCALE); to use it
 *         at all for a nonzero output_shift, this kernel (in its
 *         MODELBLASTER_GEMMINI_Q31_ACC_SCALE branch below) pre-folds
 *         (mult, shift) into a single Q0.31 scale = round(mult/2^shift)
 *         BEFORE the hardware multiply, i.e. computes
 *         round_half_up(acc * round(mult, shift), 2^31) instead. That is
 *         structurally NOT the same operation as double rounding:
 *         pre-rounding the SCALE turns a fixed +-0.5-output-unit rounding
 *         error into an error that scales with acc, so a single hardware
 *         round cannot reproduce the two-stage golden bit-for-bit for
 *         every acc value. This is unavoidable with this HW datapath (no
 *         Q0.31 fold choice fixes it) — the real fix is
 *         gemmini_im2col_full_C, which bypasses this mvout unit entirely
 *         (tiled_matmul_auto full_C=true drains the RAW accumulator, then
 *         the exact two-stage formula runs on the CPU/RVV) and is
 *         verified bit-exact in isolation. Kept here (not removed) as the
 *         faster, drift-tolerant option for backends/models whose
 *         atol_override covers ~1 LSB/layer; gemmini_q31 and
 *         gemmini_q31_rvv (pipeline/backends.py) no longer select it —
 *         see gemmini_q31_rvv_conv2d_s8_gemmini_im2col_full_C.c.
 *
 *         WEIGHT LAYOUT CONTRACT: this kernel expects `weight` to already
 *         be in flat HWIO layout (= `[KH*KW*IC, OC]`, the form
 *         `tiled_conv_auto` consumes directly).  The skeleton emitter at
 *         modelblaster/pipeline/generate_skeleton.py::_backend_pack_weight
 *         applies the OIHW→HWIO permutation at codegen time when
 *         `--backend gemmini`, so the runtime weight-transpose loop that
 *         used to live here (which copied into a per-call ws_weight
 *         workspace) is gone — the kernel passes `weight` straight into
 *         tiled_conv_auto.  Saves ~OC*IC*KH*KW int8 copies per layer
 *         (yolov8's biggest layer: 128 IC × 256 OC × 3×3 = 295 KB of
 *         scalar copy that's now zero). */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

/* 512 KB covers all square conv layers in dronet and yolov8_nano.
 *   dronet max weight: IC=128, KH=3, KW=3, OC=128 →  144 KB
 *   yolov8 max weight: IC=128, KH=3, KW=3, OC=256 →  288 KB
 *   yolov8 max input:  IC=3,   IH=160, IW=160      →   75 KB
 *   yolov8 max output: IC=16,  OH=80,  OW=80        →  100 KB
 * All three fit well within 512 KB. */
enum { GEMMINI_WS_BYTES = 512 * 1024 };

/* REENTRANCY (added 2026-09-01, quad-hetero ablation).
 * The workspaces below used to be plain function-scope statics. That is safe
 * only while ONE hart at a time is inside this kernel. It is NOT safe on an
 * SoC with more than one Gemmini tile -- e.g. chipyard.config.
 * SatGemQuadHeteroTacitConfig, whose harts 0 and 1 each carry their own
 * Q0.31 Gemmini -- where an OC-sharded conv puts one tile on each hart and
 * both call this function concurrently. Both would then transpose their
 * input into the same ws_input, let their accelerator drain into the same
 * ws_output, and overwrite each other's ws_bias: wrong results, plus timing
 * polluted by false sharing across 1 MB of shared lines.
 *
 * The runtime intra-op shard path (generate_skeleton's
 * _CONV2D_S8_SHARD_WRAPPER -> modelblaster_pool) calls this kernel from
 * several pool helpers at once, so the hazard is reachable there too; it had
 * simply never been exercised with the gemmini backend (the RVV conv kernel,
 * kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c, holds no mutable statics and
 * is reentrant as written).
 *
 * Fix: one workspace slot per Gemmini-bearing hart, selected by CPU id.
 * Single-hart builds keep exactly one slot, so nothing changes for them. */
#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_GEM_WS_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_GEM_WS_SLOT ((int)arch_proc_id())
#else
enum { MB_GEM_WS_SLOTS = 1 };
#define MB_GEM_WS_SLOT 0
#endif
enum { MB_GEM_BIAS_SLOT_ELEMS = 4096 };

/* ---- phase attribution (-DMB_GEM_PHASE_TRACE, off by default) ------------
 *
 * Splits one conv call into input-transpose / gemmini / output-transpose and
 * prints the three cycle counts. Measured this way on F2 (fq 475): input 34.3%,
 * gemmini 12.8%, output 52.9% -- i.e. 87% of "gemmini conv time" is layout
 * conversion, and the OUTPUT half is the larger one. The indirect estimate this
 * replaced (T = 2*max_tile - P over split A/Bs) said 57% and pointed at the
 * input half, so it was both low and aimed at the wrong place.
 *
 * TACIT would measure this without a source change and without having to guess
 * which phases to bracket, but its F2 bridge datapath drops data -- see
 * experiments/tacit/F2_TACIT_VERDICT.md.
 *
 * Guarded on CONFIG_PRINTK and using the file's own MB_GEM_WS_SLOT rather than
 * arch_proc_id(): the curated VERIFY build compiles this same kernels.c under
 * the spike harness, which is not a full Zephyr build. Calling arch_proc_id()
 * directly there fails to compile, and the curator then SILENTLY falls back to
 * a different conv algorithm rather than erroring.
 */
#if defined(MB_GEM_PHASE_TRACE) && defined(CONFIG_PRINTK)
#include <zephyr/sys/printk.h>
static inline uint64_t mb_gem_rdc(void)
{
    uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c;
}
#define MB_GEM_PH(v) uint64_t v = mb_gem_rdc()
#define MB_GEM_PH_EMIT(IC,IH,IW,OC,a,b,c,d)                                   \
    printk("MB_GEMPHASE,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",                     \
           MB_GEM_WS_SLOT, (IC), (IH), (IW), (OC),                             \
           (unsigned long long)((b) - (a)), (unsigned long long)((c) - (b)),   \
           (unsigned long long)((d) - (c)))
#else
#define MB_GEM_PH(v)            do { } while (0)
#define MB_GEM_PH_EMIT(...)     do { } while (0)
#endif


/* ---- OH row-tile window -------------------------------------------------
 *
 * An OH split hands each tile a band of output rows. In NCHW a row band is
 * contiguous at neither end -- it is IC runs inside the input planes and OC
 * runs inside the output planes -- so the generated wrapper used to gather the
 * tile's padded input band into a scratch buffer, call this kernel on it, and
 * scatter the output rows back. That is two extra passes over the activations,
 * and it is what made an OH split LOSE on gemmini in every measured cell
 * (0.67-1.20x against OC's 1.09-1.79x): gemmini runs the convolution ~10x
 * faster than rvv, so it has nothing to hide the copy traffic behind.
 *
 * Both passes are redundant here, because this kernel ALREADY walks the
 * activations with an explicit plane stride: it transposes NCHW->NHWC on the
 * way in and NHWC->NCHW on the way out. Teaching those two walks the PARENT's
 * stride and row offset makes them read and write the parent tensor in place,
 * which removes the gather and the scatter at no added cost -- the same bytes
 * move exactly once instead of three times.
 *
 * The window is per-hart for the same reason the workspaces above are: sibling
 * tiles of one split run concurrently on different harts, and one shared
 * window would be a data race that only appears under a schedule that actually
 * overlaps them. It is set immediately before the call and cleared after, so
 * every non-OH call sees active == 0 and takes byte-identical code paths.
 */
struct mb_gem_ohwin {
    int active;
    int row_lo;      /* parent input row of tile input row 0 (may be < 0) */
    int parent_IH;   /* parent input rows */
    int parent_IW;   /* parent input cols, UNPADDED */
    int PW;          /* horizontal pad folded into the tile's IW */
    int parent_OH;   /* parent output rows */
    int oh0;         /* parent output row of tile output row 0 */
};
static struct mb_gem_ohwin mb_gem_ohwin_all[MB_GEM_WS_SLOTS];

void mb_gem_ohwin_set(int row_lo, int parent_IH, int parent_IW, int PW,
                      int parent_OH, int oh0)
{
    struct mb_gem_ohwin *w = &mb_gem_ohwin_all[MB_GEM_WS_SLOT];
    w->row_lo = row_lo; w->parent_IH = parent_IH; w->parent_IW = parent_IW;
    w->PW = PW; w->parent_OH = parent_OH; w->oh0 = oh0;
    w->active = 1;
}

void mb_gem_ohwin_clear(void)
{
    mb_gem_ohwin_all[MB_GEM_WS_SLOT].active = 0;
}

/* One input element in TILE coordinates, resolved against the parent tensor.
 * Returns 0 for anything outside the parent -- which is exactly what the
 * materialised zero rows/columns used to supply, and what the conv2d_s8 spec
 * says an out-of-bounds tap contributes before input_offset is added. */
static inline int32_t mb_gem_win_in(const int8_t *in,
                                    const struct mb_gem_ohwin *w,
                                    int n, int IC, int ic, int ih, int iw,
                                    int IH, int IW)
{
    if (!w->active) {
        if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) return 0;
        return in[(((size_t)n * IC + ic) * IH + ih) * IW + iw];
    }
    /* A window only ever describes ONE tile of one image; the wrapper always
     * calls with N == 1, and the caller below refuses anything else. */
    int pih = w->row_lo + ih;
    int piw = iw - w->PW;
    if (pih < 0 || pih >= w->parent_IH || piw < 0 || piw >= w->parent_IW)
        return 0;
    return in[((size_t)ic * w->parent_IH + pih) * w->parent_IW + piw];
}

/* Address of one output element in TILE coordinates, inside the parent. */
static inline int8_t *mb_gem_win_out(int8_t *out, const struct mb_gem_ohwin *w,
                                     int n, int OC, int oc, int oh, int ow,
                                     int OH, int OW)
{
    if (!w->active)
        return out + ((((size_t)n * OC + oc) * OH + oh) * OW + ow);
    return out + ((size_t)oc * w->parent_OH + w->oh0 + oh) * (size_t)OW + ow;
}


void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    /* ws_weight is gone — the weight emitter pre-packs to HWIO at
     * codegen time, so we hand `weight` directly to tiled_conv_auto.
     * ws_input / ws_output are still needed for the activation
     * NCHW↔NHWC transposes on entry / exit. */
    static elem_t ws_input_all  [MB_GEM_WS_SLOTS][GEMMINI_WS_BYTES]
        __attribute__((aligned(64)));
    static elem_t ws_output_all [MB_GEM_WS_SLOTS][GEMMINI_WS_BYTES]
        __attribute__((aligned(64)));
    elem_t *const ws_input  = ws_input_all [MB_GEM_WS_SLOT];
    elem_t *const ws_output = ws_output_all[MB_GEM_WS_SLOT];
    const struct mb_gem_ohwin *const win = &mb_gem_ohwin_all[MB_GEM_WS_SLOT];

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    /* The windowed paths index the parent as a single image. apply_split_hint
     * only ever emits N == 1 tiles, so this is unreachable -- but a silent
     * wrong answer here would be a batch of images overwriting each other's
     * rows, which is exactly the class of bug the wrapper's scatter used to
     * make impossible. */
    if (win->active && N != 1)
        return;

    /* Fall back to scalar for non-square kernels, non-zero offsets, tensors
     * that exceed the static workspace, or output_shift outside the foldable
     * Q0.31 range. The earlier B_rows = (OC/DIM)*KH*KW*IC > BANK_NUM*BANK_ROWS/2
     * guard was a misdiagnosis — tiled_conv_auto handles arbitrary B_rows by
     * splitting the spatial loop. The original symptom (mcause=1, mepc=0)
     * came from a missing post-call gemmini_fence(), which we now emit. */
    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0
            || output_offset != 0
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
            || output_shift < 0 || output_shift > 30
#endif
            || (size_t)(N * IH * IW * IC) > GEMMINI_WS_BYTES
            || (size_t)(N * OH * OW * OC)  > GEMMINI_WS_BYTES) {
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
                                    int32_t in_v = mb_gem_win_in(
                                            input, win, n, IC, ic, ih, iw,
                                            IH, IW) + input_offset;
                                    /* weight is HWIO-packed per the
                                     * file-header contract:
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
                        *mb_gem_win_out(output, win, n, OC, oc, oh, ow,
                                        OH, OW) = (int8_t)scaled;
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

    MB_GEM_PH(mb_ph_t0);
    /* Transpose input NCHW → NHWC into ws_input.
     * Blocked (TB x TB) so the contiguous-read side and the IC-strided
     * write side both stay resident in L1. The old 4-deep loop nest
     * recomputed ((n*IH+h)*IW+w)*IC+c and ((n*IC+c)*IH+h)*IW+w per
     * element; on dronet conv0 the two transposes move 137,984 bytes and
     * were the single largest term in the layer's 1.75M cycles. */
    if (win->active) {
        /* Windowed: the same contiguous-read / IC-strided-write shape as the
         * blocked path below, walked one parent row at a time so the row and
         * column bounds are hoisted out of the inner loop. This REPLACES the
         * wrapper's gather -- the band is materialised straight into NHWC,
         * padding included, in a single pass. */
        const int TB = 32;
        for (int j = 0; j < IH; j++) {
            int pih = win->row_lo + j;
            int row_ok = (pih >= 0 && pih < win->parent_IH);
            for (int c0 = 0; c0 < IC; c0 += TB) {
                int cn = IC - c0 < TB ? IC - c0 : TB;
                for (int c = 0; c < cn; c++) {
                    const int8_t *sr = row_ok
                        ? input + ((size_t)(c0 + c) * win->parent_IH + pih)
                                  * win->parent_IW
                        : NULL;
                    elem_t *d = ws_input + (size_t)j * IW * IC + (c0 + c);
                    /* Split the row into its three runs -- left pad, real
                     * data, right pad -- so the inner loops carry no bounds
                     * test at all. The obvious single loop with a per-element
                     * `piw >= 0 && piw < parent_IW` test costs a branch on
                     * every byte, which on conv0 (IC=3, IW=112, 57 rows a
                     * tile) measured as a 245 us/tile penalty -- larger than
                     * the copies this path exists to remove. */
                    int i = 0;
                    if (!row_ok) {
                        for (; i < IW; i++) d[(size_t)i * IC] = (elem_t)0;
                    } else {
                        for (; i < win->PW; i++) d[(size_t)i * IC] = (elem_t)0;
                        for (int x = 0; x < win->parent_IW; x++, i++)
                            d[(size_t)i * IC] = (elem_t)sr[x];
                        for (; i < IW; i++) d[(size_t)i * IC] = (elem_t)0;
                    }
                }
            }
        }
    } else {
        const int TB = 32;
        int HW = IH * IW;
        for (int n = 0; n < N; n++) {
            const int8_t *inb = input   + (size_t)n * IC * HW;
            elem_t       *ob  = ws_input + (size_t)n * HW * IC;
            for (int p0 = 0; p0 < HW; p0 += TB) {
                int pn = HW - p0 < TB ? HW - p0 : TB;
                for (int c0 = 0; c0 < IC; c0 += TB) {
                    int cn = IC - c0 < TB ? IC - c0 : TB;
                    for (int c = 0; c < cn; c++) {
                        const int8_t *s = inb + (size_t)(c0 + c) * HW + p0;
                        elem_t       *d = ob  + (size_t)p0 * IC + (c0 + c);
                        for (int p = 0; p < pn; p++) d[(size_t)p * IC] = s[p];
                    }
                }
            }
        }
    }

    MB_GEM_PH(mb_ph_t1);
    /* Weight is already in flat HWIO layout ([KH*KW*IC, OC]) — see the
     * file-header WEIGHT LAYOUT CONTRACT.  The OIHW→HWIO permutation
     * happens at codegen time in
     * modelblaster/pipeline/generate_skeleton.py::_backend_pack_weight when
     * --backend=gemmini, so we pass `weight` directly into
     * tiled_conv_auto without going through a ws_weight workspace. */

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    /* Q31 gemmini config: acc_scale_t = SInt(32). HW mvout requantize is
     *   y = sat_int8((acc * scale + (1<<30)) >> 31)
     * The TFLite/modelblaster formula we want is
     *   y = sat_int8((acc * mult + (1<<30)) >> (31 + output_shift))
     * Fold (mult, shift) → single Q0.31 multiplier:
     *   scale = (mult + (1<<(s-1))) >> s   (compile-time round-to-nearest).
     * Differs from two-stage rounding by ≤1 LSB per layer; for deep nets
     * (yolov8) the drift accumulates and can exceed atol — those layers
     * fall back to gemmini_im2col_full_C (CPU im2col + scalar Q0.31). */
    int32_t scale_q31 = output_shift == 0
        ? output_multiplier
        : (int32_t)(((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;

    /* EXPERIMENT (beta=1 rounding-bias compensation).
     * The golden's two-stage round is exactly a ONE-stage round with a
     * larger numerator bias:
     *   round_s(round_31(acc*M)) == (acc*M + 2^30 + 2^(30+s)) >> (31+s)
     * The HW mvout gives  ((acc)*m + 2^30) >> 31  with m = M/2^s, i.e. it
     * is short by 2^(30-s) in the numerator.  The only lever that adds an
     * acc-INDEPENDENT constant to the numerator is the accumulator bias:
     * bias += beta contributes beta*m.  The exact beta would be
     * 2^(30-s)/m = 2^30/M in (0.5, 1]; beta=1 is the nearest integer and
     * is provably the error-minimising choice (2^30/M > 0.5 always).
     * This does NOT make the path exact -- see the kernel_opt_log entry --
     * it just moves it to the floor of what this datapath can do. */
    static acc_t ws_bias_all[MB_GEM_WS_SLOTS][MB_GEM_BIAS_SLOT_ELEMS];
    acc_t *const ws_bias = ws_bias_all[MB_GEM_WS_SLOT];
    const acc_t *bias_used;
    if (OC <= (int)MB_GEM_BIAS_SLOT_ELEMS) {
        for (int oc = 0; oc < OC; oc++)
            ws_bias[oc] = (bias ? bias[oc] : 0) + 1;
        bias_used = ws_bias;
    } else {
        bias_used = bias;
    }
#else
    /* Default (Saturn FireSim) gemmini config: acc_scale_t = float.
     * effective_scale = output_multiplier * 2^(-(31 + output_shift)). f32
     * has ~24 bits of mantissa precision; loses ~1 LSB per layer. */
    float scale = ldexpf((float)output_multiplier, -(31 + output_shift));
#endif

    /* RELU (1) clamps to [0, INT8_MAX]; NO_ACTIVATION (0) allows full range. */
    int act_kind = (activation_min == 0) ? 1 : 0;

    /* Drain CPU store buffer before gemmini mvin reads ws_input/ws_weight. */
    asm volatile("fence" ::: "memory");

    /* tiled_conv_auto: gemmini does im2col + GEMM + requantize in hardware.
     * WS = weight-stationary dataflow. */
    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, weight, bias_used, ws_output,   /* weight is pre-packed HWIO from codegen */
        act_kind, scale,
        0, 0, 0,
        WS
    );

    /* tiled_conv_auto's body (tiled_conv) does NOT end with a
     * gemmini_fence — unlike tiled_matmul_outer_eigen.  Without this
     * explicit drain, the post-conv NHWC->NCHW read and the next op's
     * gemmini_flush race with in-flight mvout DMAs and corrupt memory
     * (FireSim Saturn: mcause=1, mepc=0). */
    gemmini_fence();
    gemmini_flush(0);

    MB_GEM_PH(mb_ph_t2);
    /* Transpose output NHWC → NCHW (same blocking as the input side). */
    {
        const int TB = 32;
        int OHW = OH * OW;
        for (int n = 0; n < N; n++) {
            int8_t       *outb = output    + (size_t)n * OC * OHW;
            const elem_t *ob   = ws_output + (size_t)n * OHW * OC;
            /* Resolve the window ONCE. It is loop-invariant, and calling
             * mb_gem_win_out() per channel put a branch on w->active plus a
             * wider address computation inside the hot nest, where the plain
             * form is arithmetic the compiler strength-reduces. Measured
             * (fq 478 vs the pre-window baseline): the per-iteration call cost
             * 6% across EVERY conv, on the non-windowed path that the OH
             * feature does not even use. Hoisting restores it.
             *
             * With a window active `obase` is the tile's first output row
             * inside the PARENT and `ostride` is the parent's plane stride, so
             * the band lands where the wrapper's scatter used to put it. */
            int8_t *const obase = mb_gem_win_out(outb, win, 0, OC, 0, 0, 0,
                                                 OH, OW);
            const size_t  ostride = win->active
                                  ? (size_t)win->parent_OH * OW : (size_t)OHW;
            for (int p0 = 0; p0 < OHW; p0 += TB) {
                int pn = OHW - p0 < TB ? OHW - p0 : TB;
                for (int c0 = 0; c0 < OC; c0 += TB) {
                    int cn = OC - c0 < TB ? OC - c0 : TB;
                    for (int c = 0; c < cn; c++) {
                        int8_t       *d = obase + (size_t)(c0 + c) * ostride
                                                + p0;
                        const elem_t *s = ob   + (size_t)p0 * OC + (c0 + c);
                        for (int p = 0; p < pn; p++) d[p] = s[(size_t)p * OC];
                    }
                }
            }
        }
    }

    MB_GEM_PH(mb_ph_t3);
    MB_GEM_PH_EMIT(IC, IH, IW, OC, mb_ph_t0, mb_ph_t1, mb_ph_t2, mb_ph_t3);

    /* Post-clamp for activation_max < 127 (gemmini RELU only handles min==0). */
    if (activation_max < 127) {
        /* Clamp the tile's OWN rows. With a window active `output` is the
         * PARENT, so the flat 0..N*OC*OH*OW walk would run off this tile's band
         * and clamp rows belonging to a sibling tile -- which is both wrong and
         * a data race. Within one plane the band IS contiguous, so one run of
         * OH*OW per output channel covers it exactly. */
        for (int oc = 0; oc < OC; oc++) {
            int8_t *row = mb_gem_win_out(output, win, 0, OC, oc, 0, 0,
                                         OH, OW);
            for (int i = 0; i < OH * OW; i++)
                if (row[i] > activation_max) row[i] = (int8_t)activation_max;
        }
    }
}
