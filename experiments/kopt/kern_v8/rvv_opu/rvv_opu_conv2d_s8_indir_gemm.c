/* source: curated */
/* algorithm: indir_gemm */
/* origin: Saturn OPU indirect-GEMM conv2d_s8. Maps conv onto the OPU
 *   outer-product MAC using an XNNPACK-style indirection buffer that
 *   addresses the input tensor in its native [N, IC, IH, IW] layout
 *   (no im2col data duplication).
 *
 *   References:
 *     - XNNPACK indirect-conv design (Marat Dukhan, Google).
 *     - Saturn OPU programming model: modelblaster/cores/saturn_opu/include/saturn_opu.h
 *     - Design note: modelblaster/notes/opu_indirect_gemm_design.md
 *
 *   Indirection table size is BOUNDED PER TILE (not per layer): the
 *   kernel handles OW_BLK = mlmax output pixels at a time, so the
 *   table is at most KH*KW*mlmax pointers (≤ 25*64 = 1600 entries =
 *   12.8 KB on V512 worst case — stack-safe).
 *
 *   Build cost is ~KH*KW*mlmax per tile (~50 cycles for 3×3 mlmax=16);
 *   amortized over the OPU MAC's K*mlmax cycles per tile, ~3% overhead.
 *   Whole-layer compile-time indirection (see design note Option A) is
 *   a follow-up; v1 builds at runtime per-tile for signature parity.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>
#include "saturn_opu.h"

#define OPU_MAX_TILE  64    /* mlmax cap (V512) */
#define OPU_MAX_K     1024  /* fits the K=IC*KH*KW reduction in our scratch */
#define OPU_MAX_IC    1280  /* largest IC across modelblaster-flow models today */
#define OPU_INDIR_PAD ((int32_t)-1)  /* sentinel: this entry is padding, read 0 */

/* Mirror reference_kernels.py CONV2D_S8.reference_impl byte-for-byte:
 * round-add happens in i64 so prod + 1<<(shift-1) can't overflow i32
 * at the high end of the Q0.31 range. The earlier i32 form silently
 * wrapped on a small fraction of outputs in yolov8n (max_abs_err=128
 * i8 sign-flip). */
static inline int32_t q31_req_conv2d(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        scaled = (int32_t)(((int64_t)scaled +
                            ((int64_t)1 << (shift - 1))) >> shift);
    } else if (shift < 0) {
        scaled = scaled << (-shift);
    }
    return scaled;
}

static void conv2d_s8_scalar_fallback(
    const int8_t *input, const int8_t *weight,
    const int32_t *bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int output_offset,
    int output_multiplier, int output_shift,
    int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    /* Loop order matches reference_kernels.py CONV2D_S8.reference_impl
     * to keep auto-vectorization decisions parallel — kw is innermost
     * so GCC's RVV auto-vec gathers unit-stride along kw, not strided
     * along ic (which on Saturn V256 would hit the strided-vlse8 GPR
     * corruption from notes/saturn_strided_memop_bug.md). */
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++) {
                        const size_t in_row_base = ((size_t)n * IC + ic) * IH;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v;
                                if (ih < 0 || ih >= IH ||
                                    iw < 0 || iw >= IW) {
                                    in_v = input_offset;
                                } else {
                                    in_v = (int32_t)input[
                                        (in_row_base + ih) * IW + iw]
                                        + input_offset;
                                }
                                int32_t w_v = (int32_t)weight[
                                    ((oc * IC + ic) * KH + kh) * KW + kw]
                                    + filter_offset;
                                acc += in_v * w_v;
                            }
                        }
                    }
                    int32_t v = q31_req_conv2d(
                        acc, output_multiplier, output_shift);
                    v += output_offset;
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    output[((n * OC + oc) * OH + oh) * OW + ow] = (int8_t)v;
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
                      int activation_min, int activation_max) {
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;

    /* Eligibility: symmetric quant (input/filter offsets = 0), IC fits
     * the static zero-buf cap, KH*KW within reason, OPU available. */
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));
    const int MLMAX = (int)mlmax;
    if (input_offset != 0 || filter_offset != 0 ||
        IC > OPU_MAX_IC || (int)mlmax > OPU_MAX_TILE ||
        KH * KW * IC > OPU_MAX_K) {
        conv2d_s8_scalar_fallback(input, weight, bias, output,
                                  N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                                  input_offset, filter_offset, output_offset,
                                  output_multiplier, output_shift,
                                  activation_min, activation_max);
        return;
    }

    /* Per-tile indirection OFFSETS: KH*KW entries × MLMAX pixels.
     * Storing offsets (i32) instead of full pointers because the
     * pointer +ic dereference would need NHWC layout (IC contiguous),
     * but our IR uses OIHW (IC stride = IH*IW). With offsets we
     * encode the (n, 0, ih, iw) base; the gather adds ic*IH*IW.
     * OPU_INDIR_PAD = -1 marks padding (zero-contribute) entries.
     * 25 * 64 * 4 = 6.4 KB max on V512 — stack-safe. */
    int32_t indir[OPU_MAX_TILE * 25 /* KH*KW max */];

    /* Padded bias and i32 drain buffers (always MLMAX wide). */
    int32_t bias_pad[OPU_MAX_TILE];
    int32_t drain[OPU_MAX_TILE * OPU_MAX_TILE];

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow_tile = 0; ow_tile < OW; ow_tile += MLMAX) {
                const int ow_blk = (OW - ow_tile < MLMAX) ?
                                   (OW - ow_tile) : MLMAX;

                /* ---- Build per-tile indirection ----
                 * indir[(kh*KW + kw) * MLMAX + p] = offset into the
                 * input tensor at (n, 0, ih, iw), where (ih, iw) is
                 * the input position for output pixel (oh, ow_tile+p,
                 * kh, kw). Out-of-bounds (padded) entries get
                 * OPU_INDIR_PAD. Lanes p >= ow_blk also padded. */
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * SH - PH + kh;
                    const int ih_ok = (ih >= 0 && ih < IH);
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw_base = ow_tile * SW - PW + kw;
                        const int kk = kh * KW + kw;
                        for (int p = 0; p < ow_blk; p++) {
                            const int iw = iw_base + p * SW;
                            if (ih_ok && iw >= 0 && iw < IW) {
                                /* Offset to (n, 0, ih, iw): the IC=0
                                 * byte. Gather adds ic*IH*IW later. */
                                indir[kk * MLMAX + p] = (int32_t)(
                                    (size_t)n * IC * IH * IW
                                  + (size_t)ih * IW + iw);
                            } else {
                                indir[kk * MLMAX + p] = OPU_INDIR_PAD;
                            }
                        }
                        for (int p = ow_blk; p < MLMAX; p++) {
                            indir[kk * MLMAX + p] = OPU_INDIR_PAD;
                        }
                    }
                }

                /* ---- OC tiling: process MLMAX output channels at a time ---- */
                for (int oc_tile = 0; oc_tile < OC; oc_tile += MLMAX) {
                    const int oc_blk = (OC - oc_tile < MLMAX) ?
                                       (OC - oc_tile) : MLMAX;

                    /* Seed m1 with bias (broadcast across all rows /
                     * output pixels). Pad oc lanes ≥ oc_blk to zero so
                     * the accumulator stays clean for the inactive lanes. */
                    if (bias) {
                        for (int c = 0; c < oc_blk; c++) {
                            bias_pad[c] = bias[oc_tile + c];
                        }
                        for (int c = oc_blk; c < MLMAX; c++) {
                            bias_pad[c] = 0;
                        }
                    } else {
                        for (int c = 0; c < MLMAX; c++) bias_pad[c] = 0;
                    }
                    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                                 : : "r"((size_t)MLMAX));
                    asm volatile("vle32.v v0, (%0)" : : "r"(bias_pad));
                    OPMVINBCAST(m1, v0);

                    /* ---- VOPACC inner loop: (kh, kw, ic) over the K dim ---- */
                    asm volatile("vsetvli zero, %0, e8, m1, ta, ma"
                                 : : "r"((size_t)MLMAX));
                    const size_t ic_stride = (size_t)IH * IW;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            const int kk = kh * KW + kw;
                            const int32_t *indir_kk = &indir[kk * MLMAX];
                            for (int ic = 0; ic < IC; ic++) {
                                /* vs1: MLMAX i8s gathered from
                                 *   input[indir_kk[p] + ic*IH*IW]
                                 * if indir_kk[p] != PAD, else 0.
                                 *
                                 * Scalar gather — slow but correct.
                                 * Replace with vluxei (indexed load)
                                 * once we move offsets into a vector
                                 * register and add ic*IH*IW via
                                 * vadd.vx. PAD entries handled via
                                 * branch inside the gather. */
                                int8_t lane_buf[OPU_MAX_TILE];
                                for (int p = 0; p < MLMAX; p++) {
                                    int32_t off = indir_kk[p];
                                    lane_buf[p] = (off == OPU_INDIR_PAD)
                                        ? 0
                                        : input[(size_t)off + ic * ic_stride];
                                }
                                asm volatile("vle8.v v16, (%0)"
                                             : : "r"(lane_buf));

                                /* vs2: MLMAX i8 weights at
                                 *   weight[oc_tile..oc_tile+MLMAX, ic, kh, kw],
                                 * stride = IC*KH*KW per OC. */
                                const int8_t *w_ptr = weight
                                    + (size_t)oc_tile * IC * KH * KW
                                    + ((size_t)ic * KH + kh) * KW + kw;
                                const ptrdiff_t w_stride =
                                    (ptrdiff_t)IC * KH * KW;
                                asm volatile(
                                    "vlse8.v v18, (%0), %1"
                                    : : "r"(w_ptr), "r"((unsigned long)w_stride));

                                VOPACC(m1, v18, v16);
                            }
                        }
                    }

                    /* ---- Drain m1 to i32 scratch, requantize, store ----
                     * Each row of m1 corresponds to one output pixel
                     * within the (oh, ow_tile) tile; each col to one
                     * output channel within (oc_tile). */
                    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                                 : : "r"((size_t)MLMAX));
                    for (int p = 0; p < MLMAX; p++) {
                        VMV_VR(v0, p, m1);
                        asm volatile("vse32.v v0, (%0)"
                                     : : "r"(&drain[p * MLMAX]));
                    }

                    /* Requantize the valid ow_blk × oc_blk sub-block. */
                    for (int p = 0; p < ow_blk; p++) {
                        for (int c = 0; c < oc_blk; c++) {
                            int32_t acc = drain[p * MLMAX + c];
                            int32_t v = q31_req_conv2d(
                                acc, output_multiplier, output_shift);
                            v += output_offset;
                            if (v < activation_min) v = activation_min;
                            if (v > activation_max) v = activation_max;
                            const int oc = oc_tile + c;
                            const int ow = ow_tile + p;
                            output[((size_t)n * OC + oc) * OH * OW
                                   + (size_t)oh * OW + ow] = (int8_t)v;
                        }
                    }
                } /* oc_tile */
            } /* ow_tile */
        } /* oh */
    } /* n */
}
