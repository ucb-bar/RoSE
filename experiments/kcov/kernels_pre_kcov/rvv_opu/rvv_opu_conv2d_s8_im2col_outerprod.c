/* source: curated */
/* algorithm: im2col_outerprod */
/* accuracy_class: bit_exact */
/* origin: Saturn-OPU i8 conv2d via scalar im2col + tiled OPU
 *   outer-product GEMM. Reuses the proven OPU opcode sequence from
 *   rvv_opu_linear_s8_outerprod.c (VOPACC / OPMVINBCAST / VMV_VR plus
 *   vsetvli, vlse8.v, vle32.v, vse32.v, vmv.v.i) — the exact set
 *   already validated bit-exact for mlp_control on the
 *   FireSimGemminiAndOPUShuttleConfig bitstream across v8/v9/v10.
 *
 *   Why scalar im2col rather than the spec's indir_gemm pattern:
 *   indir_gemm uses vluxei8 (indexed gather), which traps as illegal
 *   instruction on this bitstream — observed during a prior FireSim
 *   run that exercised vluxei8 in a candidate silu kernel and hung
 *   in the fault loop for 37 minutes before being cancelled. Scalar
 *   im2col + tiled outer-product uses only the opcode set with
 *   direct prior evidence of working on this FPGA, paying the
 *   im2col cost in O(M_tile * IC*KH*KW) scalar ops — small next to
 *   M*N*K OPU MACs.
 *
 *   Tiling: M_tile = N_tile = mlmax = VLEN/8. For VLEN=128 the
 *   matrix tile is 16x16. Outer loops walk (n, oh, ow_tile, oc_tile);
 *   the K reduction (=IC*KH*KW) runs unrolled per tile with vlse8.v
 *   strided reads of the im2col strip (input side) and the OIHW
 *   weight matrix (weight side). Bit-exact with the reference impl
 *   for the symmetric-quant case; falls back to the reference scalar
 *   for asymmetric quant or any K that exceeds the per-hart scratch.
 *
 *   Bit-exactness rationale: VOPACC accumulates int8 x int8 into the
 *   int32 OPU accumulator (cWidth=32 per opuParams); this matches the
 *   reference's `acc += (int32_t)in * (int32_t)w` exactly. For
 *   padded pixels the strip is memset to 0 so the VOPACC contribution
 *   is `0 * w = 0`, matching the reference's "ih/iw out-of-bounds =>
 *   in_v = input_offset = 0" path (symmetric quant). Bias broadcast
 *   via OPMVINBCAST matches the reference's `bias[oc]` initialization.
 *   The Q0.31 requantize tail is byte-for-byte the same routine.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>
#define SATURN_OPU_KEEP_REGISTER_MACROS
#include "saturn_opu.h"

/* Per-hart im2col scratch. Sized for the worst-case yolov8 conv shape
 * we expect on rvv_opu: IC*KH*KW = 1152 (l21.cv1.conv: 128*3*3),
 * times OW_BLK = mlmax = 16 = 18 KB. We use 24 KB to leave headroom
 * for future shapes; oversize shapes fall back to the reference
 * scalar kernel. The bitstream has 2 harts (mhartid {0, 1}); two
 * independent buffers prevent cross-hart aliasing if both ever dispatch
 * a conv concurrently. */
#define MODELBLASTER_OPU_CONV_SCRATCH_BYTES (24 * 1024)
static int8_t g_conv_scratch[2][MODELBLASTER_OPU_CONV_SCRATCH_BYTES]
    __attribute__((aligned(16)));

static inline int read_mhartid(void) {
    unsigned long h;
    asm volatile("csrr %0, mhartid" : "=r"(h));
    return (int)h;
}

static inline int32_t q31_requantize(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return ((int32_t)prod + round) >> shift;
    }
    return (int32_t)prod << -shift;
}

/* Scalar reference fallback. Identical math to the spec's reference_impl
 * in pipeline/reference_kernels.py::CONV2D_S8 so this kernel can stand
 * in for the reference when shape/quant constraints push us off the
 * OPU path. Same code shape that pre-Phase-G yolov8 ran on rvv_opu. */
static void conv2d_s8_scalar_fallback(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    int8_t *output, int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int output_offset,
    int output_multiplier, int output_shift,
    int activation_min, int activation_max
) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
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
                  if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                    in_v = input_offset;
                  } else {
                    in_v = (int32_t)input[(in_row_base + ih) * IW + iw]
                         + input_offset;
                  }
                  int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw]
                              + filter_offset;
                  acc += in_v * w_v;
                }
              }
            }
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0) {
              scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (output_shift - 1))) >> output_shift);
            } else if (output_shift < 0) {
              scaled = scaled << (-output_shift);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)scaled;
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
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K = IC * KH * KW;

    /* mlmax = VLMAX for e8/m1 = VLEN/SEW = 128/8 = 16. Must NOT be
     * a `const = 16` — gcc would constant-propagate it and lower
     * subsequent intrinsics to vsetivli (immediate-AVL form), which
     * the Saturn bitstream rejects even though the register-AVL form
     * works (see commit log on the im2col_rvv_reduce sibling for the
     * full mtval decode). Read via inline-asm vsetvli SET form so the
     * compiler must keep mlmax in a register, then a `volatile` write
     * back to defeat any later constant propagation. */
    size_t mlmax;
    asm volatile("vsetvli %0, %1, e8, m1, ta, ma"
                 : "=r"(mlmax) : "r"((size_t)16));
    asm volatile("" : "+r"(mlmax));
    int OW_BLK = (int)mlmax;

    if (input_offset != 0 || filter_offset != 0 ||
        (size_t)(K * OW_BLK) > (size_t)MODELBLASTER_OPU_CONV_SCRATCH_BYTES) {
        conv2d_s8_scalar_fallback(input, weight, bias, output, N, IC, IH, IW, OC,
                                   KH, KW, SH, SW, PH, PW,
                                   input_offset, filter_offset, output_offset,
                                   output_multiplier, output_shift,
                                   activation_min, activation_max);
        return;
    }

    int8_t *strip = g_conv_scratch[read_mhartid() & 1];
    /* Row drain scratch — sized for mlmax up to 64 (V512 D256). */
    int32_t row_i32[64];

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow_tile = 0; ow_tile < OW; ow_tile += OW_BLK) {
                int M_tile = OW - ow_tile;
                if (M_tile > OW_BLK) M_tile = OW_BLK;

                /* --- Scalar im2col strip: strip[m, k] for m in [0, M_tile),
                 *     k = (ic*KH + kh)*KW + kw. Output pixel (oh, ow_tile+m).
                 *     Default zero (memset) handles padded pixels: for
                 *     symmetric quant (input_offset = 0), the reference's
                 *     padded contribution is exactly 0, so leaving the
                 *     strip lane at 0 is bit-exact. */
                memset(strip, 0, (size_t)K * (size_t)M_tile);
                for (int m = 0; m < M_tile; m++) {
                    int ow = ow_tile + m;
                    for (int ic = 0; ic < IC; ic++) {
                        const size_t in_row_base = ((size_t)n * IC + ic) * IH;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                strip[(size_t)m * K + (size_t)(ic*KH + kh)*KW + kw] =
                                    input[(in_row_base + (size_t)ih) * IW + iw];
                            }
                        }
                    }
                }

                /* --- Tiled OPU outer-product GEMM:
                 *     m1[m, c] = bias[oc_tile+c]
                 *              + sum_k strip[m, k] * weight[oc_tile+c, k]
                 *     where k = (ic*KH + kh)*KW + kw — both im2col strip
                 *     and OIHW weights index this k the same way, so a
                 *     single linear k-loop with strided int8 loads
                 *     suffices. The OPU MAC body is identical to the
                 *     proven linear_s8_outerprod kernel. */
                for (int oc_tile = 0; oc_tile < OC; oc_tile += OW_BLK) {
                    int N_tile = OC - oc_tile;
                    if (N_tile > OW_BLK) N_tile = OW_BLK;

                    /* Seed m1 with bias broadcast (or zero). N_tile i32
                     * lanes in v0; OPMVINBCAST replicates across all rows. */
                    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                                 : : "r"((size_t)N_tile));
                    if (bias) {
                        asm volatile("vle32.v v0, (%0)"
                                     : : "r"(&bias[oc_tile]));
                    } else {
                        asm volatile("vmv.v.i v0, 0");
                    }
                    OPMVINBCAST(m1, v0);

                    /* K reduction. Each iter loads M_tile lanes of the
                     * strip column k (input rows) and N_tile lanes of
                     * the weight column k (weight rows = output channels
                     * in oc_tile), then one VOPACC fused outer-product
                     * MAC into m1. Strides are K bytes — strip is
                     * row-major [M_tile, K], weight is OIHW with leading
                     * dim K = IC*KH*KW. */
                    const ptrdiff_t strip_stride = (ptrdiff_t)K * sizeof(int8_t);
                    const ptrdiff_t w_stride     = (ptrdiff_t)K * sizeof(int8_t);
                    for (int k = 0; k < K; k++) {
                        asm volatile("vsetvli zero, %0, e8, m1, ta, ma"
                                     : : "r"((size_t)M_tile));
                        asm volatile("vlse8.v v16, (%0), %1"
                                     : : "r"(&strip[k]),
                                         "r"((unsigned long)strip_stride));
                        asm volatile("vsetvli zero, %0, e8, m1, ta, ma"
                                     : : "r"((size_t)N_tile));
                        asm volatile("vlse8.v v18, (%0), %1"
                                     : : "r"(&weight[(size_t)oc_tile * K + k]),
                                         "r"((unsigned long)w_stride));
                        VOPACC(m1, v18, v16);
                    }

                    /* Drain m1 rows to i32 scratch, requantize, store as
                     * i8 into output[n, oc, oh, ow]. Vector width = N_tile
                     * for the vse32. The drain reads only M_tile rows and
                     * N_tile elements per row, so any garbage lanes from
                     * tail-agnostic VOPACC behavior on rows >= M_tile or
                     * cols >= N_tile are discarded. */
                    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                                 : : "r"((size_t)N_tile));
                    for (int m = 0; m < M_tile; m++) {
                        VMV_VR(v0, m, m1);
                        asm volatile("vse32.v v0, (%0)" : : "r"(row_i32));
                        int ow = ow_tile + m;
                        for (int c = 0; c < N_tile; c++) {
                            int oc = oc_tile + c;
                            int32_t v = q31_requantize(row_i32[c],
                                                       output_multiplier,
                                                       output_shift);
                            v += output_offset;
                            if (v < activation_min) v = activation_min;
                            if (v > activation_max) v = activation_max;
                            output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)v;
                        }
                    }
                }
            }
        }
    }
}
