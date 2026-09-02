/* source: curated */
/* algorithm: im2col_vlA_scalarMAC */
/* accuracy_class: bit_exact */
/* origin: Worst-case-safe conv2d_s8 for the Saturn-OPU FireSim
 *   bitstream. Uses ONLY the RVV opcodes with direct prior evidence
 *   of working on the FPGA: vsetvli SET form for e8/m1 with rs1 = a
 *   small positive constant, and vle8.v + vse8.v at that vtype.
 *   No e16/m2, no e32/m4, no widening multiplies, no reductions, no
 *   OPU custom ops, no probes. Multiply-accumulate is scalar — we
 *   vectorize the LOAD path only and let the inner MAC be a plain C
 *   loop over the K vector lanes we just streamed.
 *
 *   Used as the fall-back if im2col_rvv_reduce trips on e32/m4 or
 *   e16/m2 vsetvli (which it did in v11 attempts 2 and 3 on
 *   FireSimGemminiAndOPUShuttleConfig). The savings vs. the pure
 *   reference scalar conv come from amortizing per-element load
 *   latency across 16-lane vle8 streams; estimated 3-5x cycle drop,
 *   substantially less than im2col_rvv_reduce's 20x but with zero
 *   FPGA risk.
 *
 *   Bit-exact rationale: each MAC is `acc += (int32_t)strip[m,k] *
 *   (int32_t)weight[oc,k]` — same form as the spec's reference impl.
 *   vle8 reads bytes verbatim; vse8 writes them back unchanged.
 *   Q0.31 requantize tail unchanged. Symmetric quant only (matching
 *   the other OPU conv variants). */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/* Strip tile + per-K scratch arrays. We only need to hold one chunk
 * of `vl` lanes at a time for the MAC loop. 64-byte buffers cover the
 * largest possible vlmax_e8 on VLEN=512 chipyard builds; for the
 * FireSim Saturn at vlen=128 only 16 are used. */
#define MODELBLASTER_OPU_CONV_VLANE_MAX 64
#define MODELBLASTER_OPU_CONV_M_TILE 8

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

    if (input_offset != 0 || filter_offset != 0 ||
        (size_t)(K * MODELBLASTER_OPU_CONV_M_TILE) >
            (size_t)MODELBLASTER_OPU_CONV_SCRATCH_BYTES) {
        conv2d_s8_scalar_fallback(input, weight, bias, output, N, IC, IH, IW, OC,
                                   KH, KW, SH, SW, PH, PW,
                                   input_offset, filter_offset, output_offset,
                                   output_multiplier, output_shift,
                                   activation_min, activation_max);
        return;
    }

    int8_t *strip = g_conv_scratch[read_mhartid() & 1];
    /* Per-call scratch for the vle8-streamed lanes. Stack-resident so
     * concurrent dispatches on the same hart can each have their own. */
    int8_t a_buf[MODELBLASTER_OPU_CONV_VLANE_MAX]
        __attribute__((aligned(16)));
    int8_t b_buf[MODELBLASTER_OPU_CONV_VLANE_MAX]
        __attribute__((aligned(16)));

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow_tile = 0; ow_tile < OW;
                 ow_tile += MODELBLASTER_OPU_CONV_M_TILE) {
                int M_tile = OW - ow_tile;
                if (M_tile > MODELBLASTER_OPU_CONV_M_TILE)
                    M_tile = MODELBLASTER_OPU_CONV_M_TILE;

                /* Scalar im2col strip [M_tile, K]. Padding zeroed. */
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

                for (int m = 0; m < M_tile; m++) {
                    int ow = ow_tile + m;
                    const int8_t *strip_row = &strip[(size_t)m * K];
                    for (int oc = 0; oc < OC; oc++) {
                        const int8_t *w_row = &weight[(size_t)oc * K];
                        int32_t acc = bias ? bias[oc] : 0;
                        int k = 0;
                        while (k < K) {
                            /* Stream up to mlmax_e8 lanes into scratch
                             * via vle8.v. Inline asm so gcc doesn't
                             * hoist a VLMAX probe. */
                            size_t vl;
                            asm volatile(
                                "vsetvli %0, %1, e8, m1, ta, ma"
                                : "=r"(vl) : "r"((size_t)(K - k)));
                            asm volatile("vle8.v v16, (%0)"
                                         : : "r"(strip_row + k));
                            asm volatile("vse8.v v16, (%0)"
                                         : : "r"(a_buf));
                            asm volatile("vle8.v v16, (%0)"
                                         : : "r"(w_row + k));
                            asm volatile("vse8.v v16, (%0)"
                                         : : "r"(b_buf));
                            for (size_t i = 0; i < vl; i++) {
                                acc += (int32_t)a_buf[i] * (int32_t)b_buf[i];
                            }
                            k += (int)vl;
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
}
