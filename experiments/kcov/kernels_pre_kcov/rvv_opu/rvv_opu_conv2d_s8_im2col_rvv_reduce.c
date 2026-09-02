/* source: curated */
/* algorithm: im2col_rvv_reduce */
/* accuracy_class: bit_exact */
/* origin: Saturn-OPU i8 conv2d via scalar im2col + per-output vector
 *   reduce. Sister kernel to rvv_opu_conv2d_s8_im2col_outerprod.c.
 *   Uses ONLY the RVV opcodes already validated bit-exact on the
 *   FireSimGemminiAndOPUShuttleConfig FPGA by the mlp_control cached
 *   linear kernel (vsetvl_e8m1, vle8_v_i8m1, vwmul_vv_i16m2,
 *   vwadd_wv_i32m4, vredsum_vs_i32m4_i32m1, vmv_v_x_i32m4,
 *   vmv_s_x_i32m1, vmv_x_s_i32m1_i32) — NO OPU custom OP-V ops
 *   (VOPACC / OPMVINBCAST / VMV_VR / VMV_RV) and NO vluxei. Serves
 *   as the FPGA-safe fallback if the outer-product OPU variant
 *   turns out to be unsupported by this specific bitstream (same
 *   risk class as the vluxei8 trap that took down a prior silu
 *   kernel attempt).
 *
 *   Algorithm: build the im2col strip [M_tile, K] once per (n, oh,
 *   ow_tile) tile; then for each (m, oc) reduce over K via
 *   vwmul + vwadd-into-i32 + vredsum tail. Same Q0.31 requantize
 *   as the spec's reference impl.
 *
 *   Bit-exactness rationale: vwmul_vv_i16m2(va, vb) computes
 *   (int16)va * (int16)vb per lane, identical to the reference's
 *   (int32)input * (int32)weight when inputs fit in int8 (they do
 *   by construction). The i32 widen-accumulate via vwadd_wv keeps
 *   the running sum in int32 with no precision loss. vredsum sums
 *   the lanes in any associative order, but integer addition IS
 *   associative for integers within int32 range — and the lane
 *   contributions per (kh, kw, ic) tuple are at most |i8| * |i8|
 *   < 2^15, so up to 2^17 such products fit before overflow. The
 *   largest K in yolov8 (= IC*KH*KW = 128*3*3 = 1152) is well
 *   inside that envelope. Padded pixels use strip = 0 so their
 *   product contributes 0, matching the reference's symmetric-quant
 *   padded path. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

/* Per-hart strip scratch. Same sizing rationale as the outerprod
 * sibling: covers IC*KH*KW up to 1152 * M_tile = 16 with 24 KB
 * headroom; oversize shapes fall back to scalar reference. */
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

/* Strip tile height: 8 rows of output per im2col build. Trades a
 * little extra scalar im2col for tighter scratch usage. With K up to
 * 1152, M_TILE=8 needs 8*1152 = 9216 bytes, well under scratch. */
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
    /* VLMAX for e32/m4 — the bitstream is vlen=128 fixed at synthesis,
     * so VLMAX = VLEN/SEW*LMUL = 128/32*4 = 16 lanes. Read it via an
     * explicit inline-asm vsetvli SET form (rs1 = a runtime register
     * holding 16). The probe ran job 221 confirmed this exact form
     * works on hart 1 (e32/m4 SET rs1=16 → vl=16 OK).
     *
     * Critically: this CANNOT be `const size_t vlmax_i32 = 16;`. gcc
     * sees the constant and lowers subsequent
     * __riscv_vmv_v_x_i32m4(0, vlmax_i32) etc. to vsetIvli (note the
     * 'I' — IMMEDIATE-AVL form, encoding bit 31 = 1), which is a
     * DIFFERENT instruction from the register-AVL vsetvli the probe
     * verified. v11c trapped with mtval=0xcd287057 — that's vsetivli
     * zero, 16, e32, m4. The register form encoding (0x0d287057 with
     * bit 31 = 0) is what the bitstream accepts.
     *
     * Inline asm forces the register form by reading vlmax_i32 from a
     * non-constant runtime register, and `volatile` keeps gcc from
     * later constant-propagating it to subsequent intrinsics. */
    size_t vlmax_i32;
    asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                 : "=r"(vlmax_i32) : "r"((size_t)16));
    /* Defensive: keep gcc from substituting the literal 16 into the
     * vlmax_i32 uses below. */
    asm volatile("" : "+r"(vlmax_i32));

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow_tile = 0; ow_tile < OW;
                 ow_tile += MODELBLASTER_OPU_CONV_M_TILE) {
                int M_tile = OW - ow_tile;
                if (M_tile > MODELBLASTER_OPU_CONV_M_TILE)
                    M_tile = MODELBLASTER_OPU_CONV_M_TILE;

                /* Scalar im2col strip — identical math to the outerprod
                 * sibling. strip[m, k] is the input pixel that maps to
                 * output (oh, ow_tile+m). Default zero handles padding. */
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

                /* Per-output vector reduce: acc = sum_k strip[m,k] * w[oc,k] */
                for (int m = 0; m < M_tile; m++) {
                    int ow = ow_tile + m;
                    const int8_t *strip_row = &strip[(size_t)m * K];
                    for (int oc = 0; oc < OC; oc++) {
                        const int8_t *w_row = &weight[(size_t)oc * K];
                        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax_i32);
                        size_t vl;
                        for (int k = 0; k < K; k += (int)vl) {
                            /* Inline asm vsetvli with explicit rs1=K-k
                             * — equivalent to __riscv_vsetvl_e8m1 but
                             * opaque to gcc, so it can't lift a probe
                             * (vsetvli rs1=zero, e8/m1) which the Saturn
                             * bitstream traps as illegal. */
                            asm volatile(
                                "vsetvli %0, %1, e8, m1, ta, ma"
                                : "=r"(vl) : "r"((size_t)(K - k)));
                            vint8m1_t va = __riscv_vle8_v_i8m1(strip_row + k, vl);
                            vint8m1_t vb = __riscv_vle8_v_i8m1(w_row + k, vl);
                            vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
                            vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
                        }
                        /* Defensive: hide the AVL=1 from gcc so it can't
                         * lower vmv_s_x to vsetivli e32/m1 uimm=1 (we don't
                         * yet have FPGA evidence that ANY vsetivli works
                         * — probe job 221 tested vsetvli SET register
                         * form only). Use a runtime register holding 1. */
                        size_t _one;
                        asm volatile("vsetvli %0, %1, e32, m1, ta, ma"
                                     : "=r"(_one) : "r"((size_t)1));
                        asm volatile("" : "+r"(_one));
                        vint32m1_t vinit = __riscv_vmv_s_x_i32m1(0, _one);
                        vint32m1_t vsum =
                            __riscv_vredsum_vs_i32m4_i32m1(vacc, vinit, vlmax_i32);
                        int32_t acc = __riscv_vmv_x_s_i32m1_i32(vsum);
                        if (bias) acc += bias[oc];

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
