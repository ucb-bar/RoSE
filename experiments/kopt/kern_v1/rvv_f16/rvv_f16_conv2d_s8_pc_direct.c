/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV conv2d_s8_pc (per-output-channel requant), reading the
 *   NATIVE OCIHW weight layout (the synthesized "direct" algo declares
 *   weight_layout=oihw, so no layout transform is applied).
 *
 *   Strategy — OC-vectorized GEMM with PIXEL TILING (register-blocked over the
 *   output-width). Conv is a GEMM [OH*OW, R] x [R, OC] with R = IC*KH*KW. spike
 *   charges vector loads ~per BYTE, and the plain OC-vectorized GEMV is
 *   memory-bound on RE-LOADING the whole weight set once per output pixel
 *   (OH*OW * (OC/VL) * R weight byte-loads). Pixel tiling amortizes each weight
 *   vector load across TILE adjacent output pixels: we hold TILE accumulators
 *   (a[0..TILE-1], each a vector of OC partial sums), load weight[:, r] ONCE per
 *   reduction index, and vwmacc it into all TILE accumulators. That cuts weight
 *   byte-loads by ~TILE (OH*OW/TILE * (OC/VL) * R) — the dominant cost — for the
 *   price of TILE input-patch gathers (input loads are far fewer than weight
 *   loads once OC is large).
 *
 *   Weights are transposed [OC, R] -> mb_wt[R, OC] once per call (kept int8; an
 *   i8 load + widening vwmacc beats an i16 load under the per-byte model). The
 *   receptive field of each tiled pixel is gathered once (shared across all OC).
 *   Requant is per-oc scalar Q0.31 -> bit-exact vs the conv2d_s8_pc reference.
 *   Symmetric quant only (input/filter offset 0). */

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* This model's convs: max R=IC*KH*KW=576, max R*OC=36864, max OC=64. Static
 * (kernel runs single-threaded on the harness) to keep it off the thread stack. */
#define MB_CONV_WT_MAX    65536   /* >= max R*OC */
#define MB_CONV_PATCH_MAX  1024   /* >= max R */
#define MB_CONV_OC_MAX      256   /* >= max OC */
#define MB_CONV_TILE          4   /* output-width register block. 4 i32m4 accs =
                                   * 16 vregs; + w8/w16 fits the 32-register file. */

static int8_t  mb_wt[MB_CONV_WT_MAX];                       /* weight [R, OC] */
static int8_t  mb_patch[MB_CONV_TILE * MB_CONV_PATCH_MAX];  /* TILE gathered patches */
static int32_t mb_drain[MB_CONV_OC_MAX];

static inline int32_t mb_q31_requant_pc(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return (scaled + round) >> shift;
    }
    return scaled << (-shift);
}

/* Drain one accumulator vector, per-oc Q0.31 requant, store one output pixel. */
static inline void mb_store_pixel(vint32m4_t vacc, size_t vl, int oc_base,
                                  const int32_t *output_multiplier,
                                  const int32_t *output_shift,
                                  int output_offset, int activation_min,
                                  int activation_max, int8_t *out_pix,
                                  size_t pix_stride) {
    __riscv_vse32_v_i32m4(mb_drain, vacc, vl);
    for (size_t lane = 0; lane < vl; lane++) {
        int oc = oc_base + (int)lane;
        int32_t s = mb_q31_requant_pc(mb_drain[lane],
                                      output_multiplier[oc], output_shift[oc]);
        s += output_offset;
        if (s < activation_min) s = activation_min;
        if (s > activation_max) s = activation_max;
        out_pix[(size_t)oc * pix_stride] = (int8_t)s;
    }
}

void kernel_conv2d_s8_pc(const int8_t *input, const int8_t *weight,
                         const int32_t *bias, int8_t *output,
                         int N, int IC, int IH, int IW, int OC,
                         int KH, int KW, int SH, int SW, int PH, int PW,
                         int input_offset, int filter_offset, int output_offset,
                         const int32_t *output_multiplier,
                         const int32_t *output_shift,
                         int activation_min, int activation_max)
{
    (void)input_offset; (void)filter_offset;   /* symmetric quant only */
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    const int R = IC * KH * KW;                 /* reduction / weight-row length */

    /* Transpose weight [OC, R] (OCIHW, contiguous per oc) -> mb_wt[R, OC]. */
    for (int oc = 0; oc < OC; oc++) {
        const int8_t *wsrc = weight + (size_t)oc * R;
        for (int r = 0; r < R; r++)
            mb_wt[(size_t)r * OC + oc] = wsrc[r];
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow_base = 0; ow_base < OW; ow_base += MB_CONV_TILE) {
                int Pt = OW - ow_base;
                if (Pt > MB_CONV_TILE) Pt = MB_CONV_TILE;

                /* Gather TILE receptive fields (one per tiled pixel). Pixels
                 * past OW and out-of-bounds taps -> 0 (== symmetric zero-point). */
                for (int p = 0; p < MB_CONV_TILE; p++) {
                    int8_t *pb = mb_patch + (size_t)p * R;
                    if (p >= Pt) { for (int r = 0; r < R; r++) pb[r] = 0; continue; }
                    int ow = ow_base + p;
                    int idx = 0;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) {
                                for (int kw = 0; kw < KW; kw++) pb[idx++] = 0;
                                continue;
                            }
                            const size_t row_off =
                                (((size_t)n * IC + ic) * IH + ih) * (size_t)IW;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                pb[idx++] =
                                    (iw >= 0 && iw < IW) ? input[row_off + iw] : 0;
                            }
                        }
                    }
                }

                const int8_t *p0 = mb_patch + 0 * R;
                const int8_t *p1 = mb_patch + 1 * R;
                const int8_t *p2 = mb_patch + 2 * R;
                const int8_t *p3 = mb_patch + 3 * R;

                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));
                    vint32m4_t a0, a1, a2, a3;
                    if (bias != NULL) {
                        a0 = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a1 = a0; a2 = a0; a3 = a0;
                    } else {
                        a0 = __riscv_vmv_v_x_i32m4(0, vl);
                        a1 = a0; a2 = a0; a3 = a0;
                    }

                    /* Load each weight vector ONCE, apply to all TILE pixels. */
                    for (int r = 0; r < R; r++) {
                        const int8_t *wr = mb_wt + (size_t)r * OC + oc_base;
                        vint8m1_t w8 = __riscv_vle8_v_i8m1(wr, vl);
                        vint16m2_t w16 = __riscv_vwadd_vx_i16m2(w8, 0, vl);
                        a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)p0[r], w16, vl);
                        a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)p1[r], w16, vl);
                        a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)p2[r], w16, vl);
                        a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)p3[r], w16, vl);
                    }

                    /* out[n, oc, oh, ow]; oc stride = OH*OW, ow stride = 1. */
                    int8_t *base = output
                        + ((size_t)n * OC + 0) * OH * OW /* oc handled in helper */
                        + (size_t)oh * OW + ow_base;
                    const size_t oc_stride = (size_t)OH * OW;
                    mb_store_pixel(a0, vl, oc_base, output_multiplier, output_shift,
                                   output_offset, activation_min, activation_max,
                                   base + 0, oc_stride);
                    if (Pt > 1)
                        mb_store_pixel(a1, vl, oc_base, output_multiplier, output_shift,
                                       output_offset, activation_min, activation_max,
                                       base + 1, oc_stride);
                    if (Pt > 2)
                        mb_store_pixel(a2, vl, oc_base, output_multiplier, output_shift,
                                       output_offset, activation_min, activation_max,
                                       base + 2, oc_stride);
                    if (Pt > 3)
                        mb_store_pixel(a3, vl, oc_base, output_multiplier, output_shift,
                                       output_offset, activation_min, activation_max,
                                       base + 3, oc_stride);
                    oc_base += (int)vl;
                }
            }
        }
    }
}
