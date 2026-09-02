/* source: curated */
/* algorithm: per_input_lut */
/* accuracy_class: bit_exact */
/* origin: v18 — same LUT pattern that shipped silu_s8_rvv_lut_gather and
 *   batchnorm2d_s8_per_channel_lut. For each input tensor i, the rescale
 *   output is a deterministic function of the int8 input pixel:
 *     q = clamp(roundf((v * scale_i) / scale_out))
 *   where v ranges -128..127. Build a 256-entry LUT per input at function
 *   entry, then the inner H*W*c_i loop becomes a scalar indexed lookup.
 *
 *   The reference vectorized direct kernel
 *   (rvv_opu_cat2_c1_s8_direct.c) is registered but fails spike-harness
 *   verify on yolov8 model run — the vfcvt + double-vnclip pipeline
 *   differs from the reference's roundf(f * s_in / s_out) order by
 *   enough rounding to trip the strict bit-exact gate. This LUT
 *   variant is bit-exact by construction because every LUT entry is
 *   computed by the SAME scalar math the reference inner loop uses.
 *
 *   Saturn-OPU FPGA constraint: vluxei8 is unimplemented (per the silu
 *   kernel's earlier finding) — the lookup stays scalar. Expected ~10x
 *   over the reference per-pixel roundf+division for typical yolov8
 *   shapes; ~5x for very small spatial where LUT-build dominates.
 */

#include <math.h>
#include <stdint.h>

static inline void _build_cat2_lut(int8_t lut[256], float scale_in, float scale_out,
                                  int activation_min, int activation_max) {
    for (int v = 0; v < 256; v++) {
        int8_t iv = (int8_t)(v - 128);
        float f = (float)iv * scale_in;
        int32_t q = (int32_t)roundf(f / scale_out);
        if (q < activation_min) q = activation_min;
        if (q > activation_max) q = activation_max;
        lut[v] = (int8_t)q;
    }
}

void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0,
                       const int8_t *in1, int c1, float scale1,
                       int8_t *output,
                       int N, int H, int W,
                       float scale_out,
                       int activation_min, int activation_max) {
    int stride = H * W;
    int8_t lut0[256], lut1[256];
    _build_cat2_lut(lut0, scale0, scale_out, activation_min, activation_max);
    _build_cat2_lut(lut1, scale1, scale_out, activation_min, activation_max);
    int c_total = c0 + c1;
    for (int n = 0; n < N; n++) {
        int out_c = 0;
        /* input 0 */
        {
            int n_in0 = c0 * stride;
            int n_out0 = (n * c_total + out_c) * stride;
            const int8_t *src = in0 + n * n_in0;
            int8_t *dst = output + n_out0;
            int total = c0 * stride;
            for (int i = 0; i < total; i++) {
                dst[i] = lut0[(int)src[i] + 128];
            }
            out_c += c0;
        }
        /* input 1 */
        {
            int n_in1 = c1 * stride;
            int n_out1 = (n * c_total + out_c) * stride;
            const int8_t *src = in1 + n * n_in1;
            int8_t *dst = output + n_out1;
            int total = c1 * stride;
            for (int i = 0; i < total; i++) {
                dst[i] = lut1[(int)src[i] + 128];
            }
        }
    }
}
