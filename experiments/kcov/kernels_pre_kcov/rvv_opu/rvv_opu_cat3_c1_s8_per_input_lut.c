/* source: curated */
/* algorithm: per_input_lut */
/* accuracy_class: bit_exact */
/* origin: v18 — same LUT pattern as cat2/per_input_lut. Three inputs,
 *   three 256-entry LUTs, scalar gather. Bit-exact with the reference
 *   roundf(input * scale_i / scale_out) math.
 */

#include <math.h>
#include <stdint.h>

static inline void _build_cat3_lut(int8_t lut[256], float scale_in, float scale_out,
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

void kernel_cat3_c1_s8(const int8_t *in0, int c0, float scale0,
                       const int8_t *in1, int c1, float scale1,
                       const int8_t *in2, int c2, float scale2,
                       int8_t *output,
                       int N, int H, int W,
                       float scale_out,
                       int activation_min, int activation_max) {
    int stride = H * W;
    int c_total = c0 + c1 + c2;
    int8_t lut0[256], lut1[256], lut2[256];
    _build_cat3_lut(lut0, scale0, scale_out, activation_min, activation_max);
    _build_cat3_lut(lut1, scale1, scale_out, activation_min, activation_max);
    _build_cat3_lut(lut2, scale2, scale_out, activation_min, activation_max);
    for (int n = 0; n < N; n++) {
        int out_c = 0;
        { /* input 0 */
            const int8_t *src = in0 + n * (c0 * stride);
            int8_t *dst = output + (n * c_total + out_c) * stride;
            int total = c0 * stride;
            for (int i = 0; i < total; i++) dst[i] = lut0[(int)src[i] + 128];
            out_c += c0;
        }
        { /* input 1 */
            const int8_t *src = in1 + n * (c1 * stride);
            int8_t *dst = output + (n * c_total + out_c) * stride;
            int total = c1 * stride;
            for (int i = 0; i < total; i++) dst[i] = lut1[(int)src[i] + 128];
            out_c += c1;
        }
        { /* input 2 */
            const int8_t *src = in2 + n * (c2 * stride);
            int8_t *dst = output + (n * c_total + out_c) * stride;
            int total = c2 * stride;
            for (int i = 0; i < total; i++) dst[i] = lut2[(int)src[i] + 128];
        }
    }
}
