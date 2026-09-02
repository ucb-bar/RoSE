/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV lstm_s8. One LSTM timestep (seq=1, batch=1). The cost
 *   is the two int8 gate GEMVs per unit — x . w_ih[g] (length in_size) and
 *   h . w_hh[g] (length H) — for 4 gates x H units. Both operands are contiguous
 *   (x and each weight row are unit-stride), so each dot vectorizes with the
 *   same vle8/vwmul/vredsum reduction that made the linear_s8 kernel ~20x
 *   faster. The gate nonlinearities + cell update stay scalar float (O(H), tiny)
 *   and identical to the reference, so the result is bit-exact: the int8 dot
 *   accumulates in int32 (order-independent, no overflow: |x*w| < 127^2 and
 *   in_size <= ~600 => < 10M << INT32_MAX), and the float math is unchanged.
 *   Deferred hidden writeback preserved (the h.w_hh GEMM reads the PREVIOUS h;
 *   new hidden goes to out[], committed to h[] after all units). */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* Unit-stride int8 dot product over n elements (both operands contiguous). */
static inline int32_t mb_lstm_dot_s8(const int8_t *a, const int8_t *b, int n) {
    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4());
    int k = 0;
    size_t vl;
    for (; k < n; k += (int)vl) {
        vl = __riscv_vsetvl_e8m1(n - k);
        vint8m1_t va = __riscv_vle8_v_i8m1(a + k, vl);
        vint8m1_t vb = __riscv_vle8_v_i8m1(b + k, vl);
        vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
        vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
    }
    vint32m1_t vinit = __riscv_vmv_s_x_i32m1(0, 1);
    vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(
        vacc, vinit, __riscv_vsetvlmax_e32m4());
    return __riscv_vmv_x_s_i32m1_i32(vsum);
}

void kernel_lstm_s8(const int8_t *x, const int8_t *w_ih, const int8_t *w_hh,
                    const float *bias, int8_t *h, int8_t *c, int8_t *out,
                    int in_size, int H,
                    float s_x, float s_wih, float s_whh, float s_h, float s_c) {
    const float sx = s_x * s_wih;   /* dequant for the x . w_ih accumulator */
    const float sr = s_h * s_whh;   /* dequant for the h . w_hh accumulator */
    for (int j = 0; j < H; j++) {
        float pre[4];
        for (int g = 0; g < 4; g++) {
            int row = g * H + j;
            const int8_t *wih_row = w_ih + (long)row * in_size;
            const int8_t *whh_row = w_hh + (long)row * H;
            int32_t acc_x = mb_lstm_dot_s8(x, wih_row, in_size);
            int32_t acc_h = mb_lstm_dot_s8(h, whh_row, H);
            pre[g] = (float)acc_x * sx + (float)acc_h * sr + bias[row];
        }
        float ig = 1.0f / (1.0f + expf(-pre[0]));   /* input gate  */
        float fg = 1.0f / (1.0f + expf(-pre[1]));   /* forget gate */
        float cg = tanhf(pre[2]);                   /* cell gate   */
        float og = 1.0f / (1.0f + expf(-pre[3]));   /* output gate */
        float c_prev = (float)c[j] * s_c;
        float c_new  = fg * c_prev + ig * cg;
        float h_new  = og * tanhf(c_new);
        int32_t cq = (int32_t)roundf(c_new / s_c);
        if (cq < -128) cq = -128;
        if (cq > 127) cq = 127;
        int32_t hq = (int32_t)roundf(h_new / s_h);
        if (hq < -128) hq = -128;
        if (hq > 127) hq = 127;
        c[j]   = (int8_t)cq;
        out[j] = (int8_t)hq;   /* into out[] so h[] stays the previous state */
    }
    for (int j = 0; j < H; j++) h[j] = out[j];   /* commit new hidden state */
}
