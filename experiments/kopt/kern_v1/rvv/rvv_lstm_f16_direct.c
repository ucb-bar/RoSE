/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV lstm_f16 (half-precision single-layer, seq-1 LSTM). Same
 *   structure as the curated lstm_s8, but fp16: the two gate GEMVs per unit —
 *   x . w_ih[g] (length in_size) and h . w_hh[g] (length H) — are vectorized with
 *   unit-stride fp16 loads + a PURE-fp16 multiply-accumulate (vfmacc_vv_f16) and
 *   an fp16 reduction (vfredusum_vs_f16). NOTHING widens to fp32: the whole vector
 *   hot loop stays SEW=16, so no fp32 vector datapath is needed (matters for the
 *   FPGA drone target — no LUT area for an fp32 vector ALU). This also matches the
 *   reference kernel's contract ("gate-GEMM reduction ACCUMULATES IN _Float16"):
 *   the only divergence from the fp16 reference is the reduction ORDER (fp16 is
 *   non-associative), hence accuracy_class=numeric_drift rather than bit_exact.
 *   The gate sigmoid/tanh + cell update are evaluated in scalar float (O(H), tiny)
 *   exactly as the reference — on the scalar FPU, not the vector unit. Deferred
 *   hidden writeback preserved (the h.w_hh GEMM reads the PREVIOUS h; new hidden
 *   goes to out[], committed to h[] after all units; out must not alias h). */

#include <math.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Unit-stride fp16 dot product over n elements, accumulated in fp16 (no fp32). */
static inline _Float16 mb_dot_f16(const _Float16 *a, const _Float16 *b, int n) {
    vfloat16m4_t vacc = __riscv_vfmv_v_f_f16m4((_Float16)0.0f,
                                               __riscv_vsetvlmax_e16m4());
    int k = 0;
    size_t vl;
    for (; k < n; k += (int)vl) {
        vl = __riscv_vsetvl_e16m4(n - k);
        vfloat16m4_t va = __riscv_vle16_v_f16m4(a + k, vl);
        vfloat16m4_t vb = __riscv_vle16_v_f16m4(b + k, vl);
        vacc = __riscv_vfmacc_vv_f16m4(vacc, va, vb, vl);   /* fp16 MAC, fp16 acc */
    }
    vfloat16m1_t vs = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vs = __riscv_vfredusum_vs_f16m4_f16m1(vacc, vs, __riscv_vsetvlmax_e16m4());
    return __riscv_vfmv_f_s_f16m1_f16(vs);
}

void kernel_lstm_f16(const _Float16 *x, const _Float16 *w_ih, const _Float16 *w_hh,
                     const _Float16 *bias, _Float16 *h, _Float16 *c, _Float16 *out,
                     int in_size, int H) {
    for (int j = 0; j < H; j++) {
        float pre[4];
        for (int g = 0; g < 4; g++) {
            int row = g * H + j;
            _Float16 ax = mb_dot_f16(x, w_ih + (long)row * in_size, in_size);
            _Float16 ah = mb_dot_f16(h, w_hh + (long)row * H, H);
            pre[g] = (float)ax + (float)ah + (float)bias[row];
        }
        float ig = 1.0f / (1.0f + expf(-pre[0]));
        float fg = 1.0f / (1.0f + expf(-pre[1]));
        float cg = tanhf(pre[2]);
        float og = 1.0f / (1.0f + expf(-pre[3]));
        float c_new = fg * (float)c[j] + ig * cg;
        c[j]   = (_Float16)c_new;
        out[j] = (_Float16)(og * tanhf(c_new));   /* into out[]; h[] stays previous */
    }
    for (int j = 0; j < H; j++) h[j] = out[j];   /* commit new hidden state */
}
