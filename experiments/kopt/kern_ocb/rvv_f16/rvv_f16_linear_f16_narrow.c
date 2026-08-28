/* source: curated */
/* algorithm: narrow */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp16 linear with PURE-fp16 accumulation. The K-reduction
 *   dot(input[m], weight[n]) is done with unit-stride fp16 loads + vfmacc_vv_f16
 *   (fp16 multiply-accumulate) and an fp16 reduction (vfredusum_vs_f16) — the whole
 *   vector hot loop stays SEW=16, NO widening to fp32. This is the fp32-free
 *   counterpart of rvv_f16_linear_f16_widening.c (which uses vfwmacc -> an fp32
 *   accumulator): use THIS on targets with no fp32 vector datapath (the FPGA drone
 *   target has no LUT area for an fp32 vector ALU). Divergence from the reference
 *   is reduction ORDER only (fp16 is non-associative) -> numeric_drift. bias +
 *   final store match the reference (output[m*N+n] = (_Float16)(acc + bias[n])). */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_linear_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int M, int K, int N) {
    const size_t vlmax = __riscv_vsetvlmax_e16m4();
    for (int m = 0; m < M; m++) {
        const _Float16 *in_row = input + (size_t)m * (size_t)K;
        for (int n = 0; n < N; n++) {
            const _Float16 *w_row = weight + (size_t)n * (size_t)K;
            vfloat16m4_t vacc = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
            int k = 0;
            size_t vl;
            for (; k < K; k += (int)vl) {
                vl = __riscv_vsetvl_e16m4(K - k);
                vfloat16m4_t va = __riscv_vle16_v_f16m4(in_row + k, vl);
                vfloat16m4_t vb = __riscv_vle16_v_f16m4(w_row + k, vl);
                vacc = __riscv_vfmacc_vv_f16m4(vacc, va, vb, vl);  /* fp16 MAC + acc */
            }
            vfloat16m1_t vs = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
            vs = __riscv_vfredusum_vs_f16m4_f16m1(vacc, vs, vlmax);
            _Float16 acc = __riscv_vfmv_f_s_f16m1_f16(vs);
            if (bias) acc = (_Float16)(acc + bias[n]);
            output[(size_t)m * N + n] = acc;
        }
    }
}
