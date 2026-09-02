/* source: curated */
/* algorithm: widening */
/* origin: RVV+Zvfh fp16 linear, fp32-accumulator dot product via vfwmacc.
 *
 * Replaces the scalar reference impl's K-element MAC loop with a vectorized
 * widening multiply-accumulate. Each iteration consumes vl fp16 input + vl
 * fp16 weight elements via vfwmacc, which produces a wide-LMUL fp32
 * accumulator. The fp32 accumulator matches the reference impl's
 * `float acc = ...; output[m*N+n] = (_Float16)acc` semantics, so this is
 * BIT-EXACT to the reference modulo summation order. (Summation order
 * differs because the vector reduction tree's pairwise structure is not
 * the same as the scalar left-to-right accumulator; both keep the result
 * in fp32, so the worst-case drift is ~1 ulp of fp16 at the final cast.)
 */

#include <riscv_vector.h>

void kernel_linear_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int M, int K, int N) {
    /* vlmax for the wide fp32 accumulator (LMUL=4); the matching narrow
     * fp16 LMUL is m2, since vfwmacc widens m2→m4. */
    const size_t vlmax_e32m4 = __riscv_vsetvlmax_e32m4();

    for (int m = 0; m < M; m++) {
        const _Float16 *in_row = input + (size_t)m * (size_t)K;
        for (int n = 0; n < N; n++) {
            const _Float16 *w_row = weight + (size_t)n * (size_t)K;

            /* fp32 accumulator vector; final reduction is below. */
            vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_e32m4);

            int k = 0;
            while (k < K) {
                size_t vl = __riscv_vsetvl_e16m2((size_t)(K - k));
                vfloat16m2_t va = __riscv_vle16_v_f16m2(in_row + k, vl);
                vfloat16m2_t vb = __riscv_vle16_v_f16m2(w_row  + k, vl);
                /* vfwmacc: vacc[i] += (float)va[i] * (float)vb[i]. */
                vacc = __riscv_vfwmacc_vv_f32m4(vacc, va, vb, vl);
                k += (int)vl;
            }

            /* Reduce the fp32 accumulator vector to a scalar. The init
             * value is 0 in an m1 fp32 vector; vfredusum sums across vl
             * lanes plus the init scalar. */
            vfloat32m1_t vsum0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m4_f32m1(
                vacc, vsum0, vlmax_e32m4);
            float acc = __riscv_vfmv_f_s_f32m1_f32(vred);

            if (bias) acc += (float)bias[n];
            output[(size_t)m * (size_t)N + (size_t)n] = (_Float16)acc;
        }
    }
}
