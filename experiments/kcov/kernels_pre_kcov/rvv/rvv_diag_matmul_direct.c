/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 diag(a)@b == row-wise scale of b[N,M] by a[N].
   output[i,j]=a[i]*b[i,j]. Pure elementwise scale, no reduction => bit_exact. */
#include <riscv_vector.h>

void kernel_diag_matmul(const float *a, const float *b, float *output,
                        int N, int M) {
    for (int i = 0; i < N; i++) {
        float ai = a[i];
        const float *br = b + (size_t)i * M;
        float *or_ = output + (size_t)i * M;
        for (int j = 0; j < M; ) {
            size_t vl = __riscv_vsetvl_e32m8(M - j);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(br + j, vl);
            v = __riscv_vfmul_vf_f32m8(v, ai, vl);
            __riscv_vse32_v_f32m8(or_ + j, v, vl);
            j += vl;
        }
    }
}
