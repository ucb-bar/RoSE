/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 matmul C=A.T@B, A stored [K,M]. Row-broadcast
   over N; A[k*M+m] is the per-k scalar. fp32 reorder => numeric_drift. */
#include <riscv_vector.h>

void kernel_matmul_ta(const float *A, const float *B, float *C,
                      int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        float *Cr = C + (size_t)m * N;
        for (int n = 0; n < N; ) {
            size_t vl = __riscv_vsetvl_e32m8(N - n);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int k = 0; k < K; k++) {
                float a = A[(size_t)k * M + m];
                vfloat32m8_t b = __riscv_vle32_v_f32m8(B + (size_t)k * N + n, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, a, b, vl);
            }
            __riscv_vse32_v_f32m8(Cr + n, acc, vl);
            n += vl;
        }
    }
}
