/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 batched matmul C[b]=A[b]@B[b]. Per-batch slice is
   a dense matmul: row-broadcast over N, K reduction in registers.
   fp32 accumulation reorder => numeric_drift. */
#include <riscv_vector.h>

void kernel_bmm(const float *A, const float *B, float *C,
                int batch, int M, int K, int N) {
    for (int b = 0; b < batch; b++) {
        const float *Ab = A + (size_t)b * M * K;
        const float *Bb = B + (size_t)b * K * N;
        float *Cb = C + (size_t)b * M * N;
        for (int m = 0; m < M; m++) {
            const float *Ar = Ab + (size_t)m * K;
            float *Cr = Cb + (size_t)m * N;
            for (int n = 0; n < N; ) {
                size_t vl = __riscv_vsetvl_e32m8(N - n);
                vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                for (int k = 0; k < K; k++) {
                    float a = Ar[k];
                    vfloat32m8_t bv = __riscv_vle32_v_f32m8(Bb + (size_t)k * N + n, vl);
                    acc = __riscv_vfmacc_vf_f32m8(acc, a, bv, vl);
                }
                __riscv_vse32_v_f32m8(Cr + n, acc, vl);
                n += vl;
            }
        }
    }
}
