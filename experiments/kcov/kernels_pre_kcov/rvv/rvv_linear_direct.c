/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 linear (matmul + bias). input[M,K] x weight[N,K]
   (out-features outer). Both operands contiguous over K => dot-product reduction
   via vfmul + vfredusum. fp32 accumulation reorder => numeric_drift. */
#include <riscv_vector.h>

void kernel_linear(const float *input, const float *weight, const float *bias,
                   float *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *ir = input + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const float *wr = weight + (size_t)n * K;
            vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f,
                                    __riscv_vsetvl_e32m8(K));
            for (int k = 0; k < K; ) {
                size_t vl = __riscv_vsetvl_e32m8(K - k);
                vfloat32m8_t a = __riscv_vle32_v_f32m8(ir + k, vl);
                vfloat32m8_t w = __riscv_vle32_v_f32m8(wr + k, vl);
                vsum = __riscv_vfmacc_vv_f32m8(vsum, a, w, vl);
                k += vl;
            }
            vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vfloat32m1_t r = __riscv_vfredusum_vs_f32m8_f32m1(vsum, z,
                                 __riscv_vsetvl_e32m8(K));
            float acc = (bias ? bias[n] : 0.0f) + __riscv_vfmv_f_s_f32m1_f32(r);
            output[(size_t)m * N + n] = acc;
        }
    }
}
