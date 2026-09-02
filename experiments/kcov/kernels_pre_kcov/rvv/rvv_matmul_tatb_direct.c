/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 matmul C=A.T@B.T, A stored [K,M], B stored [N,K],
   C[M,N]; C[m,n] = sum_k A[k,m]*B[n,k].

   SATURN-SAFE REWRITE: the previous form (vfmul_vv + vfredusum tree-reduction over
   K) is CORRECT on spike (VLEN=128) but MISCOMPUTES on FireSim/Saturn V256D128
   (VLEN=256). The vfredusum (m8 reduction) is the trigger; the scalar-broadcast
   vfmacc_vf pattern (matmul_ta/bmm/diag) is correct on Saturn. So we vectorize
   across N with vfmacc_vf broadcasting the A[k,m] scalar, and gather B[n..n+vl, k]
   via a strided load (stride K). No reduction, no vfmul_vv/vfmacc_vv.
   fp32 reorder => numeric_drift. */
#include <riscv_vector.h>
#include <stddef.h>

void kernel_matmul_tatb(const float *A, const float *B, float *C,
                        int M, int K, int N) {
    const ptrdiff_t n_stride_bytes = (ptrdiff_t)K * (ptrdiff_t)sizeof(float);
    for (int m = 0; m < M; m++) {
        float *Cr = C + (size_t)m * N;            /* C[m, :] */
        for (int n = 0; n < N; ) {
            size_t vl = __riscv_vsetvl_e32m8(N - n);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int k = 0; k < K; k++) {
                float a = A[(size_t)k * M + m];   /* scalar A[k,m] (A is [K,M]) */
                /* B[n+i, k] for i in [0,vl): base B[n*K+k], stride K elems. */
                const float *Bp = B + (size_t)n * K + k;
                vfloat32m8_t b = __riscv_vlse32_v_f32m8(Bp, n_stride_bytes, vl);
                acc = __riscv_vfmacc_vf_f32m8(acc, a, b, vl);
            }
            __riscv_vse32_v_f32m8(Cr + n, acc, vl);
            n += vl;
        }
    }
}
