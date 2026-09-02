/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 matmul C=A@B.T, A[M,K] row-major, B[N,K] row-major,
   C[M,N]; C[m,n] = sum_k A[m,k]*B[n,k].

   SATURN-SAFE REWRITE: the previous dot-product form (vfmacc_vv over K into a
   vector accumulator, then vfredusum to a scalar) is CORRECT on spike (VLEN=128)
   but MISCOMPUTES on FireSim/Saturn V256D128 (VLEN=256) -- rel_err=1. The
   vfmacc_vv + vfredusum (m8 tree-reduction) pattern is the trigger; the scalar-
   broadcast vfmacc_vf pattern used by matmul_ta/bmm/diag is correct on Saturn.
   So here we vectorize across N (output columns) with vfmacc_vf broadcasting the
   A[m,k] scalar, and gather B[n..n+vl, k] via a strided load (stride K). No
   reduction, no vfmacc_vv. fp32 reorder => numeric_drift. */
#include <riscv_vector.h>
#include <stddef.h>

void kernel_matmul_tb(const float *A, const float *B, float *C,
                      int M, int K, int N) {
    const ptrdiff_t n_stride_bytes = (ptrdiff_t)K * (ptrdiff_t)sizeof(float);
    for (int m = 0; m < M; m++) {
        const float *Ar = A + (size_t)m * K;      /* A[m, :] */
        float *Cr = C + (size_t)m * N;            /* C[m, :] */
        for (int n = 0; n < N; ) {
            size_t vl = __riscv_vsetvl_e32m8(N - n);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int k = 0; k < K; k++) {
                float a = Ar[k];                  /* scalar A[m,k] */
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
