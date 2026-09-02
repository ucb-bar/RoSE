/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 eval-mode BatchNorm2d. scale[c] and bias[c]
   are per-channel scalars already fused (scale=gamma/sqrt(var+eps),
   bias=beta-mean*scale). For each (n,c) the H*W activation plane is a plain
   affine map output = s*x + b, so vectorize the contiguous inner plane and
   apply vfmul_vf then vfadd_vf — two rounded ops, exactly matching the
   scalar reference `s*input[idx]+b` element for element. */
#include <riscv_vector.h>

void kernel_batchnorm2d(const float *input, const float *scale,
                        const float *bias, float *output,
                        int N, int C, int H, int W) {
    long plane = (long)H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            long off = ((long)n * C + c) * plane;
            const float *in = input + off;
            float *out = output + off;
            for (long i = 0; i < plane; ) {
                size_t vl = __riscv_vsetvl_e32m8(plane - i);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(in + i, vl);
                v = __riscv_vfmul_vf_f32m8(v, s, vl);
                v = __riscv_vfadd_vf_f32m8(v, b, vl);
                __riscv_vse32_v_f32m8(out + i, v, vl);
                i += (long)vl;
            }
        }
    }
}
