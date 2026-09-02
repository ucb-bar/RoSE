/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 mean-squared-error loss (KernelBench 94,
   mean((a-b)^2)). Elementwise difference and square via vfsub/vfmul. The
   flat n-element sum uses a Kahan-compensated vfloat32m8 vector accumulator
   (VL parallel partial sums, per-lane running compensation) to keep the
   reduction accurate over the ~1e7 terms that a plain fp32 running sum would
   drift on; the VL lane sums are then tree-reduced with vfredusum and a
   short scalar tail finishes the ragged end. Order differs from the scalar
   double reference -> numeric_drift. Scaled by 1/n. */
#include <riscv_vector.h>

void kernel_mse_loss(const float *a, const float *b, float *output, int n) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    vfloat32m8_t comp = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    int i = 0;
    for (; i + (int)vlmax <= n; i += (int)vlmax) {
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vlmax);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vlmax);
        vfloat32m8_t d = __riscv_vfsub_vv_f32m8(va, vb, vlmax);
        vfloat32m8_t x = __riscv_vfmul_vv_f32m8(d, d, vlmax);
        /* Kahan: y = x - comp; t = sum + y; comp = (t - sum) - y; sum = t */
        vfloat32m8_t y = __riscv_vfsub_vv_f32m8(x, comp, vlmax);
        vfloat32m8_t t = __riscv_vfadd_vv_f32m8(sum, y, vlmax);
        comp = __riscv_vfsub_vv_f32m8(
            __riscv_vfsub_vv_f32m8(t, sum, vlmax), y, vlmax);
        sum = t;
    }
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    float total = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(sum, z, vlmax));
    for (; i < n; i++) {
        float d = a[i] - b[i];
        total += d * d;
    }
    output[0] = total / (float)n;
}
