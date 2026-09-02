/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 Smooth-L1 / Huber loss (F.smooth_l1_loss,
   reduction=mean; KernelBench 96). Per element d=|a-b|: 0.5*d^2/beta if
   d<beta else d-0.5*beta, selected lane-wise with a vmflt mask + vmerge. The
   flat n-element sum uses a Kahan-compensated vfloat32m8 vector accumulator
   (accurate over ~1e7 terms where a plain fp32 running sum drifts), a final
   vfredusum over the VL lanes, and a scalar tail. Order differs from the
   scalar double reference -> numeric_drift. Scaled by 1/n. */
#include <math.h>
#include <riscv_vector.h>

void kernel_huber_loss(const float *a, const float *b, float *output,
                       int n, float beta) {
    float half_inv_beta = 0.5f / beta;
    float half_beta = 0.5f * beta;
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    vfloat32m8_t comp = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    int i = 0;
    for (; i + (int)vlmax <= n; i += (int)vlmax) {
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vlmax);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vlmax);
        vfloat32m8_t d = __riscv_vfabs_v_f32m8(
            __riscv_vfsub_vv_f32m8(va, vb, vlmax), vlmax);
        vfloat32m8_t quad = __riscv_vfmul_vf_f32m8(
            __riscv_vfmul_vv_f32m8(d, d, vlmax), half_inv_beta, vlmax);
        vfloat32m8_t lin = __riscv_vfsub_vf_f32m8(d, half_beta, vlmax);
        vbool4_t small = __riscv_vmflt_vf_f32m8_b4(d, beta, vlmax);
        vfloat32m8_t x = __riscv_vmerge_vvm_f32m8(lin, quad, small, vlmax);
        /* Kahan */
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
        float d = fabsf(a[i] - b[i]);
        total += (d < beta) ? (half_inv_beta * d * d) : (d - half_beta);
    }
    output[0] = total / (float)n;
}
