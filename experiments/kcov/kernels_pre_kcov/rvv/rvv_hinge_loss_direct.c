/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 hinge loss (KernelBench 100,
   mean(clamp(1 - pred*targ, min=0))). targ broadcasts over pred with period
   targ_len (targ_len==n -> no broadcast; targ_len==1 -> per-batch scalar; an
   inner-dim divisor otherwise). Each period is vectorized: 1-pred*targ via
   vfmul+vfrsub, clamped with vfmax(.,0), summed with vfredusum
   (numeric_drift vs the scalar double reference), scaled by 1/n. */
#include <riscv_vector.h>

void kernel_hinge_loss(const float *pred, const float *targ, float *output,
                       int n, int targ_len) {
    vfloat32m1_t vacc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    if (targ_len == 1) {
        float t = targ[0];
        for (int i = 0; i < n; ) {
            size_t vl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(pred + i, vl);
            vfloat32m8_t h = __riscv_vfrsub_vf_f32m8(
                __riscv_vfmul_vf_f32m8(vp, t, vl), 1.0f, vl); /* 1 - p*t */
            h = __riscv_vfmax_vf_f32m8(h, 0.0f, vl);
            vacc = __riscv_vfredusum_vs_f32m8_f32m1(h, vacc, vl);
            i += (int)vl;
        }
        output[0] = __riscv_vfmv_f_s_f32m1_f32(vacc) / (float)n;
        return;
    }
    int rows = n / targ_len;
    for (int r = 0; r < rows; r++) {
        const float *p = pred + (long)r * targ_len;
        for (int j = 0; j < targ_len; ) {
            size_t vl = __riscv_vsetvl_e32m8(targ_len - j);
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(p + j, vl);
            vfloat32m8_t vt = __riscv_vle32_v_f32m8(targ + j, vl);
            vfloat32m8_t h = __riscv_vfrsub_vf_f32m8(
                __riscv_vfmul_vv_f32m8(vp, vt, vl), 1.0f, vl); /* 1 - p*t */
            h = __riscv_vfmax_vf_f32m8(h, 0.0f, vl);
            vacc = __riscv_vfredusum_vs_f32m8_f32m1(h, vacc, vl);
            j += (int)vl;
        }
    }
    /* leftover tail if targ_len does not divide n (targ index resets) */
    float extra = 0.0f;
    for (int i = rows * targ_len; i < n; i++) {
        float h = 1.0f - pred[i] * targ[i % targ_len];
        if (h > 0.0f) extra += h;
    }
    output[0] = (__riscv_vfmv_f_s_f32m1_f32(vacc) + extra) / (float)n;
}
