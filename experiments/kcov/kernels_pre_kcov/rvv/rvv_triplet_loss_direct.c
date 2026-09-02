/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 triplet margin loss (nn.TripletMarginLoss, p=2,
   reduction=mean; KernelBench 99). Per sample b over F features, the two
   squared L2 distances ||a-p||^2 and ||a-n||^2 are accumulated with vfsub/
   vfmul + vfredusum (numeric_drift vs the scalar double reference), then
   loss_b = max(0, sqrt(dp)-sqrt(dn)+margin); output is mean over B. */
#include <math.h>
#include <riscv_vector.h>

void kernel_triplet_loss(const float *anchor, const float *pos,
                         const float *neg, float *output,
                         int B, int F, float margin) {
    float total = 0.0f, kc = 0.0f; /* scalar Kahan over the B samples */
    for (int b = 0; b < B; b++) {
        const float *a = anchor + (long)b * F;
        const float *p = pos + (long)b * F;
        const float *nn = neg + (long)b * F;
        vfloat32m1_t accp = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t accn = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int f = 0; f < F; ) {
            size_t vl = __riscv_vsetvl_e32m8(F - f);
            vfloat32m8_t va = __riscv_vle32_v_f32m8(a + f, vl);
            vfloat32m8_t dp = __riscv_vfsub_vv_f32m8(
                va, __riscv_vle32_v_f32m8(p + f, vl), vl);
            vfloat32m8_t dn = __riscv_vfsub_vv_f32m8(
                va, __riscv_vle32_v_f32m8(nn + f, vl), vl);
            accp = __riscv_vfredusum_vs_f32m8_f32m1(
                __riscv_vfmul_vv_f32m8(dp, dp, vl), accp, vl);
            accn = __riscv_vfredusum_vs_f32m8_f32m1(
                __riscv_vfmul_vv_f32m8(dn, dn, vl), accn, vl);
            f += (int)vl;
        }
        float sp = __riscv_vfmv_f_s_f32m1_f32(accp);
        float sn = __riscv_vfmv_f_s_f32m1_f32(accn);
        float loss = sqrtf(sp) - sqrtf(sn) + margin;
        if (loss > 0.0f) {
            float yk = loss - kc;
            float tk = total + yk;
            kc = (tk - total) - yk;
            total = tk;
        }
    }
    output[0] = total / (float)B;
}
