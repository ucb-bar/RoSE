/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 GroupNorm over an NCHW tensor (InstanceNorm is
   the G==C case). For a given (sample n, group g) the C/G channels are
   consecutive in c, so their HW blocks form ONE contiguous run of
   cnt=(C/G)*HW floats.

   NOTE (Saturn V256D128 fp64 bug): the fp32->fp64 widening-reduction path
   miscomputes on the Saturn vector unit, so all statistics are kept in pure
   fp32.  A one-pass fp32 E[x^2]-E[x]^2 catastrophically cancels
   on large / large-magnitude groups, so we use a numerically-stable TWO-PASS
   fp32 algorithm instead:
     pass 1: mean = sum(x)/cnt
     pass 2: var  = sum((x-mean)^2)/cnt     (centering removes the large
             common component, so the fp32 accumulation of the centered
             squares does not catastrophically cancel)
   Each pass accumulates with a plain fp32 reduction (vfredusum f32m8->f32m1,
   which is CONFIRMED correct on Saturn); no fp64 vector ops are used.
   inv=1/sqrt(var+eps) (scalar), then each channel's HW plane is
   normalized+affined in fp32:  out = (x-mean)*(gamma[c]*inv) + beta[c].
   Reduction order differs from the scalar loop -> numeric_drift. */
#include <math.h>
#include <riscv_vector.h>

void kernel_group_norm(const float *input, const float *gamma,
                       const float *beta, float *output,
                       int N, int C, int G, int HW, float eps) {
    int cpg = C / G;                 /* channels per group */
    long cnt = (long)cpg * HW;       /* elements per (sample, group) */
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            const float *blk = input + ((long)n * C + (long)g * cpg) * HW;

            /* pass 1: mean = sum(x)/cnt, fp32 reduction (no fp64) */
            vfloat32m1_t vs = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            for (long i = 0; i < cnt; ) {
                size_t vl = __riscv_vsetvl_e32m8(cnt - i);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(blk + i, vl);
                vs = __riscv_vfredusum_vs_f32m8_f32m1(v, vs, vl);
                i += (long)vl;
            }
            float mean = __riscv_vfmv_f_s_f32m1_f32(vs) / (float)cnt;

            /* pass 2: var = sum((x-mean)^2)/cnt, fp32 reduction (no fp64) */
            vfloat32m1_t vq = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            for (long i = 0; i < cnt; ) {
                size_t vl = __riscv_vsetvl_e32m8(cnt - i);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(blk + i, vl);
                vfloat32m8_t d = __riscv_vfsub_vf_f32m8(v, mean, vl);
                vfloat32m8_t sq = __riscv_vfmul_vv_f32m8(d, d, vl);
                vq = __riscv_vfredusum_vs_f32m8_f32m1(sq, vq, vl);
                i += (long)vl;
            }
            float var = __riscv_vfmv_f_s_f32m1_f32(vq) / (float)cnt;
            float inv = 1.0f / sqrtf(var + eps);

            /* apply per channel: out = (x-mean)*(gamma*inv) + beta */
            for (int cc = 0; cc < cpg; cc++) {
                int c = g * cpg + cc;
                float f = gamma[c] * inv;
                float bt = beta[c];
                const float *p = input + ((long)n * C + c) * HW;
                float *o = output + ((long)n * C + c) * HW;
                for (int i = 0; i < HW; ) {
                    size_t vl = __riscv_vsetvl_e32m8(HW - i);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(p + i, vl);
                    vfloat32m8_t d = __riscv_vfsub_vf_f32m8(v, mean, vl);
                    d = __riscv_vfmul_vf_f32m8(d, f, vl);
                    d = __riscv_vfadd_vf_f32m8(d, bt, vl);
                    __riscv_vse32_v_f32m8(o + i, d, vl);
                    i += (int)vl;
                }
            }
        }
    }
}
