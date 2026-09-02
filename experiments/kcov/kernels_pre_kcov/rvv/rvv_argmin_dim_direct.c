/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 argmin over the middle axis of a logically
   [outer, reduce, inner] tensor; int64 index output. Mirror of argmax with
   a strict `v < best` compare, so the FIRST minimum index wins — matching
   torch.argmin on CPU. e32m4 value/int32-index widened to i64m8 at store. */
#include <stdint.h>
#include <riscv_vector.h>

void kernel_argmin_dim(const float *input, int64_t *output,
                       int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        int64_t *out = output + (long)o * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m4(inner - i);
            vfloat32m4_t best = __riscv_vle32_v_f32m4(base + i, vl);
            vint32m4_t bidx = __riscv_vmv_v_x_i32m4(0, vl);
            for (int r = 1; r < reduce; r++) {
                vfloat32m4_t v =
                    __riscv_vle32_v_f32m4(base + (long)r * inner + i, vl);
                vbool8_t lt = __riscv_vmflt_vv_f32m4_b8(v, best, vl);
                best = __riscv_vmerge_vvm_f32m4(best, v, lt, vl);
                bidx = __riscv_vmerge_vxm_i32m4(bidx, r, lt, vl);
            }
            vint64m8_t bidx64 = __riscv_vsext_vf2_i64m8(bidx, vl);
            /* int64_t is `long long` in this picolibc, but the RVV i64
               store builtin expects `long *`; both are 64-bit on LP64. */
            __riscv_vse64_v_i64m8((long *)(out + i), bidx64, vl);
            i += (int)vl;
        }
    }
}
