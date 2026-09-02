/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV int8 -> _Float16 dequantize. The reference computes
 *   out[i] = (_Float16)((float)in[i] * scale) -- one fp32 multiply, then ONE
 *   rounding at the store. This reproduces exactly that chain, so it is
 *   bit-exact rather than numeric_drift: widen int8 -> int16 -> fp32
 *   (vsext.vf2 + vfwcvt.f.x.v), scale in fp32 (vfmul.vf), narrow to fp16
 *   (vfncvt.f.f.w). Multiplying in fp16 instead would round twice and drift.
 *
 *   The per-width vsetvl calls are deliberate and load-bearing: GCC 13.2 does
 *   not carry vtype across a kernel's width changes and will issue a widening
 *   convert under the previous SEW, which is illegal and SIGILLs on the first
 *   dispatch (see kernels/rvv/rvv_cat3_c1_s8_direct.c and
 *   scripts/check_rvv_vtype.py, which gates on it at build time). Element
 *   COUNT is identical across e8m1 / e16m2 / e32m4 because EMUL scales with
 *   SEW, so naming the widths changes no arithmetic. */

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* These intrinsics need RVV intrinsics API >= v0.12 (the fp16 vector types
 * vfloat16m*_t and vfmacc.vv / vfredusum.vs / vfncvt at SEW=16). GCC 13.2
 * reports __riscv_v_intrinsic == 11000 and has none of them: it accepts the
 * calls as implicit declarations, emits warnings rather than errors, and the
 * build then fails at LINK with undefined __riscv_vle16_v_f16m4 -- or worse,
 * would pass the wrong types if a declaration existed. Say so here instead.
 * Fix: build kernels.c with a GCC >= 14 of the same lp64d ABI via
 * MODELBLASTER_KERNEL_CC (see KERNEL_CC in harness_linux/Makefile). */
#if !defined(__riscv_v_intrinsic) || __riscv_v_intrinsic < 12000
#error "curated RVV fp16 kernel needs RVV intrinsics >= v0.12 (GCC >= 14); set MODELBLASTER_KERNEL_CC"
#endif


void kernel_cast_i8_to_f16(const int8_t *in, _Float16 *out,
                           int n, float scale) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        /* Element count in its own variable, handed to every width.
         * Chaining `vsetvl_e16m2(vl)` on a previous vsetvl's result is
         * miscompiled by GCC 14.3.0: it passes an ADDRESS register as the AVL
         * operand, vl saturates to VLMAX, and the vl-preserving forms carry
         * that to the store, which then writes VLMAX elements on a partial
         * tail. See rvv_cat2_c1_s8_direct.c for the disassembly and the
         * guard-page proof, and scripts/check_rvv_avl.py for the detector. */
        const size_t n_elem = (size_t)(n - i);
        vl = __riscv_vsetvl_e8m1(n_elem);
        size_t vl8 = vl;
        size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
        size_t vl32 = __riscv_vsetvl_e32m4(n_elem);
        vint8m1_t v8 = __riscv_vle8_v_i8m1(in + i, vl8);
        vint16m2_t v16 = __riscv_vsext_vf2_i16m2(v8, vl16);
        vfloat32m4_t vf = __riscv_vfwcvt_f_x_v_f32m4(v16, vl32);
        vf = __riscv_vfmul_vf_f32m4(vf, scale, vl32);
        vfloat16m2_t vh = __riscv_vfncvt_f_f_w_f16m2(vf, vl16);
        __riscv_vse16_v_f16m2(out + i, vh, vl16);
    }
}
