/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: native half-precision RVV relu_f16 -- one vfmax.vf against +0.0 at
 * eew=16, LMUL=8. No arithmetic, so nothing to round: vfmax reproduces the
 * reference's `v > 0 ? v : 0` exactly, including -0.0 (vfmax(-0.0, +0.0)
 * returns +0.0, and the reference's comparison is false for -0.0 so it also
 * stores +0.0). */

void kernel_relu_f16(const _Float16 *input, _Float16 *output, int n) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e16m8(n - i);
        vfloat16m8_t v = __riscv_vle16_v_f16m8(input + i, vl);
        __riscv_vse16_v_f16m8(output + i,
                              __riscv_vfmax_vf_f16m8(v, (_Float16)0.0f, vl),
                              vl);
    }
}
