/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: native half-precision RVV mul_c1_f16 (per-channel gate x plane).
 *
 * The gate is one fp16 per channel, so each (n, c) plane is a scalar-times-
 * vector multiply: vfmul.vf at eew=16, LMUL=8, with the gate broadcast from
 * the scalar half register the Zfh extension provides.
 *
 * Bit-exact for the same reason as rvv_f16_mul_f16_direct.c: the exact
 * product of two fp16 values fits in fp32, so the reference's
 * (_Float16)(g * (float)x) rounds exactly once, and so does the native fp16
 * multiply. */

void kernel_mul_c1_f16(const _Float16 *gate, const _Float16 *x,
                       _Float16 *output, int N, int C, int HW) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const _Float16 g = gate[c];
            const size_t base = (size_t)(n * C + c) * HW;
            const _Float16 *xp = x + base;
            _Float16 *op = output + base;
            int i = 0;
            size_t vl;
            for (; i < HW; i += (int)vl) {
                vl = __riscv_vsetvl_e16m8(HW - i);
                vfloat16m8_t v = __riscv_vle16_v_f16m8(xp + i, vl);
                __riscv_vse16_v_f16m8(op + i,
                                      __riscv_vfmul_vf_f16m8(v, g, vl), vl);
            }
        }
    }
}
