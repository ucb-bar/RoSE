/* source: curated */
/* algorithm: direct */
/* origin: native half-precision RVV add_f16 -- one vfadd.vv at eew=16.
 *
 * Zfh + Zvfh target, so this is a native fp16 vector add at LMUL=8, not an
 * f32 emulation.
 *
 * Relationship to the scalar reference ((_Float16)((float)a + (float)b)): for
 * operands within ~13 binades of each other the exact sum is representable in
 * fp32, the reference rounds once, and a native fp16 add rounds the same
 * value once -- identical. Only when the exponents are more than ~13 apart can
 * the fp32 intermediate itself round, making the reference a double rounding;
 * in that regime the smaller operand is already far below the ulp of the
 * result, so the two answers differ by at most one fp16 ulp and only on exact
 * ties. Left at the default accuracy class for that reason rather than
 * claimed bit_exact. */

void kernel_add_f16(const _Float16 *a, const _Float16 *b,
                    _Float16 *output, int n) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e16m8(n - i);
        vfloat16m8_t va = __riscv_vle16_v_f16m8(a + i, vl);
        vfloat16m8_t vb = __riscv_vle16_v_f16m8(b + i, vl);
        __riscv_vse16_v_f16m8(output + i,
                              __riscv_vfadd_vv_f16m8(va, vb, vl), vl);
    }
}
