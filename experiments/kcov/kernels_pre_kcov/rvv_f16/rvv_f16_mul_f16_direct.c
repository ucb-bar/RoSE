/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: native half-precision RVV mul_f16 -- one vfmul.vv at eew=16.
 *
 * This target is -march=rv64gcv_zfh_zvfh, so the vector unit does fp16
 * arithmetic natively and there is no reason to widen: LMUL=8 at eew=16 gives
 * twice the lanes per instruction that an f32 path would.
 *
 * Bit-exactness against the scalar reference, which computes
 * (_Float16)((float)a * (float)b): the exact product of two fp16 values needs
 * at most 22 significand bits, so it is representable in fp32 with no
 * rounding; the reference therefore rounds exactly once, on the fp32 -> fp16
 * store. A native fp16 multiply also rounds the exact product exactly once.
 * Same value, one instruction instead of three. */

void kernel_mul_f16(const _Float16 *a, const _Float16 *b,
                    _Float16 *output, int n) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e16m8(n - i);
        vfloat16m8_t va = __riscv_vle16_v_f16m8(a + i, vl);
        vfloat16m8_t vb = __riscv_vle16_v_f16m8(b + i, vl);
        __riscv_vse16_v_f16m8(output + i,
                              __riscv_vfmul_vv_f16m8(va, vb, vl), vl);
    }
}
