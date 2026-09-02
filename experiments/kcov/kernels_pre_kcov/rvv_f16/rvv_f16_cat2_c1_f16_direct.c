/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV cat2_c1_f16 -- the two source planes are contiguous runs, so
 * this is two vector copies per batch at eew=16 / LMUL=8 instead of two calls
 * into Zephyr's scalar minimal-libc memcpy. Copies only, so bit-exact. */

static inline void mb_cat2f16_copy(const _Float16 *src, _Float16 *dst,
                                   size_t n) {
    size_t i = 0, vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m8(n - i);
        __riscv_vse16_v_f16m8(dst + i, __riscv_vle16_v_f16m8(src + i, vl), vl);
    }
}

void kernel_cat2_c1_f16(const _Float16 *a, const _Float16 *b,
                        _Float16 *output,
                        int N, int H, int W, int Ca, int Cb) {
    const size_t HW = (size_t)H * W;
    const size_t Cout = (size_t)(Ca + Cb);
    for (int n = 0; n < N; n++) {
        mb_cat2f16_copy(a + (size_t)n * Ca * HW,
                        output + (size_t)n * Cout * HW, (size_t)Ca * HW);
        mb_cat2f16_copy(b + (size_t)n * Cb * HW,
                        output + (size_t)n * Cout * HW + (size_t)Ca * HW,
                        (size_t)Cb * HW);
    }
}
