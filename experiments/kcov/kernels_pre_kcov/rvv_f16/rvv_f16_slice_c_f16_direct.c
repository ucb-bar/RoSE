/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV slice_c_f16 -- a channel slice is a contiguous run per channel,
 * so copy it with vle16/vse16 at LMUL=8 rather than per-channel memcpy calls
 * into a scalar libc. Copies only, so bit-exact. */

void kernel_slice_c_f16(const _Float16 *input, _Float16 *output,
                        int N, int IC, int C_start, int C_end,
                        int H, int W) {
    const int OC = C_end - C_start;
    const size_t HW = (size_t)H * W;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            const _Float16 *src =
                input + ((size_t)(n * IC + (C_start + oc)) * HW);
            _Float16 *dst = output + ((size_t)(n * OC + oc) * HW);
            size_t i = 0, vl;
            for (; i < HW; i += vl) {
                vl = __riscv_vsetvl_e16m8(HW - i);
                __riscv_vse16_v_f16m8(dst + i,
                                      __riscv_vle16_v_f16m8(src + i, vl), vl);
            }
        }
    }
}
