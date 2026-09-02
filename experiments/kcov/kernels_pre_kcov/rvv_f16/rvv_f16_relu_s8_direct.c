/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV relu_s8. */

void kernel_relu_s8(const int8_t *input, int8_t *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
        v = __riscv_vmax_vx_i8m8(v, 0, vl);
        __riscv_vse8_v_i8m8(output + i, v, vl);
    }
}