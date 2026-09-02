/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV relu6_s8. Clamp int8 activations to [0, qmax],
   where qmax is the int8 encoding of the 6.0 ReLU6 ceiling. */

void kernel_relu6_s8(const int8_t *input, int8_t *output, int n, int qmax) {
    int8_t hi = (int8_t)qmax;
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
        v = __riscv_vmax_vx_i8m8(v, 0, vl);    /* clamp negatives to 0 */
        v = __riscv_vmin_vx_i8m8(v, hi, vl);   /* clamp at the 6.0 ceiling */
        __riscv_vse8_v_i8m8(output + i, v, vl);
    }
}
