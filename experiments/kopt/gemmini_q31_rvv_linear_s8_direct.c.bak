/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV linear_s8. */

void kernel_linear_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int M, int K, int N,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            size_t vl;
            vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4());
            const int8_t *in_row  = input + m * K;
            const int8_t *w_row   = weight + n * K;
            int k = 0;
            for (; k + 2*(__riscv_vsetvlmax_e8m1()) <= K; k += 2*(__riscv_vsetvlmax_e8m1())) {
                vl = __riscv_vsetvl_e8m1(K - k);
                vint8m1_t va = __riscv_vle8_v_i8m1(in_row + k, vl);
                vint8m1_t vb = __riscv_vle8_v_i8m1(w_row + k, vl);
                vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
                vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
                va = __riscv_vle8_v_i8m1(in_row + k + vl, vl);
                vb = __riscv_vle8_v_i8m1(w_row + k + vl, vl);
                prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
                vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
            }
            for (; k < K; k += vl) {
                vl = __riscv_vsetvl_e8m1(K - k);
                vint8m1_t va = __riscv_vle8_v_i8m1(in_row + k, vl);
                vint8m1_t vb = __riscv_vle8_v_i8m1(w_row + k, vl);
                vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
                vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
            }
            vint32m1_t vinit = __riscv_vmv_s_x_i32m1(0, 1);
            vint32m1_t vsum  = __riscv_vredsum_vs_i32m4_i32m1(vacc, vinit, __riscv_vsetvlmax_e32m4());
            int32_t acc = __riscv_vmv_x_s_i32m1_i32(vsum);
            if (bias) acc += bias[n];
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0) {
                int32_t round = (1 << (output_shift - 1));
                scaled = (scaled + round) >> output_shift;
            } else {
                scaled = scaled << (-output_shift);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[m * N + n] = (int8_t)scaled;
        }
    }
}