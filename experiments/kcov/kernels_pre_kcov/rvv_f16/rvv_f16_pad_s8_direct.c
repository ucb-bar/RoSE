/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV pad_s8 -- vector splat for the border, vector copy for the body.
 *
 * The scalar reference is the worst shape this operator can take: a per-output
 * -element loop with two range tests and a five-term index expression, even
 * though every element is either a constant or a straight copy from a
 * contiguous run. It measured 16.8 cycles per output element on ViNT's stem
 * pad, where the whole op is 1.36 M elements.
 *
 * Splitting each output row into (left border, body, right border) makes all
 * three runs contiguous: vse8 of a splat for the borders, vle8/vse8 at LMUL=8
 * for the body. Copies only, so bit-exact. */

void kernel_pad_s8(const int8_t *input, int8_t *output,
                   int N, int C, int IH, int IW,
                   int pad_left, int pad_right, int pad_top, int pad_bottom,
                   int pad_value) {
    const int OH = IH + pad_top + pad_bottom;
    const int OW = IW + pad_left + pad_right;
    const int8_t pv = (int8_t)pad_value;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *src = input + (size_t)(n * C + c) * IH * IW;
            int8_t *dst = output + (size_t)(n * C + c) * OH * OW;

            for (int oh = 0; oh < OH; oh++) {
                int8_t *orow = dst + (size_t)oh * OW;
                int ih = oh - pad_top;
                if (ih < 0 || ih >= IH) {
                    int i = 0;
                    size_t vl;
                    for (; i < OW; i += (int)vl) {
                        vl = __riscv_vsetvl_e8m8(OW - i);
                        __riscv_vse8_v_i8m8(orow + i,
                                            __riscv_vmv_v_x_i8m8(pv, vl), vl);
                    }
                    continue;
                }
                {
                    int i = 0;
                    size_t vl;
                    for (; i < pad_left; i += (int)vl) {
                        vl = __riscv_vsetvl_e8m8(pad_left - i);
                        __riscv_vse8_v_i8m8(orow + i,
                                            __riscv_vmv_v_x_i8m8(pv, vl), vl);
                    }
                }
                {
                    const int8_t *irow = src + (size_t)ih * IW;
                    int8_t *o = orow + pad_left;
                    int i = 0;
                    size_t vl;
                    for (; i < IW; i += (int)vl) {
                        vl = __riscv_vsetvl_e8m8(IW - i);
                        __riscv_vse8_v_i8m8(o + i,
                                            __riscv_vle8_v_i8m8(irow + i, vl),
                                            vl);
                    }
                }
                {
                    int8_t *o = orow + pad_left + IW;
                    int i = 0;
                    size_t vl;
                    for (; i < pad_right; i += (int)vl) {
                        vl = __riscv_vsetvl_e8m8(pad_right - i);
                        __riscv_vse8_v_i8m8(o + i,
                                            __riscv_vmv_v_x_i8m8(pv, vl), vl);
                    }
                }
            }
        }
    }
}
