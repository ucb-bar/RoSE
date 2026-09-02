/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV pad_f16 -- vector zero-fill and vector row copy at eew=16.
 *
 * Pure data movement, so the only question is how many bytes move per
 * instruction. The reference leans on the C library: memset over the whole
 * output then a memcpy per input row. Zephyr's minimal libc has no vector
 * string routines, so both degrade to scalar loops -- measured at 23 cycles
 * per output element, for an operation that touches each element twice.
 *
 * Here the fill is vse16 of a splat and the row copy is vle16/vse16 at LMUL=8,
 * and the fill is restricted to the border rows and columns instead of
 * blanketing the plane, so interior elements are written once rather than
 * twice. Values are copied, never arithmetic, so this is bit-exact. */

void kernel_pad_f16(const _Float16 *input, _Float16 *output,
                    int N, int C, int IH, int IW,
                    int pad_left, int pad_right, int pad_top, int pad_bottom) {
    const int OH = IH + pad_top + pad_bottom;
    const int OW = IW + pad_left + pad_right;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const _Float16 *src = input + (size_t)(n * C + c) * IH * IW;
            _Float16 *dst = output + (size_t)(n * C + c) * OH * OW;

            for (int oh = 0; oh < OH; oh++) {
                _Float16 *orow = dst + (size_t)oh * OW;
                int ih = oh - pad_top;
                if (ih < 0 || ih >= IH) {
                    int i = 0;
                    size_t vl;
                    for (; i < OW; i += (int)vl) {
                        vl = __riscv_vsetvl_e16m8(OW - i);
                        __riscv_vse16_v_f16m8(
                            orow + i,
                            __riscv_vfmv_v_f_f16m8((_Float16)0.0f, vl), vl);
                    }
                    continue;
                }
                /* left border */
                {
                    int i = 0;
                    size_t vl;
                    for (; i < pad_left; i += (int)vl) {
                        vl = __riscv_vsetvl_e16m8(pad_left - i);
                        __riscv_vse16_v_f16m8(
                            orow + i,
                            __riscv_vfmv_v_f_f16m8((_Float16)0.0f, vl), vl);
                    }
                }
                /* body */
                {
                    const _Float16 *irow = src + (size_t)ih * IW;
                    _Float16 *o = orow + pad_left;
                    int i = 0;
                    size_t vl;
                    for (; i < IW; i += (int)vl) {
                        vl = __riscv_vsetvl_e16m8(IW - i);
                        __riscv_vse16_v_f16m8(
                            o + i, __riscv_vle16_v_f16m8(irow + i, vl), vl);
                    }
                }
                /* right border */
                {
                    _Float16 *o = orow + pad_left + IW;
                    int i = 0;
                    size_t vl;
                    for (; i < pad_right; i += (int)vl) {
                        vl = __riscv_vsetvl_e16m8(pad_right - i);
                        __riscv_vse16_v_f16m8(
                            o + i,
                            __riscv_vfmv_v_f_f16m8((_Float16)0.0f, vl), vl);
                    }
                }
            }
        }
    }
}
