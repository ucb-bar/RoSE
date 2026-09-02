/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV adaptive_avg_pool2d_s8 -- vector integer window reduction,
 * scalar requantize.
 *
 * The window sum is int32 and therefore EXACT, so reassociating it across
 * vector lanes cannot change the answer: each window row widens i8 -> i16
 * (vwadd.vx with 0) and folds into a running i32 vector, and the lanes are
 * summed with one vwredsum at the end. The requantize is one operation per
 * OUTPUT, not per input element, so it stays scalar and character-for-
 * character identical to the reference -- which makes the whole kernel
 * bit-exact, not merely close.
 *
 * This matters most where ViNT actually uses the op: the squeeze-excite and
 * final global pools reduce a whole HxW plane to one value, so the reference's
 * per-element scalar loop is the entire cost. */

void kernel_adaptive_avg_pool2d_s8(const int8_t *input, int8_t *output,
                                   int N, int C, int IH, int IW,
                                   int OH, int OW,
                                   float scale_in, float scale_out,
                                   int activation_min, int activation_max) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *plane = input + (size_t)(n * C + c) * IH * IW;
            int8_t *op = output + (size_t)(n * C + c) * OH * OW;
            for (int oh = 0; oh < OH; oh++) {
                int ih0 = (oh * IH) / OH;
                int ih1 = ((oh + 1) * IH + OH - 1) / OH;
                if (ih1 > IH) ih1 = IH;
                for (int ow = 0; ow < OW; ow++) {
                    int iw0 = (ow * IW) / OW;
                    int iw1 = ((ow + 1) * IW + OW - 1) / OW;
                    if (iw1 > IW) iw1 = IW;
                    int win = (ih1 - ih0) * (iw1 - iw0);
                    if (win <= 0) win = 1;
                    const int run = iw1 - iw0;

                    int32_t acc = 0;
                    if (run > 0) {
                        for (int ih = ih0; ih < ih1; ih++) {
                            const int8_t *row = plane + (size_t)ih * IW + iw0;
                            int i = 0;
                            size_t vl;
                            for (; i < run; i += (int)vl) {
                                vl = __riscv_vsetvl_e8m2(run - i);
                                vint8m2_t v8 =
                                    __riscv_vle8_v_i8m2(row + i, vl);
                                vint16m4_t v16 =
                                    __riscv_vwadd_vx_i16m4(v8, 0, vl);
                                vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
                                vint32m1_t r =
                                    __riscv_vwredsum_vs_i16m4_i32m1(v16, z, vl);
                                acc += __riscv_vmv_x_s_i32m1_i32(r);
                            }
                        }
                    }
                    float mean = (float)acc * scale_in / (float)win;
                    int32_t v = (int32_t)roundf(mean / scale_out);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    op[(size_t)oh * OW + ow] = (int8_t)v;
                }
            }
        }
    }
}
