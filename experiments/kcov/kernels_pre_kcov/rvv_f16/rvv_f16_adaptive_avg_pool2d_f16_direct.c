/* source: curated */
/* algorithm: direct */
/* origin: RVV adaptive_avg_pool2d_f16 -- widening vector reduction over the
 * window (vfwredusum, fp16 elements into an fp32 accumulator), scalar divide.
 *
 * The elements stay fp16 in memory and are loaded at eew=16; only the
 * ACCUMULATOR is fp32, which is what the scalar reference does too (it sums
 * into a float). vfwredusum is the instruction that expresses exactly that,
 * so this is not an f32 emulation of an fp16 op -- it is the fp16 reduction
 * the ISA provides.
 *
 * One honest deviation: a vector reduction associates the additions
 * differently from the reference's row-major scalar loop, and fp32 addition is
 * not associative, so windows larger than a couple of elements can differ in
 * the last fp32 bit before the fp16 store. Left at the default accuracy class
 * for that reason. */

void kernel_adaptive_avg_pool2d_f16(const _Float16 *input, _Float16 *output,
                                    int N, int C, int IH, int IW,
                                    int OH, int OW) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const _Float16 *plane = input + (size_t)(n * C + c) * IH * IW;
            _Float16 *op = output + (size_t)(n * C + c) * OH * OW;
            for (int oh = 0; oh < OH; oh++) {
                int h0 = (oh * IH) / OH;
                int h1 = ((oh + 1) * IH + OH - 1) / OH;
                if (h1 > IH) h1 = IH;
                for (int ow = 0; ow < OW; ow++) {
                    int w0 = (ow * IW) / OW;
                    int w1 = ((ow + 1) * IW + OW - 1) / OW;
                    if (w1 > IW) w1 = IW;
                    const int run = w1 - w0;
                    int cnt = (h1 - h0) * run;
                    float sum = 0.0f;
                    for (int h = h0; h < h1; h++) {
                        const _Float16 *row = plane + (size_t)h * IW + w0;
                        int i = 0;
                        size_t vl;
                        for (; i < run; i += (int)vl) {
                            vl = __riscv_vsetvl_e16m4(run - i);
                            vfloat16m4_t v = __riscv_vle16_v_f16m4(row + i, vl);
                            vfloat32m1_t z =
                                __riscv_vfmv_v_f_f32m1(0.0f, 1);
                            vfloat32m1_t r =
                                __riscv_vfwredusum_vs_f16m4_f32m1(v, z, vl);
                            sum += __riscv_vfmv_f_s_f32m1_f32(r);
                        }
                    }
                    op[(size_t)oh * OW + ow] =
                        (_Float16)(sum / (float)(cnt > 0 ? cnt : 1));
                }
            }
        }
    }
}
