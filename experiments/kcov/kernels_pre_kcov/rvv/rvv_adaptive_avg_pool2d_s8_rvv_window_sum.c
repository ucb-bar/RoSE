/* source: curated */
/* algorithm: rvv_window_sum */
/* accuracy_class: bit_exact */
/* origin: hand-written.
 *
 *   WHY THIS ONE IS EASY, AND WHY THAT IS WORTH SAYING. layernorm and softmax
 *   needed care because their reductions are FLOATING-POINT and sequential, so
 *   reduction order is part of the answer. This accumulator is `int32_t acc`
 *   over int8 inputs: integer addition is associative, so any reduction shape
 *   gives the identical sum. The vector version is not an approximation of the
 *   scalar one, it is the same arithmetic.
 *
 *   OVERFLOW IS NOT A RISK AND IS NOT BEING TRUSTED TO LUCK. The widest
 *   possible window is IH*IW; at |int8| <= 128 the sum stays inside int32 for
 *   any plane up to 16.7M elements, which no feature map here approaches. The
 *   accumulation is done at 32 bits for that reason rather than the 16 a
 *   narrower reduction would allow -- vwredsum into i16 would overflow on a
 *   104-element window of large values, and the failure would be silent and
 *   data-dependent.
 *
 *   WHAT IT IS FOR. `adaptive_avg_pool2d` is one of the ops ViNT needs and no
 *   model in this tree currently emits; the extractor only supports
 *   output_size=(1,1), which is the global-average-pool case, so the window is
 *   the whole plane and the inner loop is a long contiguous run. That is the
 *   shape this is written for; the general case still works, just with shorter
 *   vectors.
 *
 *   THE FLOAT TAIL IS THE REFERENCE'S, unchanged and scalar: one
 *   `roundf(mean / scale_out)` per OUTPUT, and there are OH*OW of those
 *   against IH*IW inputs. Vectorising it would mean reproducing roundf's
 *   ties-away rounding for a handful of elements -- possible (frm=RMM, see
 *   rvv_add_s8_rvv_frm_rmm.c) and not worth the surface, since at
 *   output_size=(1,1) there is exactly one such element per channel.
 *
 *   VTYPE. e8m1 loads widened to e32m4 for the accumulator; the AVL handed to
 *   each vsetvl is the ELEMENT COUNT, never a previous vsetvl's result.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Below this a row is not worth a vector setup; the scalar loop is the
 * reference's own. */
#ifndef MB_AAP_VEC_MIN
#define MB_AAP_VEC_MIN 8
#endif

void kernel_adaptive_avg_pool2d_s8(const int8_t *input, int8_t *output,
                                   int N, int C, int IH, int IW,
                                   int OH, int OW,
                                   float scale_in, float scale_out,
                                   int activation_min, int activation_max)
{
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *plane = input + (size_t)(n * C + c) * IH * IW;
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
                    if (run >= MB_AAP_VEC_MIN) {
                        /* One i32 accumulator vector across every row of the
                         * window, reduced once at the end. Integer, so the
                         * shape of the reduction cannot change the value. */
                        size_t vlmax = __riscv_vsetvlmax_e32m4();
                        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax);
                        for (int ih = ih0; ih < ih1; ih++) {
                            const int8_t *row = plane + (size_t)ih * IW + iw0;
                            int k = 0;
                            while (k < run) {
                                const size_t n_elem = (size_t)(run - k);
                                size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
                                vint8m1_t v8 = __riscv_vle8_v_i8m1(row + k, vl8);
                                size_t vl = __riscv_vsetvl_e32m4(n_elem);
                                vint32m4_t v32 = __riscv_vsext_vf4_i32m4(v8, vl);
                                vacc = __riscv_vadd_vv_i32m4(vacc, v32, vl);
                                k += (int)vl8;
                            }
                        }
                        vint32m1_t red = __riscv_vmv_s_x_i32m1(0, 1);
                        red = __riscv_vredsum_vs_i32m4_i32m1(vacc, red, vlmax);
                        acc = __riscv_vmv_x_s_i32m1_i32(red);
                    } else {
                        for (int ih = ih0; ih < ih1; ih++)
                            for (int iw = iw0; iw < iw1; iw++)
                                acc += (int32_t)plane[(size_t)ih * IW + iw];
                    }

                    /* The reference's float tail, verbatim. */
                    float mean = (float)acc * scale_in / (float)win;
                    int32_t v = (int32_t)roundf(mean / scale_out);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    output[((size_t)(n * C + c) * OH + oh) * OW + ow] = (int8_t)v;
                }
            }
        }
    }
}
