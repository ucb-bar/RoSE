/* source: curated */
/* algorithm: gemmini_blocked_tb32 */
/* accuracy_class: bit_exact */
/* origin: LIFTED, unmodified in structure, from the input-transpose phase of
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c -- the non-windowed
 * `else` arm of the NCHW -> NHWC walk that fills ws_input (the `const int TB
 * = 32` nest). This file is code motion, not new tuning: the blocking factor,
 * the loop order and the pointer arithmetic are byte-for-byte the same
 * decisions that kernel already measured. `elem_t` becomes `int8_t` (the two
 * are the same type under Gemmini's int8 config, and dropping the typedef is
 * what lets this file compile without gemmini.h), `IC` becomes `C`, and
 * `IH*IW` becomes `HW`.
 *
 * Why blocked at all. The read side walks one channel plane contiguously; the
 * write side walks with stride C. Un-blocked, a full plane sweep of HW bytes
 * evicts the whole write window before the next channel reuses it, so every
 * channel pays a cold miss on every output line. Blocking the spatial index at
 * TB=32 bounds the live write window to 32*C bytes -- 4 KB at C=128, inside
 * Rocket's L1D -- so the C channels of one 32-position block all hit. The
 * source kernel's comment at :551-558 records that removing this blocking cost
 * conv0 46%.
 *
 * TB=32 is also, not by accident, exactly VLMAX for e8m1 at Saturn's
 * VLEN=256, so the rvv_strided sibling of this kernel is the same nest with
 * the innermost loop replaced by one vector op. That is the comparison stage 1
 * exists to make: this file is the scalar floor, and it is the floor on the
 * hart type that HAS no vector unit (harts 0/1 are Rocket + Q31 Gemmini with
 * no Saturn -- RoSEConfigs.scala SatGemQuadHeteroTacitConfig).
 *
 * Pure scalar C. Nothing here touches the Gemmini RoCC: a permutation moves
 * bytes and does no arithmetic, so there is no accumulator for mvout to
 * requantize and nothing for the systolic array to do. It lives under
 * kernels/gemmini/ because that is the backend whose harts will run it, not
 * because it uses the accelerator. */

void kernel_nchw_to_nhwc_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const int TB = 32;
    const int HW = H * W;

    for (int n = 0; n < N; n++) {
        const int8_t *inb = input  + (size_t)n * C * HW;
        int8_t       *ob  = output + (size_t)n * HW * C;
        for (int p0 = 0; p0 < HW; p0 += TB) {
            int pn = HW - p0 < TB ? HW - p0 : TB;
            for (int c0 = 0; c0 < C; c0 += TB) {
                int cn = C - c0 < TB ? C - c0 : TB;
                for (int c = 0; c < cn; c++) {
                    const int8_t *s = inb + (size_t)(c0 + c) * HW + p0;
                    int8_t       *d = ob  + (size_t)p0 * C + (c0 + c);
                    for (int p = 0; p < pn; p++)
                        d[(size_t)p * C] = s[p];
                }
            }
        }
    }
}
