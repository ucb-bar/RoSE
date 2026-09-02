/* source: curated */
/* algorithm: gemmini_blocked_tb32 */
/* accuracy_class: bit_exact */
/* origin: LIFTED from the output-transpose phase of
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c -- the NHWC -> NCHW
 * walk that drains ws_output back into the caller's NCHW buffer. Same
 * `const int TB = 32` nest, same loop order; the only changes are the ones
 * that make it standalone.
 *
 * Two things from the source are deliberately NOT carried over:
 *
 *   - the mb_gem_ohwin window (`obase` / `ostride`). That machinery exists so
 *     an OH-split tile can scatter its band straight into the PARENT tensor,
 *     which only makes sense while the transpose is welded to a conv the
 *     splitter is slicing. As a standalone dispatch this op is split by the
 *     ordinary graph-level mechanism, so `ostride` collapses to the plain
 *     plane stride HW and `obase` to `output + n*C*HW`. The generated code is
 *     identical to the source's non-windowed path, which is the path every
 *     non-OH conv already took.
 *   - the activation_max post-clamp. That is conv arithmetic, not layout.
 *
 * This direction is the more expensive one in the measured phase table:
 * 1,317,580 cycles against the input side's 852,788 over dronet's ten conv
 * dispatches (docs/IR_TENSOR_LAYOUT_DESIGN.md section 1), because the strided
 * side is now the READ and the contiguous side the WRITE. Same bytes, same
 * blocking, ~1.5x the cost -- which is itself a fact the cycles/byte curve
 * should confirm rather than assume.
 *
 * Pure scalar C; see the nchw_to_nhwc sibling for why a Gemmini-directory
 * kernel touches no RoCC instruction. */

void kernel_nhwc_to_nchw_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const int TB = 32;
    const int HW = H * W;

    for (int n = 0; n < N; n++) {
        int8_t       *outb = output + (size_t)n * C * HW;
        const int8_t *ob   = input  + (size_t)n * HW * C;
        for (int p0 = 0; p0 < HW; p0 += TB) {
            int pn = HW - p0 < TB ? HW - p0 : TB;
            for (int c0 = 0; c0 < C; c0 += TB) {
                int cn = C - c0 < TB ? C - c0 : TB;
                for (int c = 0; c < cn; c++) {
                    int8_t       *d = outb + (size_t)(c0 + c) * HW + p0;
                    const int8_t *s = ob   + (size_t)p0 * C + (c0 + c);
                    for (int p = 0; p < pn; p++)
                        d[p] = s[(size_t)p * C];
                }
            }
        }
    }
}
