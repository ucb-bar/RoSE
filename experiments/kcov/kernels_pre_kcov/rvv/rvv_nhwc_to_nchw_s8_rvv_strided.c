/* source: curated */
/* algorithm: rvv_strided */
/* accuracy_class: bit_exact */
/* origin: the exact mirror of rvv_nchw_to_nhwc_s8_rvv_strided.c -- vlse8 with
 * byte-stride C gathers one channel out of the interleaved source, vse8 writes
 * it contiguously into its NCHW plane. Same e8m1 spatial block for the same
 * working-set reason (see that file's header), so the pair is directly
 * comparable and the round-trip identity exercises both halves of the same
 * access pattern.
 *
 * This direction moves the strided access to the LOAD side. That is worth
 * measuring separately rather than assuming symmetry: the scalar phase data in
 * docs/IR_TENSOR_LAYOUT_DESIGN.md section 1 has the output transpose costing
 * 1.55x the input one over the same ten dispatches, so the two directions are
 * already known NOT to be symmetric on a core with no vector unit. Whether a
 * gather is cheaper or dearer than a scatter on Saturn is a separate question
 * with a separate answer. */

void kernel_nhwc_to_nchw_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const size_t HW = (size_t)H * (size_t)W;

    for (int n = 0; n < N; n++) {
        const int8_t *ib = input  + (size_t)n * HW * (size_t)C;
        int8_t       *ob = output + (size_t)n * (size_t)C * HW;

        for (size_t p0 = 0; p0 < HW; ) {
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);
            for (int c = 0; c < C; c++) {
                vint8m1_t v = __riscv_vlse8_v_i8m1(
                        ib + p0 * (size_t)C + (size_t)c, (ptrdiff_t)C, vl);
                __riscv_vse8_v_i8m1(ob + (size_t)c * HW + p0, v, vl);
            }
            p0 += vl;
        }
    }
}
