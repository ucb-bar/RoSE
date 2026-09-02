/* source: curated */
/* algorithm: rvv_strided */
/* accuracy_class: bit_exact */
/* origin: the gemmini_blocked_tb32 nest with its innermost loop replaced by
 * one vector memory operation. Contiguous vle8 on the read side (a channel
 * plane is HW contiguous bytes in NCHW), vsse8 with byte-stride C on the write
 * side (consecutive spatial positions are C apart in NHWC).
 *
 * Why e8m1 and not a wider LMUL. LMUL selects the spatial block size, and the
 * spatial block size IS the blocking factor the scalar kernel tuned to 32:
 * for one window of `vl` positions the live write region spans vl*C bytes, so
 * vl=32 at C=128 keeps it at 4 KB and vl=128 (m4) would make it 16 KB and
 * thrash the L1D that the blocking exists to fit inside. At Saturn's VLEN=256
 * e8m1 gives VLMAX=32 -- exactly the scalar TB. So this file and its scalar
 * twin have the SAME memory access order and the same working set, and the
 * only difference measured between them is scalar loop vs vector unit. That is
 * the whole point: docs/IR_TENSOR_LAYOUT_DESIGN.md section 5.2a claims the
 * measured ~9.7 cycles/byte is "a transpose on the one hart type that cannot
 * vectorise it", and this is the kernel that tests the claim.
 *
 * No channel blocking. The scalar kernel blocks C at TB as well, which bounds
 * the number of live read streams; here the read of plane c is a single vle8
 * that retires before plane c+1 issues, so there is only ever one read stream
 * regardless of C, and the extra loop level would only add address arithmetic.
 *
 * Honest caveat, and the reason the vsseg sibling exists: a strided store is
 * only a win if the vector unit coalesces it. An implementation that cracks
 * vsse8 into one memory operation per element does the same number of accesses
 * as the scalar loop and merely pays for the vector issue on top. Saturn's
 * store path (generators/saturn/src/main/scala/mem/) does have a compactor,
 * but its throughput on a stride this large is not something this file can
 * assert -- it is what the measurement decides. */

void kernel_nchw_to_nhwc_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const size_t HW = (size_t)H * (size_t)W;

    for (int n = 0; n < N; n++) {
        const int8_t *ib = input  + (size_t)n * (size_t)C * HW;
        int8_t       *ob = output + (size_t)n * HW * (size_t)C;

        for (size_t p0 = 0; p0 < HW; ) {
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);
            for (int c = 0; c < C; c++) {
                vint8m1_t v = __riscv_vle8_v_i8m1(
                        ib + (size_t)c * HW + p0, vl);
                __riscv_vsse8_v_i8m1(ob + p0 * (size_t)C + (size_t)c,
                                     (ptrdiff_t)C, v, vl);
            }
            p0 += vl;
        }
    }
}
