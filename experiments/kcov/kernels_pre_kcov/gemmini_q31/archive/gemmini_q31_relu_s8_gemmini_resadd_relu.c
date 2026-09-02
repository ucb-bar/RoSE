/* source: curated */
/* algorithm: gemmini_resadd_relu */
/* accuracy_class: numeric_drift */
/* ARCHIVED 2026-08-28: this kernel declared accuracy_class=bit_exact but
 * is NOT bit-exact — isolation-tested on dronet (this kernel alone as the
 * only curated swap, every other op forced to scalar reference_impl):
 * max_abs_err=17 (experiments/kernel_opt_log.jsonl id 1100). Same root
 * cause as the maxpool2d_s8/gemmini_tiled_conv_pool kernel archived
 * alongside this file: the C_scale=ACC_SCALE_IDENTITY "no rescale"
 * passthrough this kernel relies on (tiled_resadd_auto's A*1 + 0*0, then
 * mvout-relu) goes through the SAME fixed-point ACC_SCALE round unit
 * conv2d_s8's HW path uses, which is not a true identity for this
 * Q31Ws32x32Acc config — so relu(A[i]) picks up ~1 LSB of noise per
 * element even though a max()-with-0 threshold has nothing to round.
 * dronet's one relu_s8 call feeds 2048 elements straight into linear1/
 * linear2 (M=1,K=2048,N=1 dot products), so per-element noise on many of
 * those inputs is exactly the kind of thing that shows up amplified at
 * the output. Relocated to archive/ (not deleted) so ModelBlaster's flat
 * curated-kernel lookup never selects it; kernels/gemmini_q31/ and
 * kernels/gemmini_q31_rvv/ both use the RVV/scalar `direct` threshold
 * kernel instead, which is exact (max(0,x) has no rounding to get
 * wrong) and also faster (kernel_opt_log id 801: 543 cycles vs this
 * kernel's 10,414).
 *
 * origin: relu(int8) via gemmini's tiled_resadd_auto with B=zeros and
 *         relu=true. Computes:
 *           C[i] = sat_int8(max(0, A[i]*1 + 0*0) * 1)
 *                = sat_int8(max(0, A[i]))
 *                = relu(A[i])
 *         The actual add is into a static zero buffer, so semantically
 *         a no-op accumulation; the only useful work is the mvout
 *         requantize-with-relu tail. Benefits over the scalar loop:
 *         streams DIM-wide chunks through gemmini's mvin/mvout pipeline,
 *         saturates the integer rounding+clamp unit instead of the
 *         scalar ALU.
 *
 *         Same chunking workaround as the add_s8 kernel: tiled_resadd
 *         crashes for I=1 with very large J, so split into ≤6272-element
 *         pieces. Static zero buffer is sized to ADD_CHUNK_MAX so we
 *         only need one allocation regardless of n.
 *
 *         Falls back to scalar for tiny n (per-call gemmini setup
 *         exceeds the body work).
 */
void kernel_relu_s8(const int8_t *input, int8_t *output, int n)
{
    enum { ADD_CHUNK_MAX = 6272 };
    static int8_t zero_buf[ADD_CHUNK_MAX] __attribute__((aligned(64)));

    if (n <= 0 || n < 256) {
        for (int i = 0; i < n; i++) {
            int8_t v = input[i];
            output[i] = v > 0 ? v : 0;
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    int remaining = n;
    int offset = 0;
    while (remaining > 0) {
        int chunk = remaining > ADD_CHUNK_MAX ? ADD_CHUNK_MAX : remaining;

        gemmini_flush(0);
        asm volatile("fence" ::: "memory");

        /* A_scale = 1.0 (passthrough), B_scale = 0 (zero contribution),
         * C_scale = ACC_SCALE_IDENTITY (no rescale), relu = true.
         * Computes C[i] = sat_int8(relu(A[i] * 1 + zero * 0))
         *               = sat_int8(relu(A[i])). */
        tiled_resadd_auto(
            /* I = */ 1, /* J = */ (size_t)chunk,
            (scale_t)1.0f, (scale_t)0.0f, ACC_SCALE_IDENTITY,
            input + offset, zero_buf, output + offset,
            /* relu = */ true,
            WS
        );

        gemmini_fence();
        gemmini_flush(0);

        offset    += chunk;
        remaining -= chunk;
    }
}
