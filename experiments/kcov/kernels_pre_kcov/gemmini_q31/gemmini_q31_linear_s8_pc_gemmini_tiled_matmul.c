/* source: curated */
/* PER-CHANNEL twin of gemmini_q31_linear_s8_gemmini_tiled_matmul.c.
 * Gemmini datapath unchanged (tiled_matmul_auto full_C=true -> raw int32);
 * only the CPU Q0.31 tail changes, indexing (multiplier, shift) by the
 * output-feature index n. Gemmini's ACC_SCALE is one scalar per matmul and
 * so cannot express per-channel scales -- the int32 drain is what makes
 * per-channel bit-exact here. */
/* algorithm: gemmini_tiled_matmul */
/* accuracy_class: bit_exact */
/* RELABELED 2026-08-28 (experiments/kernel_opt_log.jsonl): was previously
 * marked numeric_drift with a "≤1 LSB / layer" bound below -- stale, and
 * from a different ACC_SCALE-style construction than the code that's
 * actually here. This kernel drives tiled_matmul_auto with full_C=true,
 * which drains the RAW int32 accumulator (no HW mvout requantize at
 * all -- ACC_SCALE_IDENTITY is inert on this path, see the resadd
 * kernels' headers for why that combination is NOT a no-op when
 * full_C=false) and lets the CPU do the entire Q0.31 requantize in
 * int64, below. That is structurally identical to
 * conv2d_s8_gemmini_im2col_full_C.c (also full_C=true, also
 * accuracy_class: bit_exact, already isolation-measured at err=0), and
 * this kernel has now likewise been isolation-tested (this kernel alone
 * curated, every other op forced to scalar reference_impl) at err=0.
 * There is no float-scale HW shortcut anywhere in this path for drift to
 * come from.
 *
 * origin: tiled_matmul_auto(full_C=true) → raw int32 accumulator + scalar
 *         Q0.31 requantize on CPU.  The same precision contract as the
 *         conv2d_s8_gemmini_im2col_full_C path: bit-exact accumulator
 *         (gemmini's 32-bit acc with no float-scale shortcut) folded into
 *         the Q0.31 multiplier on CPU, exactly matching the reference
 *         formula's int64 rounding.
 *
 *         Computes C[M,N] = A[M,K] @ B[K,N] + bias[N] where:
 *           A = input  (row-major [M,K], stride_A = K)
 *           B = weight (physically row-major [N,K]; passed with
 *                       transpose_B=true so gemmini sees logical [K,N])
 *           C = output (row-major [M,N], stride_C = N)
 *         Bias is accumulated on the CPU side in the requantize loop —
 *         keeps the gemmini path uniform for biased / unbiased linears
 *         and avoids a stride_D=0 broadcast hassle for tiny M.
 *
 *         Saturn DIM=16: even degenerate (M=1, K=2048, N=1) uses one
 *         column of the systolic array (16-wide MAC line) per K-tile,
 *         giving DIM=16× ALU parallelism over scalar even though the
 *         other 15 columns are wasted.  For larger linears (M*N ≥ DIM²,
 *         K ≥ DIM) the speedup approaches the full DIM*DIM=256 ALU
 *         parallelism.
 *
 *         Limitations (caller falls back to scalar reference impl):
 *           * input_offset / filter_offset / output_offset must be 0
 *             (symmetric per-tensor int8; matches extract_int8 output).
 *           * output_shift in [0, 30] for the Q0.31 fold path.
 *           * M*N <= GEMMINI_LIN_ACC_MAX  (16*4096 ints = 256 KB).
 *           * total_out * K >= 256  (below this the gemmini per-call
 *             setup cost — mstatus, gemmini_flush, fence — exceeds the
 *             scalar dot-product cost).
 */
void kernel_linear_s8_pc(const int8_t *input, const int8_t *weight,
                         const int32_t *bias, int8_t *output,
                         int M, int K, int N,
                         int input_offset, int filter_offset, int output_offset,
                         const int32_t *output_multiplier,
                         const int32_t *output_shift,
                         int activation_min, int activation_max)
{
    enum { GEMMINI_LIN_ACC_MAX = 16 * 4096 };
    static int32_t ws_acc[GEMMINI_LIN_ACC_MAX] __attribute__((aligned(64)));

    int total_out = M * N;
    /* Per-channel: the Q0.31 fold path needs every channel's shift in
     * [0, 30]; one out-of-range channel sends the whole call to the
     * scalar fallback (same contract as the per-tensor kernel). */
    int shift_ok = 1;
    for (int n = 0; n < N; n++) {
        if (output_shift[n] < 0 || output_shift[n] > 30) { shift_ok = 0; break; }
    }
    if (input_offset != 0 || filter_offset != 0 || output_offset != 0
            || !shift_ok
            || (size_t)(M * N) > GEMMINI_LIN_ACC_MAX
            || M <= 0 || K <= 0 || N <= 0
            || total_out * K < 256) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int32_t acc = bias ? bias[n] : 0;
                for (int k = 0; k < K; k++) {
                    int32_t in_v = (int32_t)input[m * K + k] + input_offset;
                    int32_t w_v  = (int32_t)weight[n * K + k] + filter_offset;
                    acc += in_v * w_v;
                }
                const int32_t q_mult  = output_multiplier[n];
                const int32_t q_shift = output_shift[n];
                int64_t prod = (int64_t)acc * (int64_t)q_mult;
                prod = (prod + (1LL << 30)) >> 31;
                int32_t scaled = (int32_t)prod;
                if (q_shift > 0) {
                    int32_t round = (1 << (q_shift - 1));
                    scaled = (scaled + round) >> q_shift;
                } else if (q_shift < 0) {
                    scaled = scaled << (-q_shift);
                }
                scaled += output_offset;
                if (scaled < activation_min) scaled = activation_min;
                if (scaled > activation_max) scaled = activation_max;
                output[m * N + n] = (int8_t)scaled;
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Drain CPU stores before gemmini mvin reads input/weight. */
    asm volatile("fence" ::: "memory");

    /* C[M,N] = A[M,K] @ B[K,N] (transpose_B sees weight as logical [K,N]
     * from physical [N,K]). full_C=true asks for raw int32 accumulator
     * output (we requantize on CPU below with the Q0.31 formula matching
     * the conv path). D=NULL since we fold bias into the post-pass —
     * keeps the gemmini path identical for biased / unbiased linears. */
    tiled_matmul_auto(
        (size_t)M, (size_t)N, (size_t)K,
        input, weight,
        NULL, (void *)ws_acc,
        (size_t)K, (size_t)K, (size_t)N, (size_t)N,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
        false,            /* repeating_bias */
        false, true,      /* transpose_A=false, transpose_B=true */
        true, false,      /* full_C=true → raw int32 out, low_D=false */
        0, WS
    );

    /* Wait for gemmini DMA writes to ws_acc to reach memory. */
    gemmini_fence();
    gemmini_flush(0);

    /* Scalar Q0.31 requantize: int32 acc + bias → int8 output.
     * Same formula as conv2d's fallback / im2col path. */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = ws_acc[m * N + n] + (bias ? bias[n] : 0);
            const int32_t q_mult  = output_multiplier[n];
            const int32_t q_shift = output_shift[n];
            int64_t prod = (int64_t)acc * (int64_t)q_mult;
            prod = (prod + ((int64_t)1 << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (q_shift > 0) {
                scaled = (int32_t)(((int64_t)scaled
                    + ((int64_t)1 << (q_shift - 1))) >> q_shift);
            } else if (q_shift < 0) {
                scaled <<= (-q_shift);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[m * N + n] = (int8_t)scaled;
        }
    }
}
