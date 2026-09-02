/* source: curated */
/* algorithm: rvv_memo_lut_gather */
/* accuracy_class: bit_exact */
/* origin: hand-written.
 *
 *   WHY THIS FILE EXISTS. elu_s8 carried no AlgorithmCandidate, so the
 *   curated probe had no (op, algorithm) pair to look for on any target and
 *   the op ran the scalar reference inside builds labelled rvv_x60 --
 *   39.6% of mlp_control, which is 0.04 ms of a 0.1 ms model. Small in
 *   absolute terms; it is here because "no vector kernel exists" and "a
 *   vector kernel would not help" are different findings and the profile
 *   could not tell them apart.
 *
 *   WHAT IS ACTUALLY EXPENSIVE. Not the width -- the expf. The reference
 *   calls it once per element, ~140 cycles each on this core, which is the
 *   whole cost of the op. Widening the datapath cannot help with that:
 *   there is no vector expf, and writing a polynomial one would give up
 *   bit-exactness for an op whose entire runtime is 0.04 ms.
 *
 *   WHAT DOES HELP. The input is int8, so ELU has at most 256 distinct
 *   outputs for a given (scale_in, scale_out, alpha) -- and a quantized
 *   activation tensor repeats values heavily, so far fewer than 256 in
 *   practice. So: one cheap pass marking which of the 256 bytes actually
 *   occur, expf only for those, then a vector indexed-load gather
 *   (vluxei8, the biased byte as its own offset into the table) for the
 *   output. Cost goes from n*expf to distinct*expf, and the elements that
 *   are not the first of their value cost a gather lane instead of a
 *   transcendental.
 *
 *   This is the memoized form of the LUT the op's own semantics text
 *   proposes, and it is better than the eager form: building all 256
 *   entries would cost 256 expf regardless of n, which LOSES for every
 *   shape in these models (n = 256, 128, 64). Building only the entries
 *   that are asked for never loses by more than the marking pass.
 *
 *   BIT-EXACT BY CONSTRUCTION. Each table entry is computed by the
 *   reference's own expression, on the scalar unit, in float32, with the
 *   same casts and the same roundf. The gather then reproduces bytes it
 *   did not compute. There is no arithmetic in the vector path at all, so
 *   there is no rounding mode to match.
 *
 *   THE SMALL-n GUARD. Below MB_ELU_MEMO_MIN the fixed cost of the 256-byte
 *   marking array outweighs the saved expf calls, and the reference
 *   expression runs per element instead -- the same guard, for the same
 *   reason, as the LUT break-even in the curated batchnorm kernels.
 *
 *   VTYPE. The vector path is a single 8-bit domain: e8m1 load, vxor, an
 *   unsigned reinterpret for the index, vluxei8, e8m1 store. No width
 *   transition to lose. Checked with scripts/check_rvv_vtype.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

/* Below this the 256-entry marking pass costs more than it saves. */
#ifndef MB_ELU_MEMO_MIN
#define MB_ELU_MEMO_MIN 32
#endif

/* Gather out[i] = table[input[i] + 128] with the biased byte used directly
 * as an offset. XOR with 0x80 is the bias: it maps -128 -> 0 and 127 -> 255
 * without leaving the 8-bit domain, so no widening is needed for the index.
 *
 * Guarded: every curated kernel body is concatenated into one kernels.c, so
 * a second file defining this static inline is a redefinition error. */
#ifndef MB_RVV_LUT_GATHER_S8_
#define MB_RVV_LUT_GATHER_S8_
static inline void mb_rvv_lut_gather_s8(const int8_t *input, int8_t *output,
                                        int n, const int8_t *table)
{
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(n - i));
        vint8m1_t vx = __riscv_vle8_v_i8m1(input + i, vl);
        vint8m1_t vb = __riscv_vxor_vx_i8m1(vx, (int8_t)-128, vl);
        vuint8m1_t vidx = __riscv_vreinterpret_v_i8m1_u8m1(vb);
        vint8m1_t vy = __riscv_vluxei8_v_i8m1(table, vidx, vl);
        __riscv_vse8_v_i8m1(output + i, vy, vl);
        i += (int)vl;
    }
}
#endif /* MB_RVV_LUT_GATHER_S8_ */

/* The reference expression, verbatim, for one input byte. */
static inline int8_t mb_elu_s8_one(int8_t x, float scale_in, float scale_out,
                                   int activation_min, int activation_max,
                                   float alpha)
{
    float f = (float)x * scale_in;
    float y = (f > 0.0f) ? f : alpha * (expf(f) - 1.0f);
    int32_t v = (int32_t)roundf(y / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_elu_s8(const int8_t *input, int8_t *output, int n,
                   float scale_in, float scale_out,
                   int activation_min, int activation_max, float alpha)
{
    if (n <= 0) return;
    if (n < MB_ELU_MEMO_MIN) {
        for (int i = 0; i < n; i++)
            output[i] = mb_elu_s8_one(input[i], scale_in, scale_out,
                                      activation_min, activation_max, alpha);
        return;
    }

    int8_t table[256];
    unsigned char seen[256];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i < n; i++)
        seen[(unsigned char)input[i] ^ 0x80u] = 1;
    for (int b = 0; b < 256; b++) {
        if (!seen[b]) continue;
        table[b] = mb_elu_s8_one((int8_t)(b - 128), scale_in, scale_out,
                                 activation_min, activation_max, alpha);
    }
    mb_rvv_lut_gather_s8(input, output, n, table);
}
