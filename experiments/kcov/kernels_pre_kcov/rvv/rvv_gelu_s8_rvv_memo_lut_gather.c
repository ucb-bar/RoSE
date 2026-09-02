/* source: curated */
/* algorithm: rvv_memo_lut_gather */
/* accuracy_class: bit_exact */
/* origin: hand-written, derived mechanically from
 *         rvv_sigmoid_s8_rvv_memo_lut_gather.c -- the same file with the
 *         reference's gelu expression substituted for its sigmoid one and
 *         nothing else changed. Read rvv_elu_s8_rvv_memo_lut_gather.c for
 *         why the memoized table is the right shape for this op family.
 *
 *   WHY DERIVED RATHER THAN GENERATED. cos_s8 was generated twice and came
 *   back both times computing its table in FLOAT where that reference
 *   computes in double -- code that passed the structural gate, passed a
 *   numeric check, and was bit-exact only as a property of the test data.
 *   The structural gate cannot express precision, so for an op whose
 *   reference is one expression away from an already board-verified kernel,
 *   substituting the expression is the cheaper and safer route.
 *
 *   (gelu_s8's reference is float32 throughout -- erff, roundf, float
 *   literals -- so that particular hazard does not arise here. The
 *   discipline is kept anyway: the expression below is a copy of the
 *   reference's, not a re-derivation of GELU.)
 *
 *   WHAT THIS IS WORTH, and it is the largest single one in the tree.
 *   Measured on the K1, ffn_block at SEQ=128, D_FF=1024, so n=131072:
 *
 *       rvv_x60   gelu_s8  12.87 ms   29.7% of the block
 *       ime_x60   gelu_s8  12.90 ms   42.2% of the block
 *
 *   The second number is the reason this could not be left. The IME result
 *   -- the MAC unit beating the vector unit on FFN linears -- was measured
 *   against a block whose largest component was a SCALAR erff loop, and a
 *   block-level speedup quoted over that is partly credit for work the
 *   accelerator did not do. Vectorising gelu makes the IME advantage
 *   SMALLER and the claim honest; that is the direction a correction is
 *   supposed to move a number you like.
 *
 *   WHAT IS EXPENSIVE. erff, not the datapath. One call per element, and
 *   there is no vector erff -- a polynomial one would trade bit-exactness
 *   for the whole reason the op is quantized in the first place. The input
 *   is int8, so for a fixed (scale_in, scale_out, clamp) the op has at most
 *   256 distinct outputs; mark which bytes occur, call erff only for those,
 *   and gather the rest with vluxei8.
 *
 *   THE MARKING PASS, AT THIS n. For n=131072 an eagerly-built 256-entry
 *   table would skip the marking pass and cost 256 erff unconditionally --
 *   cheaper here than the O(n) marking scan. It is not what this file does:
 *   the memo form is what is board-verified for four other ops, the
 *   difference is a scalar pass over 128 KB against 12.87 ms of erff, and
 *   splitting the kernel into a small-n and a large-n path would double the
 *   surface that has to stay bit-exact for a fraction of a percent. If the
 *   measurement shows the scan dominating, that is the time to add it.
 *
 *   VTYPE. Single 8-bit domain in the vector path -- e8m1 load, vxor,
 *   unsigned reinterpret, vluxei8, e8m1 store. No width transition to lose.
 *   Checked with scripts/check_rvv_vtype.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

/* Below this the table costs more than the erff it saves, so the reference
 * expression runs per element instead.
 *
 * MEASURED, not copied from the sigmoid kernel's 32. The table has 256
 * entries and the marking pass is a 256-byte memset plus one store per
 * element, so at n below the table size a build cannot amortize even in the
 * best case. At n=64 (norm_block's gelu) the memo form measured 0.69x and
 * 1.66x on the two data regimes -- i.e. it can LOSE there. At n=131072
 * (ffn_block) it measured 39.88x and 29.53x. 256 is the size of the thing
 * being built, which is the right place to put the guard: above it the memo
 * can win, below it it is paying a fixed cost out of a smaller budget than
 * the cost itself.
 */
#ifndef MB_GELU_MEMO_MIN
#define MB_GELU_MEMO_MIN 256
#endif

/* See rvv_elu_s8_rvv_memo_lut_gather.c; guarded because every curated body
 * lands in the same kernels.c and a second definition is an error rather
 * than a duplicate symbol at link time. */
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

/* The reference expression, verbatim, for one input byte. Every constant,
 * cast and rounding call is the reference's own -- kInvSqrt2 to the same
 * eleven digits, erff not erf, roundf not round, float not double. */
static inline int8_t mb_gelu_s8_one(int8_t x, float scale_in, float scale_out,
                                    int activation_min, int activation_max)
{
    const float kInvSqrt2 = 0.70710678118f;
    float f = (float)x * scale_in;
    float y = 0.5f * f * (1.0f + erff(f * kInvSqrt2));
    int32_t v = (int32_t)roundf(y / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_gelu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max)
{
    if (n <= 0) return;
    if (n < MB_GELU_MEMO_MIN) {
        for (int i = 0; i < n; i++)
            output[i] = mb_gelu_s8_one(input[i], scale_in, scale_out,
                                       activation_min, activation_max);
        return;
    }

    int8_t table[256];
    unsigned char seen[256];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i < n; i++)
        seen[(unsigned char)input[i] ^ 0x80u] = 1;
    for (int b = 0; b < 256; b++) {
        if (!seen[b]) continue;
        table[b] = mb_gelu_s8_one((int8_t)(b - 128), scale_in, scale_out,
                                  activation_min, activation_max);
    }
    mb_rvv_lut_gather_s8(input, output, n, table);
}
