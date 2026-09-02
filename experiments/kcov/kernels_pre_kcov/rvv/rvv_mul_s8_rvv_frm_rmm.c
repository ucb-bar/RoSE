/* source: curated */
/* algorithm: rvv_frm_rmm */
/* accuracy_class: bit_exact */
/* origin: hand-written, derived MECHANICALLY from
 *         rvv_add_s8_rvv_frm_rmm.c. The two references differ in exactly one
 *         operator --
 *
 *             add:  fout = (fa + fb) / scale_out
 *             mul:  fout = (fa * fb) / scale_out
 *
 *         -- so this file is that one, with `vfadd_vv` replaced by
 *         `vfmul_vv` and the intermediate renamed. Nothing else changed;
 *         `scripts/check_derived_kernel.py` re-derives it from the add file
 *         and diffs, so the claim is checkable rather than asserted.
 *
 *   WHY DERIVED RATHER THAN GENERATED. Five separate routes to "labelled
 *   vector, actually wrong" have been found in this tree, and the most
 *   recent -- cos_s8's table built in float where the reference computes in
 *   double -- passed the structural gate AND a numeric check, because at
 *   int8-in/int8-out the two precisions agree on most inputs. When an op's
 *   reference is one operator away from a kernel already verified
 *   bit-exact on the board, substituting the operator removes the whole
 *   class of question.
 *
 *   WHY THIS FILE EXISTS. mul_s8 carried no AlgorithmCandidate, so the
 *   curated probe had no (op, algorithm) pair to look for and the op ran
 *   the scalar reference inside a build labelled rvv_x60. Measured on the
 *   K1, attn_block at SEQ=8:
 *
 *       mul_s8   0.0415 ms   19.3% of the block, the largest remaining
 *                            reference-kernel share in it
 *
 *   It is the RoPE rotation -- the elementwise product that consumes the
 *   sin/cos tables the two curated LUT kernels next to it produce. Those
 *   two were vectorised first and this one was not, which is how it became
 *   the largest share: it did not get slower, its neighbours got faster.
 *
 *   THE ROUNDING MODE. The reference ends in `(int32_t)roundf(fout)`.
 *   roundf is round-to-nearest, TIES AWAY FROM ZERO; vfcvt.x.f rounds by
 *   `frm`, and frm=RMM is exactly that mode. So the conversion is not an
 *   approximation of roundf, it IS roundf. Read the add_s8 header for the
 *   tie sweep that established this on the board, and for why frm is
 *   toggled around the conversion only and not held across the arithmetic.
 *
 *   NOTE ON THE OP SPEC. `MUL_S8.semantics` used to tell a generator "use
 *   roundf to match numpy.round (banker's rounding compatible)", which is
 *   backwards -- roundf is ties-away and numpy.round is ties-to-even, and
 *   they disagree on every exact .5. Corrected there in the same commit as
 *   this file; a kernel generated against the old text could have picked
 *   frm=RNE and been wrong only on ties.
 *
 *   The arithmetic is four separate operations (vfmul, vfmul, vfmul,
 *   vfdiv) and NOT contracted: folding a*b into an fma with the divide, or
 *   turning the divide into a multiply by a reciprocal, removes an
 *   intermediate rounding the reference performs.
 *
 *   VTYPE. Every width-domain transition is named explicitly. Checked with
 *   scripts/check_rvv_vtype.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* frm helpers, guarded: every curated kernel body is concatenated into one
 * kernels.c, so a second file defining the same static inline is a
 * redefinition error rather than a duplicate symbol at link time. */
#ifndef MB_RVV_FRM_HELPERS_
#define MB_RVV_FRM_HELPERS_
/* frm encodings, from the unprivileged spec: 0 RNE, 1 RTZ, 2 RDN, 3 RUP,
 * 4 RMM. Only RMM matters here; the rest are named so the constant is not
 * a bare 4. */
#define MB_FRM_RNE 0
#define MB_FRM_RMM 4

static inline unsigned mb_frm_read(void)
{
    unsigned r;
    __asm__ volatile("frrm %0" : "=r"(r));
    return r;
}

static inline void mb_frm_write(unsigned mode)
{
    __asm__ volatile("fsrm %0" :: "r"(mode));
}
#endif /* MB_RVV_FRM_HELPERS_ */

void kernel_mul_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    const unsigned frm_saved = mb_frm_read();
    int i = 0;

    while (i < n) {
        const size_t n_elem = (size_t)(n - i);
        /* Element count in its own variable, handed to every
         * width. Chaining `vsetvl_e16m4(vl)` on a previous
         * vsetvl's result is miscompiled by this GCC: it passes an
         * ADDRESS register as the AVL operand, vl saturates to
         * VLMAX, and the vl-preserving forms carry that to the
         * store. See rvv_cat2_c1_s8_direct.c for the disassembly
         * and the guard-page proof. Only bites when the count is
         * not a whole multiple of VLMAX, i.e. on a partial tail. */
        size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
        vint8m1_t va8 = __riscv_vle8_v_i8m1(a + i, vl8);
        vint8m1_t vb8 = __riscv_vle8_v_i8m1(b + i, vl8);

        /* Into the 32-bit domain explicitly before the 4x widen: vsext.vf4
         * under SEW=8 would imply a 2-bit source and is an illegal
         * instruction, which is a SIGILL on the board and not a build
         * error. The element COUNT is unchanged -- e8m1 and e32m4 hold the
         * same number of elements. */
        size_t vl = __riscv_vsetvl_e32m4(n_elem);
        vint32m4_t va32 = __riscv_vsext_vf4_i32m4(va8, vl);
        vint32m4_t vb32 = __riscv_vsext_vf4_i32m4(vb8, vl);

        /* The reference's four float32 operations, in its order, each one
         * separately rounded. The caller's frm (RNE) is in force here. This
         * is the ONLY paragraph that differs from the add_s8 kernel: vfmul
         * where that one has vfadd. */
        vfloat32m4_t vfa = __riscv_vfcvt_f_x_v_f32m4(va32, vl);
        vfloat32m4_t vfb = __riscv_vfcvt_f_x_v_f32m4(vb32, vl);
        vfa = __riscv_vfmul_vf_f32m4(vfa, scale_a, vl);
        vfb = __riscv_vfmul_vf_f32m4(vfb, scale_b, vl);
        vfloat32m4_t vprod = __riscv_vfmul_vv_f32m4(vfa, vfb, vl);
        vfloat32m4_t vout = __riscv_vfdiv_vf_f32m4(vprod, scale_out, vl);

        /* roundf(), spelled as a rounding mode. */
        mb_frm_write(MB_FRM_RMM);
        vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vout, vl);
        mb_frm_write(frm_saved);

        vi = __riscv_vmax_vx_i32m4(vi, activation_min, vl);
        vi = __riscv_vmin_vx_i32m4(vi, activation_max, vl);

        /* Step back down through the widths explicitly, same reason as the
         * widen above; each narrowing names its DESTINATION width. */
        size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
        vint16m2_t vi16 = __riscv_vncvt_x_x_w_i16m2(vi, vl16);
        size_t vlo8 = __riscv_vsetvl_e8m1(n_elem);
        vint8m1_t vi8 = __riscv_vncvt_x_x_w_i8m1(vi16, vlo8);
        __riscv_vse8_v_i8m1(output + i, vi8, vlo8);

        i += (int)vl8;
    }
}
