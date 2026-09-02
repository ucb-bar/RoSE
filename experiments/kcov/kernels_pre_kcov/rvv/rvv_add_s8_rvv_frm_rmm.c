/* source: curated */
/* algorithm: rvv_frm_rmm */
/* accuracy_class: bit_exact */
/* origin: hand-written. The add_s8 reference expression, unchanged, issued
 *         on the vector unit -- including its rounding mode.
 *
 *   WHY THIS FILE EXISTS. add_s8 had one AlgorithmCandidate and it was
 *   gemmini-affined, so on an RVV target the curated probe had no
 *   (op, algorithm) pair to look for and every residual add ran the scalar
 *   reference inside a build labelled rvv_x60. Measured on the K1: 3.6 ms
 *   of yolov8_nano's 229.2 ms and 0.6 ms of DroNet's 10.3 ms -- 1.6% and
 *   6.0%, small in the profile only because the fused convolutions next to
 *   them had just got 20x faster.
 *
 *   THE ROUNDING MODE, WHICH IS THE WHOLE POINT. The reference ends in
 *   `(int32_t)roundf(fout)`. roundf is round-to-nearest, TIES AWAY FROM
 *   ZERO, and the previous curated RVV work declined to vectorise anything
 *   ending that way on the grounds that vfcvt has no such mode. That is
 *   true of vfcvt's *default* behaviour and not of the instruction: vfcvt.x.f
 *   rounds by `frm`, and frm=RMM (encoding 4, "round to nearest, ties to
 *   Max Magnitude") is exactly ties-away-from-zero. So the conversion is
 *   not an approximation of roundf, it IS roundf, and the kernel is
 *   bit-exact by construction rather than by tolerance.
 *
 *   Verified separately from the kernel before being relied on: a 16-value
 *   sweep of exact .5 ties either side of zero, vfcvt under frm=RMM against
 *   scalar roundf on the board, zero mismatches.
 *
 *   WHY frm IS TOGGLED PER BLOCK RATHER THAN HELD. frm also governs
 *   vfmul/vfadd/vfdiv. Leaving it at RMM for the whole kernel would round
 *   the dequantize, the sum and the divide to ties-away as well, where C's
 *   float arithmetic rounds to nearest-even -- a silent 1-LSB drift in
 *   exactly the stage that is supposed to be exact. So the arithmetic runs
 *   under the caller's frm and only the integer conversion is bracketed.
 *   Two csrw per 32 outputs, against a vfdiv that costs far more.
 *
 *   The arithmetic is written as three separate operations (vfmul, vfmul,
 *   vfadd, vfdiv) and NOT as vfmacc: contracting a*b+c into an fma removes
 *   the intermediate rounding the reference performs. Same reason the
 *   division is a division and not a multiply by a precomputed reciprocal.
 *
 *   VTYPE. Every width-domain transition is named explicitly
 *   (__riscv_vsetvl_e8m1 for the loads, e32m4 for the extend and the float
 *   body, e16m2/e8m1 stepping back down for the store). GCC 13.2 does not
 *   carry vtype across these on its own -- that is the vfmv.v.f-under-SEW=8
 *   SIGILL rvv_batchnorm2d_s8_direct.c documents. Checked with
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

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
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
         * separately rounded. The caller's frm (RNE) is in force here. */
        vfloat32m4_t vfa = __riscv_vfcvt_f_x_v_f32m4(va32, vl);
        vfloat32m4_t vfb = __riscv_vfcvt_f_x_v_f32m4(vb32, vl);
        vfa = __riscv_vfmul_vf_f32m4(vfa, scale_a, vl);
        vfb = __riscv_vfmul_vf_f32m4(vfb, scale_b, vl);
        vfloat32m4_t vsum = __riscv_vfadd_vv_f32m4(vfa, vfb, vl);
        vfloat32m4_t vout = __riscv_vfdiv_vf_f32m4(vsum, scale_out, vl);

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
