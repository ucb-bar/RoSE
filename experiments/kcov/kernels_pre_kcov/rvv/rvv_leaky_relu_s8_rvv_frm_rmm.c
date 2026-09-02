/* source: curated */
/* algorithm: rvv_frm_rmm */
/* accuracy_class: bit_exact */
/* origin: hand-written, same shape as rvv_add_s8_rvv_frm_rmm.c -- the
 *         reference expression issued on the vector unit, with roundf()
 *         spelled as frm=RMM so the result is bit-exact by construction
 *         rather than by tolerance. Read that file's header for why the
 *         rounding mode is the interesting part and why frm is toggled per
 *         block instead of held.
 *
 *   WHY THIS FILE EXISTS. leaky_relu_s8 carried no AlgorithmCandidate at
 *   all, so the curated probe had nothing to look for on any target and
 *   the op ran the scalar reference inside builds labelled rvv_x60.
 *
 *   HOW BIG. Small, and worth saying so: 0.9% of vitfly_frontend and
 *   two dispatches of 64 and 16 elements in vitfly_lstm. At n=16 a vector
 *   kernel is one masked pass and the call overhead dominates either way.
 *   This is here for coverage and for the models that use the op at width,
 *   not because it moves either of these two.
 *
 *   THE SELECT. The reference branches on `f > 0.0f` -- on the dequantized
 *   float, not on the int8 -- so the mask is computed there too, with
 *   vmfgt against +0.0f. For scale_in > 0 the two agree, but taking the
 *   comparison where the reference takes it removes the question.
 *
 *   THE AVL IS THE ELEMENT COUNT, NEVER A PREVIOUS vsetvl'S RESULT, AND
 *   THAT IS NOT STYLE. Written as `__riscv_vsetvl_e32m4(vl8)` -- chaining
 *   one vsetvl on another's return value -- GCC 14.3 substitutes an
 *   unrelated register for the AVL. Measured in the avgpool kernel, which
 *   had the same shape: its second vsetvl was issued with the OUTER LOOP
 *   BOUND as its AVL, vl came out 5 where the row is 11
 *   wide, the `vsetvli zero,zero` forms carried that 5 down to the store,
 *   and six of every eleven outputs were never written at all. Not a
 *   rounding difference: max_abs_err=68 against the reference, and silent.
 *
 *   It was correct under GCC 13.2 -- the compiler these kernels were
 *   verified on, and still what the default `CROSS` points at. 13.2 has its
 *   own bug (it reorders a vsetvl across a widening op and the binary
 *   SIGILLs), which is why 14.3 is mandatory. So these kernels moved from a
 *   compiler that crashes loudly to one that answers wrongly, and the only
 *   form that is correct under both is the element count, every time.
 *
 *   VTYPE. e8m1 loads, an explicit e32m4 window for the extend and the
 *   float body, e16m2/e8m1 stepping back down for the store. Checked with
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

void kernel_leaky_relu_s8(const int8_t *input, int8_t *output, int n,
                          float scale_in, float scale_out,
                          int activation_min, int activation_max,
                          float negative_slope)
{
    const unsigned frm_saved = mb_frm_read();
    int i = 0;

    while (i < n) {
        const size_t n_elem = (size_t)(n - i);
        size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
        vint8m1_t vx8 = __riscv_vle8_v_i8m1(input + i, vl8);

        size_t vl = __riscv_vsetvl_e32m4(n_elem);
        vint32m4_t vx32 = __riscv_vsext_vf4_i32m4(vx8, vl);
        vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vx32, vl);
        vf = __riscv_vfmul_vf_f32m4(vf, scale_in, vl);

        /* y = (f > 0) ? f : negative_slope * f */
        vbool8_t pos = __riscv_vmfgt_vf_f32m4_b8(vf, 0.0f, vl);
        vfloat32m4_t vneg = __riscv_vfmul_vf_f32m4(vf, negative_slope, vl);
        vfloat32m4_t vy = __riscv_vmerge_vvm_f32m4(vneg, vf, pos, vl);

        vfloat32m4_t vq = __riscv_vfdiv_vf_f32m4(vy, scale_out, vl);

        mb_frm_write(MB_FRM_RMM);
        vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vq, vl);
        mb_frm_write(frm_saved);

        vi = __riscv_vmax_vx_i32m4(vi, activation_min, vl);
        vi = __riscv_vmin_vx_i32m4(vi, activation_max, vl);

        size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
        vint16m2_t vi16 = __riscv_vncvt_x_x_w_i16m2(vi, vl16);
        size_t vlo8 = __riscv_vsetvl_e8m1(n_elem);
        vint8m1_t vi8 = __riscv_vncvt_x_x_w_i8m1(vi16, vlo8);
        __riscv_vse8_v_i8m1(output + i, vi8, vlo8);

        i += (int)vl8;
    }
}
