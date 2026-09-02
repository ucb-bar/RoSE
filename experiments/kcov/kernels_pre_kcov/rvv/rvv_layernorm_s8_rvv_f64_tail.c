/* source: curated */
/* algorithm: rvv_f64_tail */
/* accuracy_class: bit_exact */
/* origin: hand-written.
 *
 *   WHAT IS AND IS NOT VECTORISED HERE, AND WHY THAT IS THE WHOLE DESIGN.
 *   The reference makes three passes over each row:
 *
 *     1.  mu  += (double)row[k] * scale_in            sequential double sum
 *     2.  var += d*d, d = row[k]*scale_in - mu        sequential double sum
 *     3.  y = ((row[k]*scale_in - mu) * inv) * gamma[k] + beta[k]
 *         out[k] = clamp(round(y / scale_out))
 *
 *   Passes 1 and 2 STAY SCALAR. They are sequential floating-point
 *   accumulations, and a vector tree reduction adds the same terms in a
 *   different ORDER, which is a different number. `mu` and `inv` then feed
 *   every output in the row, so a few ULP there is not a rounding detail --
 *   it moves every element. RVV does have an ordered reduction (vfredosum)
 *   that would preserve the order exactly, but it serialises the add chain by
 *   construction, so it buys the loads and multiplies and not the adds; that
 *   is a larger kernel for a smaller and less certain win, and it can be
 *   added later against a measurement rather than a guess.
 *
 *   PASS 3 IS WHERE THE TIME IS, and it is embarrassingly parallel: every
 *   element is independent once mu and inv are known. It also carries the
 *   only DIVISION in the kernel, one per element, which is the most expensive
 *   thing in the loop after the transcendental-free arithmetic. That is what
 *   this vectorises.
 *
 *   EVERYTHING IN PASS 3 IS DONE IN DOUBLE, because the reference does. The
 *   int8 input is sign-extended 8x to i64 and converted to f64; gamma and
 *   beta are float32 in memory and are WIDENED to f64 exactly as
 *   `(double)gamma[k]` does. Doing this pass in float32 would be much faster
 *   and would not be this kernel: it would be a different answer, and the
 *   whole point of a curated kernel is that the scheduler's cost is for code
 *   that computes what the model computes.
 *
 *   round() IS TIES-AWAY-FROM-ZERO, spelled as frm=RMM around the f64->i64
 *   conversion. Same instrument as rvv_add_s8_rvv_frm_rmm.c; read that header
 *   for the tie sweep that established it on the board, and for why frm is
 *   toggled around the conversion only and never held across the arithmetic.
 *
 *   THE DIVIDE STAYS A DIVIDE. `y / scale_out` is not turned into a multiply
 *   by a precomputed reciprocal; that changes the rounding.
 *
 *   gamma and beta are each optional in the reference (NULL means 1.0 and 0.0).
 *   Both are handled, and the NULL cases skip the widening load rather than
 *   materialising a vector of constants.
 *
 *   VTYPE. The pass-3 body lives at e64m4; the int8 source for that element
 *   count is e8mf2 and the float32 source is e32m2, and each transition names
 *   its own width. The AVL handed to every vsetvl is the ELEMENT COUNT, never
 *   a previous vsetvl's result -- see scripts/check_rvv_avl.py for what that
 *   costs when it is not.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_RVV_FRM_HELPERS_
#define MB_RVV_FRM_HELPERS_
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

void kernel_layernorm_s8(const int8_t *input, const float *gamma,
                         const float *beta, int8_t *output,
                         int M, int K, float scale_in, float scale_out,
                         float eps, int activation_min, int activation_max)
{
    const unsigned frm_saved = mb_frm_read();

    for (int m = 0; m < M; m++) {
        const int8_t *row = input + (size_t)m * (size_t)K;
        int8_t *orow = output + (size_t)m * (size_t)K;

        /* ---- passes 1 and 2: the reference's own order, unchanged ------- */
        double mu = 0.0;
        for (int k = 0; k < K; k++) mu += (double)row[k] * (double)scale_in;
        mu /= (double)K;
        double var = 0.0;
        for (int k = 0; k < K; k++) {
            const double d = (double)row[k] * (double)scale_in - mu;
            var += d * d;
        }
        var /= (double)K;
        const double inv = 1.0 / sqrt(var + (double)eps);

        /* ---- pass 3, vectorised in f64 ---------------------------------- */
        int k = 0;
        while (k < K) {
            const size_t n_elem = (size_t)(K - k);

            size_t vl8 = __riscv_vsetvl_e8mf2(n_elem);
            vint8mf2_t v8 = __riscv_vle8_v_i8mf2(row + k, vl8);

            size_t vl = __riscv_vsetvl_e64m4(n_elem);
            vint64m4_t vi64 = __riscv_vsext_vf8_i64m4(v8, vl);
            vfloat64m4_t vx = __riscv_vfcvt_f_x_v_f64m4(vi64, vl);

            /* ((row[k]*scale_in - mu) * inv), exactly the reference's order */
            vx = __riscv_vfmul_vf_f64m4(vx, (double)scale_in, vl);
            vx = __riscv_vfsub_vf_f64m4(vx, mu, vl);
            vfloat64m4_t vy = __riscv_vfmul_vf_f64m4(vx, inv, vl);

            if (gamma) {
                size_t vl32 = __riscv_vsetvl_e32m2(n_elem);
                vfloat32m2_t g32 = __riscv_vle32_v_f32m2(gamma + k, vl32);
                size_t vl64 = __riscv_vsetvl_e64m4(n_elem);
                vfloat64m4_t g64 = __riscv_vfwcvt_f_f_v_f64m4(g32, vl64);
                vy = __riscv_vfmul_vv_f64m4(vy, g64, vl64);
            }
            if (beta) {
                size_t vl32 = __riscv_vsetvl_e32m2(n_elem);
                vfloat32m2_t b32 = __riscv_vle32_v_f32m2(beta + k, vl32);
                size_t vl64 = __riscv_vsetvl_e64m4(n_elem);
                vfloat64m4_t b64 = __riscv_vfwcvt_f_f_v_f64m4(b32, vl64);
                vy = __riscv_vfadd_vv_f64m4(vy, b64, vl64);
            }

            size_t vld = __riscv_vsetvl_e64m4(n_elem);
            vfloat64m4_t vq = __riscv_vfdiv_vf_f64m4(vy, (double)scale_out, vld);

            mb_frm_write(MB_FRM_RMM);
            vint64m4_t vr = __riscv_vfcvt_x_f_v_i64m4(vq, vld);
            mb_frm_write(frm_saved);

            vr = __riscv_vmax_vx_i64m4(vr, (int64_t)activation_min, vld);
            vr = __riscv_vmin_vx_i64m4(vr, (int64_t)activation_max, vld);

            /* Step down 64 -> 32 -> 16 -> 8, each naming its DESTINATION. */
            size_t vl32n = __riscv_vsetvl_e32m2(n_elem);
            vint32m2_t r32 = __riscv_vncvt_x_x_w_i32m2(vr, vl32n);
            size_t vl16n = __riscv_vsetvl_e16m1(n_elem);
            vint16m1_t r16 = __riscv_vncvt_x_x_w_i16m1(r32, vl16n);
            size_t vl8n = __riscv_vsetvl_e8mf2(n_elem);
            vint8mf2_t r8 = __riscv_vncvt_x_x_w_i8mf2(r16, vl8n);
            __riscv_vse8_v_i8mf2(orow + k, r8, vl8n);

            k += (int)vl;
        }
    }
}
