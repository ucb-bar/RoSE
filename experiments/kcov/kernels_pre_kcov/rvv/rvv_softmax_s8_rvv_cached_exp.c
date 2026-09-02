/* source: curated */
/* algorithm: rvv_cached_exp */
/* accuracy_class: bit_exact */
/* origin: hand-written. The reference's own comment proposes this:
 *         "Curated kernels can elide the second pass by caching exp values in
 *         a scratch buffer." That is what this does, plus vectorising the
 *         requantize tail that the cache makes reachable.
 *
 *   WHY THIS IS BIT-EXACT AND NOT A TOLERANCE. The reference calls expf TWICE
 *   per element with the SAME argument -- once to build the denominator, once
 *   to quantize. Computing it once and reading it back is not an
 *   approximation of that, it is the same value: expf is a pure function of
 *   its argument. Nothing about the arithmetic changes, so there is no
 *   rounding question to answer.
 *
 *   WHAT STAYS SCALAR, DELIBERATELY. The denominator. `sum += expf(...)` is a
 *   sequential float accumulation, and a vector tree reduction would add the
 *   same terms in a different ORDER -- a different result, by a few ULP, on a
 *   value every output is divided by. RVV does have an ordered reduction
 *   (vfredosum), but here the sum is K adds against K transcendentals: it is
 *   not where the time is, and making it vector would trade the only part of
 *   this kernel that is cheap for the only part that is delicate.
 *
 *   The row maximum IS vectorised. max is associative and commutative over a
 *   total order, so reduction order cannot change it -- unlike the sum, this
 *   one is exact whatever shape the reduction has.
 *
 *   WHAT THIS IS WORTH. Two expf per element becomes one, and the requantize
 *   tail goes from scalar to vector. expf is ~140 cycles on this core and the
 *   rest of the row is loads and two divides, so halving the transcendental
 *   count is most of the available win.
 *
 *   THE SCRATCH IS A STACK ARRAY, NOT A STATIC ONE. `modelblaster_pool` runs
 *   one worker per hart in a single address space, so a file-scope buffer
 *   would be shared by every hart running this kernel at once -- and the
 *   corruption would be intermittent, data-dependent, and invisible on a
 *   single-core profile. Bounded, with the reference's two-pass loop as the
 *   fallback above the bound, so a large K is slow rather than wrong.
 *
 *   THE DIVIDES STAY DIVIDES. `e / sum` and `p / scale_out` are two separate
 *   divisions in the reference; folding them into one, or into a multiply by
 *   a reciprocal, removes an intermediate rounding it performs.
 *
 *   roundf() IS SPELLED AS A ROUNDING MODE -- frm=RMM is round-to-nearest,
 *   ties away from zero, which is what roundf does. See
 *   rvv_add_s8_rvv_frm_rmm.c for the tie sweep that established this on the
 *   board, and for why frm is toggled around the conversion only.
 *
 *   VTYPE. Every width-domain transition is named explicitly, and the AVL
 *   handed to each is the ELEMENT COUNT, never a previous vsetvl's result --
 *   see scripts/check_rvv_avl.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Rows wider than this fall back to the reference's two-pass loop. 1024
 * float32 is 4 KB of stack per call, comfortable inside a pthread's default
 * stack; the softmaxes in these models are K=8 (attention) and K<=128. */
#ifndef MB_SOFTMAX_MAX_K
#define MB_SOFTMAX_MAX_K 1024
#endif

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

void kernel_softmax_s8(const int8_t *input, int8_t *output, int M, int K,
                       float scale_in, float scale_out)
{
    const unsigned frm_saved = mb_frm_read();

    for (int m = 0; m < M; m++) {
        const int8_t *row_in  = input  + (size_t)m * (size_t)K;
        int8_t       *row_out = output + (size_t)m * (size_t)K;

        if (K <= 0) continue;

        if (K > MB_SOFTMAX_MAX_K) {
            /* Reference loop, verbatim. */
            int8_t shift = row_in[0];
            for (int k = 1; k < K; k++)
                if (row_in[k] > shift) shift = row_in[k];
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += expf(((float)row_in[k] - (float)shift) * scale_in);
            for (int k = 0; k < K; k++) {
                float e = expf(((float)row_in[k] - (float)shift) * scale_in);
                float p = e / sum;
                int32_t v = (int32_t)roundf(p / scale_out);
                if (v < -128) v = -128;
                if (v >  127) v =  127;
                row_out[k] = (int8_t)v;
            }
            continue;
        }

        /* ---- row maximum, vectorised. Order-independent, so exact. ------- */
        int8_t shift = row_in[0];
        {
            vint8m1_t vmax = __riscv_vmv_v_x_i8m1(shift, __riscv_vsetvlmax_e8m1());
            int k = 0;
            while (k < K) {
                size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                vint8m1_t v = __riscv_vle8_v_i8m1(row_in + k, vl);
                vmax = __riscv_vmax_vv_i8m1(vmax, v, vl);
                k += (int)vl;
            }
            /* The tail lanes still hold row_in[0], which is a real element of
             * the row, so seeding with it cannot invent a larger maximum. */
            size_t vlm = __riscv_vsetvlmax_e8m1();
            vint8m1_t red = __riscv_vmv_v_x_i8m1(row_in[0], 1);
            red = __riscv_vredmax_vs_i8m1_i8m1(vmax, red, vlm);
            shift = __riscv_vmv_x_s_i8m1_i8(red);
        }

        /* ---- one expf per element, and the reference's own sum order ----- */
        float ebuf[MB_SOFTMAX_MAX_K];
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            ebuf[k] = expf(((float)row_in[k] - (float)shift) * scale_in);
            sum += ebuf[k];
        }

        /* ---- requantize, vectorised over the cached exponentials --------- */
        int k = 0;
        while (k < K) {
            const size_t n_elem = (size_t)(K - k);
            size_t vl = __riscv_vsetvl_e32m4(n_elem);
            vfloat32m4_t ve = __riscv_vle32_v_f32m4(ebuf + k, vl);
            vfloat32m4_t vp = __riscv_vfdiv_vf_f32m4(ve, sum, vl);
            vfloat32m4_t vq = __riscv_vfdiv_vf_f32m4(vp, scale_out, vl);

            mb_frm_write(MB_FRM_RMM);
            vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vq, vl);
            mb_frm_write(frm_saved);

            vi = __riscv_vmax_vx_i32m4(vi, -128, vl);
            vi = __riscv_vmin_vx_i32m4(vi, 127, vl);

            size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
            vint16m2_t vi16 = __riscv_vncvt_x_x_w_i16m2(vi, vl16);
            size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
            vint8m1_t vi8 = __riscv_vncvt_x_x_w_i8m1(vi16, vl8);
            __riscv_vse8_v_i8m1(row_out + k, vi8, vl8);

            k += (int)vl;
        }
    }
}
