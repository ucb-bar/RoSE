/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV lstm_f16 (half-precision single-layer, seq-1 LSTM). Same
 *   structure as the curated lstm_s8, but fp16: the two gate GEMVs per unit —
 *   x . w_ih[g] (length in_size) and h . w_hh[g] (length H) — are vectorized with
 *   unit-stride fp16 loads + a PURE-fp16 multiply-accumulate (vfmacc_vv_f16) and
 *   an fp16 reduction (vfredusum_vs_f16). NOTHING widens to fp32: the whole vector
 *   hot loop stays SEW=16, so no fp32 vector datapath is needed (matters for the
 *   FPGA drone target — no LUT area for an fp32 vector ALU). This also matches the
 *   reference kernel's contract ("gate-GEMM reduction ACCUMULATES IN _Float16"):
 *   the only divergence from the fp16 reference is the reduction ORDER (fp16 is
 *   non-associative), hence accuracy_class=numeric_drift rather than bit_exact.
 *   The gate sigmoid/tanh + cell update are evaluated in scalar float (O(H), tiny)
 *   exactly as the reference — on the scalar FPU, not the vector unit. Deferred
 *   hidden writeback preserved (the h.w_hh GEMM reads the PREVIOUS h; new hidden
 *   goes to out[], committed to h[] after all units; out must not alias h).
 *
 * GATE-BLOCKED (2026-08-28, kernel_opt_log exp 1000+): x and h are IDENTICAL
 * across all 4 gates of a given unit j -- only the weight row differs (row =
 * g*H+j). The original per-gate mb_dot_f16() call reloaded x (length in_size)
 * and h (length H) from memory FOUR TIMES per unit (once per gate), via
 * vle16, on Saturn's single-ported vector LSU (one LoadSequencer for the
 * whole core, confirmed in saturn/src/main/scala/backend/Backend.scala --
 * `vls`/`vss` are each instantiated exactly once). mb_dot4_f16() below loads
 * each K-chunk of `a` (x or h) ONCE per chunk and reuses that same register
 * across all 4 gates' independent accumulators, cutting a/b-stream load
 * traffic ~4x for the x-dot and h-dot passes without changing weight traffic
 * (already loaded exactly once per element, unavoidable) or reduction count
 * (still exactly 4 reductions per pass, matching the original 4 separate
 * mb_dot_f16 calls -- see below). B=4 also happens to match Saturn's FMA
 * pipe depth (fmaPipeDepth=4 in VectorParams), so the 4 independent
 * accumulator chains issued back-to-back keep the FMA pipe's throughput
 * covered instead of stalling on the single-accumulator loop-carried RAW
 * dependency the original had.
 *
 * NUMERICALLY IDENTICAL to the previous curated kernel (not just same
 * accuracy_class): for a fixed gate g, vacc_g is updated by the exact same
 * sequence of vfmacc(vacc_g, va_chunk_k, vb_g_chunk_k) operations in the same
 * chunk order as the old mb_dot_f16(a, w[g], n) call would have produced --
 * only the *interleaving* with the other 3 gates' independent accumulators
 * changed, which does not affect any single gate's floating-point result. */

#include <math.h>
#include <stddef.h>
#include <riscv_vector.h>

/* 4-way "gate-blocked" fp16 dot product: computes dot(a, w[g]) for g=0..3
 * simultaneously, loading `a` ONCE per K-chunk and reusing it across all 4
 * gate weight rows. See file header for why this beats 4x mb_dot_f16(). */
static inline void mb_dot4_f16(const _Float16 *a, const _Float16 *const w[4],
                                int n, _Float16 out[4]) {
    const size_t vlmax = __riscv_vsetvlmax_e16m4();
    vfloat16m4_t vacc0 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
    vfloat16m4_t vacc1 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
    vfloat16m4_t vacc2 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
    vfloat16m4_t vacc3 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
    int k = 0;
    size_t vl;
    for (; k < n; k += (int)vl) {
        vl = __riscv_vsetvl_e16m4(n - k);
        vfloat16m4_t va = __riscv_vle16_v_f16m4(a + k, vl);   /* loaded ONCE, shared by 4 gates */

        vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(w[0] + k, vl);
        vacc0 = __riscv_vfmacc_vv_f16m4(vacc0, va, vb0, vl);

        vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(w[1] + k, vl);
        vacc1 = __riscv_vfmacc_vv_f16m4(vacc1, va, vb1, vl);

        vfloat16m4_t vb2 = __riscv_vle16_v_f16m4(w[2] + k, vl);
        vacc2 = __riscv_vfmacc_vv_f16m4(vacc2, va, vb2, vl);

        vfloat16m4_t vb3 = __riscv_vle16_v_f16m4(w[3] + k, vl);
        vacc3 = __riscv_vfmacc_vv_f16m4(vacc3, va, vb3, vl);
    }
    vfloat16m1_t vs0 = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vfloat16m1_t vs1 = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vfloat16m1_t vs2 = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vfloat16m1_t vs3 = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vs0 = __riscv_vfredusum_vs_f16m4_f16m1(vacc0, vs0, vlmax);
    vs1 = __riscv_vfredusum_vs_f16m4_f16m1(vacc1, vs1, vlmax);
    vs2 = __riscv_vfredusum_vs_f16m4_f16m1(vacc2, vs2, vlmax);
    vs3 = __riscv_vfredusum_vs_f16m4_f16m1(vacc3, vs3, vlmax);
    out[0] = __riscv_vfmv_f_s_f16m1_f16(vs0);
    out[1] = __riscv_vfmv_f_s_f16m1_f16(vs1);
    out[2] = __riscv_vfmv_f_s_f16m1_f16(vs2);
    out[3] = __riscv_vfmv_f_s_f16m1_f16(vs3);
}

void kernel_lstm_f16(const _Float16 *x, const _Float16 *w_ih, const _Float16 *w_hh,
                     const _Float16 *bias, _Float16 *h, _Float16 *c, _Float16 *out,
                     int in_size, int H) {
    for (int j = 0; j < H; j++) {
        const _Float16 *wih_rows[4] = {
            w_ih + (long)(0 * H + j) * in_size,
            w_ih + (long)(1 * H + j) * in_size,
            w_ih + (long)(2 * H + j) * in_size,
            w_ih + (long)(3 * H + j) * in_size,
        };
        const _Float16 *whh_rows[4] = {
            w_hh + (long)(0 * H + j) * H,
            w_hh + (long)(1 * H + j) * H,
            w_hh + (long)(2 * H + j) * H,
            w_hh + (long)(3 * H + j) * H,
        };
        _Float16 ax[4], ah[4];
        mb_dot4_f16(x, wih_rows, in_size, ax);
        mb_dot4_f16(h, whh_rows, H, ah);

        float pre[4];
        for (int g = 0; g < 4; g++) {
            int row = g * H + j;
            pre[g] = (float)ax[g] + (float)ah[g] + (float)bias[row];
        }
        float ig = 1.0f / (1.0f + expf(-pre[0]));
        float fg = 1.0f / (1.0f + expf(-pre[1]));
        float cg = tanhf(pre[2]);
        float og = 1.0f / (1.0f + expf(-pre[3]));
        float c_new = fg * (float)c[j] + ig * cg;
        c[j]   = (_Float16)c_new;
        out[j] = (_Float16)(og * tanhf(c_new));   /* into out[]; h[] stays previous */
    }
    for (int j = 0; j < H; j++) h[j] = out[j];   /* commit new hidden state */
}
