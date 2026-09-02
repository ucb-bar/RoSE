/* source: curated */
/* algorithm: narrow */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp16 linear with PURE-fp16 accumulation. The K-reduction
 *   dot(input[m], weight[n]) is done with unit-stride fp16 loads + vfmacc_vv_f16
 *   (fp16 multiply-accumulate) and an fp16 reduction (vfredusum_vs_f16) — the whole
 *   vector hot loop stays SEW=16, NO widening to fp32. This is the fp32-free
 *   counterpart of rvv_f16_linear_f16_widening.c (which uses vfwmacc -> an fp32
 *   accumulator): use THIS on targets with no fp32 vector datapath (the FPGA drone
 *   target has no LUT area for an fp32 vector ALU). Divergence from the reference
 *   is reduction ORDER only (fp16 is non-associative) -> numeric_drift. bias +
 *   final store match the reference (output[m*N+n] = (_Float16)(acc + bias[n])).
 *
 * N-BLOCKED BY 4 (2026-08-28, kernel_opt_log exp 1000+): input[m]'s K-length row
 * is IDENTICAL across every output n -- only the weight row differs. The
 * original per-n loop reloaded the whole input row from memory via vle16 once
 * per K-chunk *per n*, i.e. N times total for a given m, on Saturn's
 * single-ported vector LSU (exactly one LoadSequencer for the whole core --
 * saturn/src/main/scala/backend/Backend.scala). For the actual production
 * shapes here (fused_full's vision_fc M=1,K=1536,N=512 and depth_fc
 * M=1,K=1024,N=64), that made the redundant input-row reload traffic roughly
 * EQUAL to the necessary (load-each-weight-element-once) weight traffic --
 * i.e. this kernel was paying for ~2x the load-port bandwidth it needed.
 *
 * This version processes N in blocks of 4: for each K-chunk, `input` is
 * loaded ONCE and fed to 4 independent accumulators (one per output in the
 * block), each fed by its own weight-chunk load. That cuts input-row reload
 * traffic ~4x while leaving weight traffic and the reduction count IDENTICAL
 * to the original (still exactly one reduction per (m,n), same per-n chunk
 * order) — the four accumulators are NUMERICALLY INDEPENDENT of each other,
 * so this is not just same-accuracy-class but bit-identical to running the
 * original per-n algorithm 4 times, just with the input load shared. B=4
 * also matches Saturn's FMA pipe depth (fmaPipeDepth=4), so 4 independent
 * accumulator chains issued back-to-back keep the FMA pipe covered instead
 * of stalling on a lone accumulator's loop-carried RAW dependency.
 *
 * The N%4 tail (if any) falls back to the original single-n algorithm
 * unchanged, so odd N is still exactly correct. */

#include <stddef.h>
#include <riscv_vector.h>

/* Original per-n algorithm, used for the N%4 tail. */
static inline _Float16 mb_linear_f16_one(const _Float16 *in_row, const _Float16 *w_row,
                                          int K, size_t vlmax) {
    vfloat16m4_t vacc = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
    int k = 0;
    size_t vl;
    for (; k < K; k += (int)vl) {
        vl = __riscv_vsetvl_e16m4(K - k);
        vfloat16m4_t va = __riscv_vle16_v_f16m4(in_row + k, vl);
        vfloat16m4_t vb = __riscv_vle16_v_f16m4(w_row + k, vl);
        vacc = __riscv_vfmacc_vv_f16m4(vacc, va, vb, vl);
    }
    vfloat16m1_t vs = __riscv_vfmv_s_f_f16m1((_Float16)0.0f, 1);
    vs = __riscv_vfredusum_vs_f16m4_f16m1(vacc, vs, vlmax);
    return __riscv_vfmv_f_s_f16m1_f16(vs);
}

void kernel_linear_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int M, int K, int N) {
    const size_t vlmax = __riscv_vsetvlmax_e16m4();
    for (int m = 0; m < M; m++) {
        const _Float16 *in_row = input + (size_t)m * (size_t)K;
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            const _Float16 *w0 = weight + (size_t)(n + 0) * (size_t)K;
            const _Float16 *w1 = weight + (size_t)(n + 1) * (size_t)K;
            const _Float16 *w2 = weight + (size_t)(n + 2) * (size_t)K;
            const _Float16 *w3 = weight + (size_t)(n + 3) * (size_t)K;

            vfloat16m4_t vacc0 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
            vfloat16m4_t vacc1 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
            vfloat16m4_t vacc2 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);
            vfloat16m4_t vacc3 = __riscv_vfmv_v_f_f16m4((_Float16)0.0f, vlmax);

            int k = 0;
            size_t vl;
            for (; k < K; k += (int)vl) {
                vl = __riscv_vsetvl_e16m4(K - k);
                vfloat16m4_t va = __riscv_vle16_v_f16m4(in_row + k, vl);  /* loaded ONCE, shared by 4 outputs */

                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(w0 + k, vl);
                vacc0 = __riscv_vfmacc_vv_f16m4(vacc0, va, vb0, vl);

                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(w1 + k, vl);
                vacc1 = __riscv_vfmacc_vv_f16m4(vacc1, va, vb1, vl);

                vfloat16m4_t vb2 = __riscv_vle16_v_f16m4(w2 + k, vl);
                vacc2 = __riscv_vfmacc_vv_f16m4(vacc2, va, vb2, vl);

                vfloat16m4_t vb3 = __riscv_vle16_v_f16m4(w3 + k, vl);
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

            _Float16 acc0 = __riscv_vfmv_f_s_f16m1_f16(vs0);
            _Float16 acc1 = __riscv_vfmv_f_s_f16m1_f16(vs1);
            _Float16 acc2 = __riscv_vfmv_f_s_f16m1_f16(vs2);
            _Float16 acc3 = __riscv_vfmv_f_s_f16m1_f16(vs3);
            if (bias) {
                acc0 = (_Float16)(acc0 + bias[n + 0]);
                acc1 = (_Float16)(acc1 + bias[n + 1]);
                acc2 = (_Float16)(acc2 + bias[n + 2]);
                acc3 = (_Float16)(acc3 + bias[n + 3]);
            }
            output[(size_t)m * N + n + 0] = acc0;
            output[(size_t)m * N + n + 1] = acc1;
            output[(size_t)m * N + n + 2] = acc2;
            output[(size_t)m * N + n + 3] = acc3;
        }
        /* N % 4 tail: original per-n algorithm, unchanged. */
        for (; n < N; n++) {
            const _Float16 *w_row = weight + (size_t)n * (size_t)K;
            _Float16 acc = mb_linear_f16_one(in_row, w_row, K, vlmax);
            if (bias) acc = (_Float16)(acc + bias[n]);
            output[(size_t)m * N + n] = acc;
        }
    }
}
