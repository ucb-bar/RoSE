/* source: curated */
/* algorithm: rvv_memo_lut_gather */
/* accuracy_class: bit_exact */
/* origin: hand-written, identical in structure to
 *         rvv_elu_s8_rvv_memo_lut_gather.c -- read that file's header for
 *         why the memoized table beats both the reference loop and an
 *         eagerly-built 256-entry LUT.
 *
 *   WHY THIS FILE EXISTS, AND WHAT IT IS WORTH. sigmoid_s8 carried no
 *   AlgorithmCandidate, so no target had an (op, algorithm) pair to probe
 *   for. Its one appearance in these models is DroNet's output head at
 *   n=1: 0.0% of the run, one expf. At n=1 this kernel takes the same
 *   scalar path the reference does, by design -- the marking pass over 256
 *   bytes would cost far more than the single transcendental it saves.
 *
 *   So the honest summary is: this closes a coverage hole and will pay for
 *   itself on a model that applies sigmoid to a real tensor. It does not
 *   make DroNet faster and is not claimed to.
 *
 *   VTYPE. Single 8-bit domain in the vector path. Checked with
 *   scripts/check_rvv_vtype.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

#ifndef MB_SIGMOID_MEMO_MIN
#define MB_SIGMOID_MEMO_MIN 32
#endif

/* See rvv_elu_s8_rvv_memo_lut_gather.c; guarded because both bodies land in
 * the same kernels.c. */
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
static inline int8_t mb_sigmoid_s8_one(int8_t x, float scale_in,
                                       float scale_out,
                                       int activation_min, int activation_max)
{
    float fv = (float)x * scale_in;
    float sig = 1.0f / (1.0f + expf(-fv));
    int32_t v = (int32_t)roundf(sig / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max)
{
    if (n <= 0) return;
    if (n < MB_SIGMOID_MEMO_MIN) {
        for (int i = 0; i < n; i++)
            output[i] = mb_sigmoid_s8_one(input[i], scale_in, scale_out,
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
        table[b] = mb_sigmoid_s8_one((int8_t)(b - 128), scale_in, scale_out,
                                     activation_min, activation_max);
    }
    mb_rvv_lut_gather_s8(input, output, n, table);
}
