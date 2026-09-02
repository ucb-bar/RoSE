/* source: curated */
/* algorithm: rvv_seg */
/* accuracy_class: bit_exact */
/* origin: new. RVV segment stores (vsseg / vssseg, NF = 2..8) are the ISA's
 * native SoA -> AoS primitive, and NCHW -> NHWC is exactly SoA -> AoS: NF
 * separately-based contiguous channel planes in, one interleaved run of NF
 * bytes per spatial position out. Nothing in this tree used a segment op
 * before this file (0 occurrences across kernels/), which is why
 * docs/IR_TENSOR_LAYOUT_DESIGN.md section 5.2a lists it as unexploited
 * hardware and stage 1 exists to settle it. Saturn does implement the family
 * (generators/saturn/.../mem/LoadSegmenter.scala, StoreSegmenter.scala).
 *
 * THE CHANNEL DECOMPOSITION, which is the whole design of this file
 * -----------------------------------------------------------------
 * NF is an instruction field, so it is fixed at compile time, while C is a
 * runtime argument. The channel axis is therefore chopped into groups:
 *
 *   2 <= C <= 8   ONE unit-stride vsseg<C>e8 covers the whole tensor. The
 *                 segment stride and the tensor's channel stride coincide, so
 *                 there is no stride field to pay for. This is the case
 *                 section 5.2a names ("C == 8 is one vsseg8e8") and it is also
 *                 where C = 3 lands.
 *   C > 8         floor(C/8) passes of vssseg8e8 at byte-stride C, then one
 *                 pass at NF = C mod 8. C = 32 is four passes and C = 128 is
 *                 sixteen, matching the section's arithmetic.
 *   C == 1        degenerate: no interleave exists, so it is a plain vsse8
 *                 (identical to the strided kernel, and to a memcpy in
 *                 effect).
 *
 * C = 3 IS THE CASE THAT MATTERS. It is conv0's input, the largest activation
 * in dronet (112x112x3 = 37,632 B, and its transposes are 39.1% + 14.7% of all
 * conv cycles per section 1), and it is the one geometry where C is not a
 * multiple of anything convenient. A decomposition that only did powers of two
 * would serve it as 2 + 1: two passes over the whole tensor, the second of
 * them a degenerate one-channel scatter. Because NF is legal for every value
 * in 2..8 rather than only powers of two, C = 3 is instead a single
 * vsseg3e8 -- the fast path, not the fallback. That is why the group width is
 * "whatever fits, greedily up to 8" and not "8 or bust".
 *
 * The tail switch is unrolled per NF because the tuple accessors
 * (__riscv_vcreate_v_i8m1x<NF>) are distinct types with a distinct intrinsic
 * per NF; there is no way to write them NF-generically in C. The macros below
 * keep that from becoming seven copies of the loop body, and are #undef'd at
 * the end of the file because kernels.c is a single translation unit that
 * concatenates every selected kernel.
 *
 * e8m1 and not wider: NF * LMUL <= 8 is an architectural constraint, so an
 * NF = 8 segment op REQUIRES LMUL = 1. Holding LMUL at 1 for every NF keeps
 * the spatial block at VLMAX = 32 bytes (VLEN = 256), which is the same
 * working-set window as the scalar gemmini_blocked_tb32 kernel's TB = 32 and
 * the same as the rvv_strided sibling -- so the three kernels differ only in
 * the instruction that moves the bytes, which is the comparison being made. */

/* One channel plane of the current spatial window, as a vector. */
#define MB_RN2H_P(K) \
    __riscv_vle8_v_i8m1(ib + (size_t)(c0 + (K)) * HW + p0, vl)

#define MB_RN2H_T2 __riscv_vcreate_v_i8m1x2(MB_RN2H_P(0), MB_RN2H_P(1))
#define MB_RN2H_T3 __riscv_vcreate_v_i8m1x3(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2))
#define MB_RN2H_T4 __riscv_vcreate_v_i8m1x4(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2), MB_RN2H_P(3))
#define MB_RN2H_T5 __riscv_vcreate_v_i8m1x5(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2), MB_RN2H_P(3), \
                                            MB_RN2H_P(4))
#define MB_RN2H_T6 __riscv_vcreate_v_i8m1x6(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2), MB_RN2H_P(3), \
                                            MB_RN2H_P(4), MB_RN2H_P(5))
#define MB_RN2H_T7 __riscv_vcreate_v_i8m1x7(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2), MB_RN2H_P(3), \
                                            MB_RN2H_P(4), MB_RN2H_P(5), \
                                            MB_RN2H_P(6))
#define MB_RN2H_T8 __riscv_vcreate_v_i8m1x8(MB_RN2H_P(0), MB_RN2H_P(1), \
                                            MB_RN2H_P(2), MB_RN2H_P(3), \
                                            MB_RN2H_P(4), MB_RN2H_P(5), \
                                            MB_RN2H_P(6), MB_RN2H_P(7))

/* C == NF: the segment stride IS the channel stride, so no stride operand. */
#define MB_RN2H_UNIT(NF)                                                      \
    for (int n = 0; n < N; n++) {                                             \
        const int8_t *ib = input  + (size_t)n * (size_t)(NF) * HW;            \
        int8_t       *ob = output + (size_t)n * HW * (size_t)(NF);            \
        const int c0 = 0;                                                     \
        for (size_t p0 = 0; p0 < HW; ) {                                      \
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);                         \
            __riscv_vsseg##NF##e8_v_i8m1x##NF(ob + p0 * (size_t)(NF),         \
                                              MB_RN2H_T##NF, vl);             \
            p0 += vl;                                                         \
        }                                                                     \
    }

void kernel_nchw_to_nhwc_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const size_t HW = (size_t)H * (size_t)W;
    const ptrdiff_t bs = (ptrdiff_t)C;

    switch (C) {
    case 2: MB_RN2H_UNIT(2) return;
    case 3: MB_RN2H_UNIT(3) return;
    case 4: MB_RN2H_UNIT(4) return;
    case 5: MB_RN2H_UNIT(5) return;
    case 6: MB_RN2H_UNIT(6) return;
    case 7: MB_RN2H_UNIT(7) return;
    case 8: MB_RN2H_UNIT(8) return;
    default: break;
    }

    for (int n = 0; n < N; n++) {
        const int8_t *ib = input  + (size_t)n * (size_t)C * HW;
        int8_t       *ob = output + (size_t)n * HW * (size_t)C;

        for (size_t p0 = 0; p0 < HW; ) {
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);
            int8_t *dp = ob + p0 * (size_t)C;
            int c0 = 0;

            for (; C - c0 >= 8; c0 += 8)
                __riscv_vssseg8e8_v_i8m1x8(dp + c0, bs, MB_RN2H_T8, vl);

            switch (C - c0) {
            case 7: __riscv_vssseg7e8_v_i8m1x7(dp + c0, bs, MB_RN2H_T7, vl);
                    break;
            case 6: __riscv_vssseg6e8_v_i8m1x6(dp + c0, bs, MB_RN2H_T6, vl);
                    break;
            case 5: __riscv_vssseg5e8_v_i8m1x5(dp + c0, bs, MB_RN2H_T5, vl);
                    break;
            case 4: __riscv_vssseg4e8_v_i8m1x4(dp + c0, bs, MB_RN2H_T4, vl);
                    break;
            case 3: __riscv_vssseg3e8_v_i8m1x3(dp + c0, bs, MB_RN2H_T3, vl);
                    break;
            case 2: __riscv_vssseg2e8_v_i8m1x2(dp + c0, bs, MB_RN2H_T2, vl);
                    break;
            case 1: __riscv_vsse8_v_i8m1(dp + c0, bs, MB_RN2H_P(0), vl);
                    break;
            default: break;
            }
            p0 += vl;
        }
    }
}

#undef MB_RN2H_P
#undef MB_RN2H_T2
#undef MB_RN2H_T3
#undef MB_RN2H_T4
#undef MB_RN2H_T5
#undef MB_RN2H_T6
#undef MB_RN2H_T7
#undef MB_RN2H_T8
#undef MB_RN2H_UNIT
