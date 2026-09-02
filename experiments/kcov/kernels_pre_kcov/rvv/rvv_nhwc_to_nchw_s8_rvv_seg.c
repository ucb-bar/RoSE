/* source: curated */
/* algorithm: rvv_seg */
/* accuracy_class: bit_exact */
/* origin: new, and the exact mirror of rvv_nchw_to_nhwc_s8_rvv_seg.c. A
 * segment LOAD (vlseg / vlsseg) is the ISA's AoS -> SoA primitive: it reads NF
 * consecutive bytes at each of vl positions and lands them in NF separate
 * registers, i.e. it de-interleaves. NHWC -> NCHW is exactly that, and each
 * de-interleaved field is then one contiguous vse8 into its own channel plane.
 *
 * The channel decomposition, the "C = 3 is a single vlseg3e8 rather than
 * 2 + 1" argument, the NF * LMUL <= 8 reason for holding e8m1, and the #undef
 * discipline are all identical to that file -- see its header, which is the
 * one place they are written out.
 *
 * Worth measuring separately from the store direction rather than assumed
 * symmetric. The scalar phase data (docs/IR_TENSOR_LAYOUT_DESIGN.md section 1)
 * has this direction costing 1.55x the other over the same ten dispatches on a
 * core with no vector unit, and Saturn's load and store segment paths are
 * distinct hardware (LoadSegmenter.scala vs StoreSegmenter.scala) with no
 * reason to have matching throughput. */

#define MB_RH2N_G(T, K) __riscv_vget_v_i8m1x##T##_i8m1((tup), (K))

#define MB_RH2N_ST(K)                                                         \
    __riscv_vse8_v_i8m1(ob + (size_t)(c0 + (K)) * HW + p0, _v##K, vl)

/* Split a loaded NF-tuple back into NF contiguous channel planes. */
#define MB_RH2N_SPILL2(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1);                \
         MB_RH2N_ST(0); MB_RH2N_ST(1); } while (0)
#define MB_RH2N_SPILL3(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2);                                      \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); } while (0)
#define MB_RH2N_SPILL4(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2), _v3 = MB_RH2N_G(T,3);                \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); MB_RH2N_ST(3);          \
    } while (0)
#define MB_RH2N_SPILL5(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2), _v3 = MB_RH2N_G(T,3),                \
                   _v4 = MB_RH2N_G(T,4);                                      \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); MB_RH2N_ST(3);          \
         MB_RH2N_ST(4); } while (0)
#define MB_RH2N_SPILL6(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2), _v3 = MB_RH2N_G(T,3),                \
                   _v4 = MB_RH2N_G(T,4), _v5 = MB_RH2N_G(T,5);                \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); MB_RH2N_ST(3);          \
         MB_RH2N_ST(4); MB_RH2N_ST(5); } while (0)
#define MB_RH2N_SPILL7(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2), _v3 = MB_RH2N_G(T,3),                \
                   _v4 = MB_RH2N_G(T,4), _v5 = MB_RH2N_G(T,5),                \
                   _v6 = MB_RH2N_G(T,6);                                      \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); MB_RH2N_ST(3);          \
         MB_RH2N_ST(4); MB_RH2N_ST(5); MB_RH2N_ST(6); } while (0)
#define MB_RH2N_SPILL8(T)                                                     \
    do { vint8m1_t _v0 = MB_RH2N_G(T,0), _v1 = MB_RH2N_G(T,1),                \
                   _v2 = MB_RH2N_G(T,2), _v3 = MB_RH2N_G(T,3),                \
                   _v4 = MB_RH2N_G(T,4), _v5 = MB_RH2N_G(T,5),                \
                   _v6 = MB_RH2N_G(T,6), _v7 = MB_RH2N_G(T,7);                \
         MB_RH2N_ST(0); MB_RH2N_ST(1); MB_RH2N_ST(2); MB_RH2N_ST(3);          \
         MB_RH2N_ST(4); MB_RH2N_ST(5); MB_RH2N_ST(6); MB_RH2N_ST(7);          \
    } while (0)

/* C == NF: the segment stride IS the channel stride, so no stride operand. */
#define MB_RH2N_UNIT(NF)                                                      \
    for (int n = 0; n < N; n++) {                                             \
        const int8_t *ib = input  + (size_t)n * HW * (size_t)(NF);            \
        int8_t       *ob = output + (size_t)n * (size_t)(NF) * HW;            \
        const int c0 = 0;                                                     \
        for (size_t p0 = 0; p0 < HW; ) {                                      \
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);                         \
            vint8m1x##NF##_t tup = __riscv_vlseg##NF##e8_v_i8m1x##NF(         \
                    ib + p0 * (size_t)(NF), vl);                              \
            MB_RH2N_SPILL##NF(NF);                                            \
            p0 += vl;                                                         \
        }                                                                     \
    }

void kernel_nhwc_to_nchw_s8(const int8_t *input, int8_t *output,
                            int N, int C, int H, int W)
{
    const size_t HW = (size_t)H * (size_t)W;
    const ptrdiff_t bs = (ptrdiff_t)C;

    switch (C) {
    case 2: MB_RH2N_UNIT(2) return;
    case 3: MB_RH2N_UNIT(3) return;
    case 4: MB_RH2N_UNIT(4) return;
    case 5: MB_RH2N_UNIT(5) return;
    case 6: MB_RH2N_UNIT(6) return;
    case 7: MB_RH2N_UNIT(7) return;
    case 8: MB_RH2N_UNIT(8) return;
    default: break;
    }

    for (int n = 0; n < N; n++) {
        const int8_t *ib = input  + (size_t)n * HW * (size_t)C;
        int8_t       *ob = output + (size_t)n * (size_t)C * HW;

        for (size_t p0 = 0; p0 < HW; ) {
            size_t vl = __riscv_vsetvl_e8m1(HW - p0);
            const int8_t *sp = ib + p0 * (size_t)C;
            int c0 = 0;

            for (; C - c0 >= 8; c0 += 8) {
                vint8m1x8_t tup =
                    __riscv_vlsseg8e8_v_i8m1x8(sp + c0, bs, vl);
                MB_RH2N_SPILL8(8);
            }

            switch (C - c0) {
            case 7: { vint8m1x7_t tup =
                          __riscv_vlsseg7e8_v_i8m1x7(sp + c0, bs, vl);
                      MB_RH2N_SPILL7(7); } break;
            case 6: { vint8m1x6_t tup =
                          __riscv_vlsseg6e8_v_i8m1x6(sp + c0, bs, vl);
                      MB_RH2N_SPILL6(6); } break;
            case 5: { vint8m1x5_t tup =
                          __riscv_vlsseg5e8_v_i8m1x5(sp + c0, bs, vl);
                      MB_RH2N_SPILL5(5); } break;
            case 4: { vint8m1x4_t tup =
                          __riscv_vlsseg4e8_v_i8m1x4(sp + c0, bs, vl);
                      MB_RH2N_SPILL4(4); } break;
            case 3: { vint8m1x3_t tup =
                          __riscv_vlsseg3e8_v_i8m1x3(sp + c0, bs, vl);
                      MB_RH2N_SPILL3(3); } break;
            case 2: { vint8m1x2_t tup =
                          __riscv_vlsseg2e8_v_i8m1x2(sp + c0, bs, vl);
                      MB_RH2N_SPILL2(2); } break;
            case 1: { vint8m1_t _v0 =
                          __riscv_vlse8_v_i8m1(sp + c0, bs, vl);
                      MB_RH2N_ST(0); } break;
            default: break;
            }
            p0 += vl;
        }
    }
}

#undef MB_RH2N_G
#undef MB_RH2N_ST
#undef MB_RH2N_SPILL2
#undef MB_RH2N_SPILL3
#undef MB_RH2N_SPILL4
#undef MB_RH2N_SPILL5
#undef MB_RH2N_SPILL6
#undef MB_RH2N_SPILL7
#undef MB_RH2N_SPILL8
#undef MB_RH2N_UNIT
