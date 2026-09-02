/* source: curated */
/* algorithm: gemmini_im2col_full_C */
/* accuracy_class: bit_exact */
/* EXPERIMENT (hw-gather): same bit-exact contract as the shipped
 * gemmini_im2col_full_C -- int32 accumulator drained via full_C and the
 * exact two-stage Q0.31 requantize on the CPU -- but the im2col GATHER is
 * done by Gemmini's mvin DMA instead of by the CPU.
 *
 * How: for a fixed kernel row kh, the A-matrix block
 *     A[p][kw*IC + ic] = padded_nhwc[n, oh*SH + kh, ow*SW + kw, ic]
 * (p = the output column index ow) is an AFFINE 2-D access pattern in
 * DRAM: row stride = SW*IC elements, and the (kw, ic) column index runs
 * over KW*IC CONTIGUOUS bytes because the padded activation is NHWC.
 * That is exactly what `config_ld` + `mvin` express, so tiled_matmul_auto
 * can read the im2col patch matrix straight out of the padded activation
 * -- overlapping rows and all -- with stride_A = SW*IC and dim_K = KW*IC.
 * No ws_im2col, no per-row scalar copy.
 *
 * The KH kernel rows are then chained through the int32 partial-sum
 * buffers: call kh=0 with D=bias, C=P0; kh=1 with D=P0, C=P1; ... The
 * chain is pure int32 (full_C on the way out, low_D=false on the way in,
 * and MVIN_SCALE_ACC is a hardwired no-op on this config), so no rounding
 * happens anywhere in the accelerator -- bit-exactness is preserved by
 * construction, exactly as in the shipped kernel.
 *
 * Zero padding is materialised once, fused into the NCHW->NHWC transpose
 * that this kernel had to do anyway, so the DMA never reads out of bounds
 * and the halo reads see real zeros.
 *
 * WEIGHT LAYOUT CONTRACT: unchanged -- `weight` is flat HWIO
 * [KH*KW*IC, OC], pre-packed at codegen time by
 * generate_skeleton.py::_backend_pack_weight. Rows [kh*KW*IC,
 * (kh+1)*KW*IC) of that matrix are exactly the B block for kernel row kh,
 * contiguous, stride OC. */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

enum {
    PAD_BYTES  = 256 * 1024,  /* zero-padded NHWC activation (yolov8 max 105 KB) */
    PART_ELEMS = 16 * 1024,   /* int32 partial-sum tile: rows*OC (yolov8 max 6 KB) */
};

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    static elem_t ws_pad[PAD_BYTES]  __attribute__((aligned(64)));
    static acc_t  ws_p0 [PART_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_p1 [PART_ELEMS] __attribute__((aligned(64)));

    const int OH  = (IH + 2*PH - KH) / SH + 1;
    const int OW  = (IW + 2*PW - KW) / SW + 1;
    const int IHp = IH + 2*PH;
    const int IWp = IW + 2*PW;

    /* Slack rows so the last mvin of a kh block (which reads KW*IC bytes
     * past the last row start) can never run off the end of ws_pad. */
    const size_t pad_elems = (size_t)N * (size_t)(IHp + KH) * (size_t)IWp * (size_t)IC;

    if (input_offset != 0 || filter_offset != 0
            || pad_elems > (size_t)PAD_BYTES
            || (size_t)OW * (size_t)OC > (size_t)PART_ELEMS) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        in_v = input_offset;
                                    else
                                        in_v = (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                                             + input_offset;
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                   + filter_offset);
                                }
                            }
                        }
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
                    }
                }
            }
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    /* NCHW -> zero-padded NHWC, single pass. */
    for (size_t i = 0; i < pad_elems; i++) ws_pad[i] = 0;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < IC; c++) {
            const int8_t *src = input + ((size_t)n*IC + c)*IH*IW;
            for (int h = 0; h < IH; h++) {
                elem_t *dst = ws_pad
                    + ((((size_t)n*(IHp + KH) + (h + PH))*IWp) + PW)*IC + c;
                for (int w = 0; w < IW; w++) dst[(size_t)w*IC] = src[(size_t)h*IW + w];
            }
        }

    asm volatile("fence" ::: "memory");

    const int    K_row     = KW * IC;              /* dim_K per kernel row   */
    const size_t stride_A  = (size_t)SW * (size_t)IC;
    const int    rows_max  = PART_ELEMS / OC;
    /* When SH*IWp == OW*SW the (oh,ow) -> DRAM map is affine across whole
     * output rows too, so one call can cover several output rows. */
    const int    merged    = (SH * IWp == OW * SW);
    const int    row_span  = merged ? (OH * OW) : OW;

    for (int n = 0; n < N; n++) {
        for (int seg = 0; seg < (merged ? 1 : OH); seg++) {
            const int   base_out = merged ? (n*OH*OW) : (n*OH*OW + seg*OW);
            const size_t base_a  = (((size_t)n*(IHp + KH) + (size_t)seg*SH)*IWp)*IC;

            for (int r0 = 0; r0 < row_span; r0 += rows_max) {
                const int rows = (row_span - r0) < rows_max ? (row_span - r0) : rows_max;
                acc_t *cur = ws_p0, *prev = ws_p1;

                for (int kh = 0; kh < KH; kh++) {
                    const elem_t *A = ws_pad + base_a
                                    + (size_t)kh * (size_t)IWp * (size_t)IC
                                    + (size_t)r0 * stride_A;
                    const elem_t *B = weight + (size_t)kh * (size_t)K_row * (size_t)OC;
                    const void   *D;
                    int repeating;
                    if (kh == 0) { D = (const void *)bias; repeating = bias != NULL; }
                    else         { D = (const void *)prev; repeating = 0; }

                    tiled_matmul_auto(
                        (size_t)rows, (size_t)OC, (size_t)K_row,
                        A, B, D, (void *)cur,
                        stride_A, (size_t)OC, (size_t)OC, (size_t)OC,
                        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
                        NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
                        repeating,
                        false, false,
                        true, false,
                        0, WS
                    );

                    acc_t *t = cur; cur = prev; prev = t;   /* result now in `prev` */
                }

                gemmini_fence();
                gemmini_flush(0);

                /* Exact two-stage Q0.31 requantize, int32 -> int8 NCHW.
                 * Byte-for-byte the same arithmetic as the shipped kernel. */
                const acc_t *acc_buf = prev;
                for (int r = 0; r < rows; r++) {
                    const int out_idx = base_out + r0 + r;
                    const int ow_idx  = out_idx % OW;
                    const int oh_idx  = (out_idx / OW) % OH;
                    const int n_idx   = out_idx / (OH * OW);
                    const size_t out_base =
                        (((size_t)n_idx*OC)*OH + (size_t)oh_idx)*OW + (size_t)ow_idx;
                    const size_t out_oc_stride = (size_t)OH * (size_t)OW;
                    for (int oc = 0; oc < OC; oc++) {
                        int32_t acc = acc_buf[(size_t)r * OC + oc];
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[out_base + (size_t)oc * out_oc_stride] = (int8_t)scaled;
                    }
                }
            }
        }
    }
}
