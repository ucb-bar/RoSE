/* source: curated */
/* algorithm: gemmini_im2col_full_C */
/* accuracy_class: bit_exact */
/* WEIGHT LAYOUT CONTRACT: like the sibling tiled_conv variant, this
 * kernel expects `weight` already in flat HWIO layout
 * ([KH*KW*IC, OC]) — the form tiled_matmul_auto consumes directly.
 * The skeleton emitter (generate_skeleton.py::_backend_pack_weight)
 * permutes OIHW→HWIO at codegen time when --backend gemmini, so
 * we pass weight straight through without a workspace copy. */
/* origin: im2col → tiled_matmul_auto(full_C=true) → scalar Q0.31 requantize.
 *         Bypasses Saturn-Gemmini float-scale mvout; bit-exact with the
 *         Q0.31 PyTorch golden (max_abs_err=0 validated on Saturn FireSim
 *         May 2026).  Handles non-square kernels, any stride/padding, and
 *         large output_shift values (int64 requantize, no UB). */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

/*
 * Static workspace limits.  512 KB covers all square conv layers in
 * dronet and yolov8_nano:
 *   WS_BYTES:     max input  = IC=3,IH=160,IW=160 →  75 KB (yolov8 l0)
 *                 max output = IC=16,OH=80,OW=80  → 100 KB (yolov8 l0)
 *                 (ws_weight is gone — weight is pre-packed HWIO at
 *                  codegen time and passed straight to tiled_matmul_auto)
 *   IM2COL_ELEMS: max K_inner = IC=256,K=3×3     → 2304 (yolov8 detect head)
 *   ACC_ELEMS:    max OC      = 256               (yolov8 l7/l8/l9)
 */
enum {
    WS_BYTES       = 512 * 1024,
    IM2COL_ELEMS   = DIM * 256 * 9,   /* DIM rows × max K_inner (IC=256, 3×3) */
    ACC_ELEMS      = DIM * 256,        /* DIM rows × max OC (256 in yolov8)    */
};

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    /* ws_weight is gone — generate_skeleton.py::_backend_pack_weight
     * pre-packs weights to flat HWIO at codegen time, so we pass
     * `weight` straight into tiled_matmul_auto below. */
    static elem_t ws_input  [WS_BYTES]     __attribute__((aligned(64)));
    static elem_t ws_output [WS_BYTES]     __attribute__((aligned(64)));
    static elem_t ws_im2col [IM2COL_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[ACC_ELEMS]    __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    /* Fall back to scalar for configs exceeding workspace or using offsets.
     * offsets != 0 would require zero-point subtraction inside the GEMM, which
     * tiled_matmul_auto does not support; gemmini_im2col_full_C assumes
     * symmetric per-tensor int8 (offsets == 0 from extract_int8). */
    if (input_offset != 0 || filter_offset != 0
            || (size_t)(N * IH * IW * IC) > WS_BYTES
            || (size_t)(K_inner * OC)      > WS_BYTES
            || (size_t)(N * OH * OW * OC)  > WS_BYTES
            || K_inner * DIM               > IM2COL_ELEMS
            || OC * DIM                    > ACC_ELEMS) {
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
                                    /* weight is HWIO-packed:
                                     * idx = ((kh*KW + kw)*IC + ic)*OC + oc */
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

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Transpose input NCHW → NHWC into ws_input. */
    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

    /* Weight is already HWIO-packed by the codegen
     * (generate_skeleton.py::_backend_pack_weight, --backend gemmini).
     * Layout = `[KH*KW*IC, OC]` flat — exactly the B-matrix layout
     * tiled_matmul_auto wants — so no ws_weight copy needed; we'll
     * pass `weight` directly to tiled_matmul_auto below. */

    /* Drain CPU store buffer (covers ws_input writes — the weight
     * was a const blob so already coherent, but keep the fence here
     * as gemmini mvin sets up A and B together). */
    asm volatile("fence" ::: "memory");

    /* Process output positions in tiles of DIM rows. */
    for (int tile_i = 0; tile_i < total_out; tile_i += DIM) {
        int tile_rows = total_out - tile_i < DIM ? total_out - tile_i : DIM;

        /* Build im2col A-matrix: DIM rows × K_inner cols.
         * Row i holds the flattened receptive field for output position
         * (tile_i + i).  Rows past tile_rows are zero-padded. */
        for (int i = 0; i < DIM; i++) {
            elem_t *row = &ws_im2col[i * K_inner];
            if (i >= tile_rows) {
                for (int k = 0; k < K_inner; k++) row[k] = 0;
                continue;
            }
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh_idx * SH - PH + kh;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow_idx * SW - PW + kw;
                    elem_t *cell = row + (kh * KW + kw) * IC;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        const elem_t *src = &ws_input[((n_idx*IH + ih)*IW + iw)*IC];
                        for (int c = 0; c < IC; c++) cell[c] = src[c];
                    } else {
                        for (int c = 0; c < IC; c++) cell[c] = 0;
                    }
                }
            }
        }

        /* Drain CPU stores to ws_im2col before gemmini mvin. */
        asm volatile("fence" ::: "memory");

        /* GEMM: ws_im2col [DIM × K_inner] × weight [K_inner × OC] + bias[OC].
         * weight is pre-packed HWIO from codegen — flat [KH*KW*IC, OC] is
         * exactly the B-matrix layout tiled_matmul_auto wants.
         * full_C=true → raw int32 accumulator output (no float-scale mvout). */
        tiled_matmul_auto(
            DIM, OC, K_inner,
            ws_im2col, weight,
            (const void *)bias, (void *)ws_acc_out,
            K_inner, OC, OC, OC,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
            bias != NULL,
            false, false,
            true, false,
            0, WS
        );

        /* Wait for gemmini DMA writes to ws_acc_out to reach L2.
         * gemmini_fence drains the in-flight mvout DMAs; gemmini_flush
         * resets the controller. Both are needed — without the fence,
         * yolov8-scale tiles (DIM * OC * K_inner large) can race with
         * the CPU's subsequent ws_acc_out reads in the requantize loop,
         * which corrupts the stack and surfaces as mcause=1 mepc=0
         * (return-address-zeroed) several frames later. See
         * modelblaster/notes/gemmini_tiled_conv_fence_required.md. */
        gemmini_fence();
        gemmini_flush(0);

        /* Scalar Q0.31 requantize: int32 accumulator → int8 NHWC.
         * Uses int64 arithmetic throughout to avoid UB on large output_shift. */
        for (int i = 0; i < tile_rows; i++) {
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            for (int oc = 0; oc < OC; oc++) {
                int32_t acc = ws_acc_out[i * OC + oc];
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
                ws_output[((n_idx*OH + oh_idx)*OW + ow_idx)*OC + oc] = (elem_t)scaled;
            }
        }
    }

    /* Transpose output NHWC → NCHW. */
    for (int n = 0; n < N; n++)
        for (int c = 0; c < OC; c++)
            for (int h = 0; h < OH; h++)
                for (int w = 0; w < OW; w++)
                    output[((n*OC + c)*OH + h)*OW + w] =
                        ws_output[((n*OH + h)*OW + w)*OC + c];
}
