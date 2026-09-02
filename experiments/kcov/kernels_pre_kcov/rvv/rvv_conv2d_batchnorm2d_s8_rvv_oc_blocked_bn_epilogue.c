/* source: curated */
/* algorithm: rvv_oc_blocked_bn_epilogue */
/* accuracy_class: bit_exact */
/* origin: rvv_conv2d_s8_rvv_oc_blocked.c + the BN stage folded into its
 *         store path. The Conv->BN->SiLU sibling lives in
 *         rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c
 *         and the two share their reasoning; only the extra SiLU table
 *         differs.
 *
 *   WHY THIS FILE EXISTS. Curated kernels are looked up by EXACT op name
 *   (kernels/<backend>/<backend>_<op>_<algorithm>.c). The graph fuses
 *   Conv->BN into `conv2d_batchnorm2d_s8`, a name the RVV library did not
 *   have, so DroNet's three fused convolutions resolved to the SCALAR
 *   reference -- 86.7% of a 62.6 ms rvv_x60 run measured on the K1, in a
 *   build labelled as vector. Having rvv_conv2d_s8 and rvv_batchnorm2d_s8
 *   separately does not help: the fused name is what is probed.
 *
 *   STRUCTURE. The convolution (OC in the vector lanes, MAC reduction over
 *   (ic, kh, kw), Q0.31 requantize tail, OC cache tiling, IHWOC weight
 *   indexing) is unchanged from rvv_conv2d_s8_rvv_oc_blocked.c. The BN
 *   stage is applied to the conv's int8 result on its way to memory, so
 *   the intermediate tensor is never materialized.
 *
 *   WHY THE EPILOGUE IS SCALAR. After the conv requantize the value is an
 *   int8, so BN has at most 256 distinct outputs per channel and reduces
 *   to a per-channel table wherever the spatial extent amortizes a
 *   256-entry build (OH*OW >= LUT_BREAKEVEN; below that the build costs
 *   more than it saves -- the guard
 *   rvv_opu_batchnorm2d_s8_per_channel_lut.c established). Either way the
 *   float arithmetic is written as the same expression the reference uses
 *   and executed on the scalar unit: reproducing roundf() in the vector
 *   unit would mean matching a rounding mode vfcvt does not have (ties
 *   away from zero, not RNE) and issuing float ops inside a kernel whose
 *   vtype is otherwise pure integer -- the vfmv.v.f-under-SEW=8 SIGILL
 *   that rvv_batchnorm2d_s8_direct.c documents. Bit-exact by construction
 *   rather than by tolerance.
 *
 *   VTYPE. The vector body is integer-only (i32m4 accumulator, i16m2
 *   requantize, i8m1 weights/output) and every width transition is named
 *   by its own intrinsic. Checked with scripts/check_rvv_vtype.py.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

/* Per-OC-tile LUT slab: MB_CB_TILE_OC * 256 bytes of stack (16 KB). */
#define MB_CB_TILE_OC   64
#define MB_CB_LUT_BREAKEVEN 256

/* BN stage, written exactly as in the fused op's reference_impl: same
 * expressions, same order, same casts -- so the compiler contracts and
 * rounds it identically and the two agree bit-for-bit. */
static inline int8_t mb_cb_bn_stage(int8_t conv_int8,
                                    float bn_s, float bn_b,
                                    float bn_scale_in, float bn_scale_out,
                                    int bn_activation_min,
                                    int bn_activation_max)
{
    float fv = (float)conv_int8 * bn_scale_in;
    float y = bn_s * fv + bn_b;
    int32_t v = (int32_t)roundf(y / bn_scale_out);
    if (v < bn_activation_min) v = bn_activation_min;
    if (v > bn_activation_max) v = bn_activation_max;
    return (int8_t)v;
}

void kernel_conv2d_batchnorm2d_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;
    const int spatial = OH * OW;

    /* Tile size selection -- identical to rvv_conv2d_s8_rvv_oc_blocked.c
     * (keep TILE_OC * IC*KH*KW int8 weights resident in L1D), plus the
     * LUT slab bound when the table path is taken. */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const int oc_slab_bytes = (int)oc_stride;     /* int8 weights */
    int TILE_OC;
    if (oc_slab_bytes > 0 && oc_slab_bytes <= L1D_OC_BUDGET_BYTES) {
        TILE_OC = L1D_OC_BUDGET_BYTES / oc_slab_bytes;
        if (TILE_OC > vlmax_oc)
            TILE_OC = (TILE_OC / vlmax_oc) * vlmax_oc;   /* multiple of vlmax */
        else
            TILE_OC = vlmax_oc;                          /* one inner pass */
    } else {
        TILE_OC = vlmax_oc;
    }
    if (TILE_OC > OC) TILE_OC = OC;
    if (TILE_OC <= 0) TILE_OC = OC;                      /* safety */

    int use_lut = (spatial >= MB_CB_LUT_BREAKEVEN)
                  && (vlmax_oc <= MB_CB_TILE_OC);
    if (use_lut && TILE_OC > MB_CB_TILE_OC) TILE_OC = MB_CB_TILE_OC;

    int8_t epi_lut[MB_CB_TILE_OC][256];

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

        if (use_lut) {
            for (int oc = oc_outer; oc < oc_end; oc++) {
                float bn_s = bn_scale[oc];
                float bn_b = bn_bias[oc];
                int8_t *row = epi_lut[oc - oc_outer];
                for (int v = 0; v < 256; v++) {
                    row[v] = mb_cb_bn_stage(
                        (int8_t)(v - 128), bn_s, bn_b,
                        bn_scale_in, bn_scale_out,
                        bn_activation_min, bn_activation_max);
                }
            }
        }

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int oc_base = oc_outer;
                    while (oc_base < oc_end) {
                        size_t vl = __riscv_vsetvl_e32m4(
                            (size_t)(oc_end - oc_base));

                        vint32m4_t vacc;
                        if (bias != NULL)
                            vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        else
                            vacc = __riscv_vmv_v_x_i32m4(0, vl);

                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                int row_in = (ih >= 0 && ih < IH);
                                /* Hoist the row offset to size_t: GCC's
                                 * 32-bit index arithmetic wraps when a
                                 * BSS-placed input buffer's low 32 bits
                                 * plus the row partial sum cross the
                                 * int32 sign boundary. */
                                const size_t row_off =
                                    (((size_t)n * IC + ic) * IH + ih) * (size_t)IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int8_t in_byte = 0;
                                    if (row_in && iw >= 0 && iw < IW)
                                        in_byte = input[row_off + iw];
                                    int32_t in_v = (int32_t)in_byte + input_offset;

                                    /* IHWOC: weight[ic][kh][kw][oc]. */
                                    const int8_t *wp = weight
                                        + ((size_t)ic * KH * KW + (size_t)kh * KW + kw) * OC
                                        + oc_base;
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wp, vl);

                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);

                                    vacc = __riscv_vwmacc_vx_i32m4(
                                        vacc, (int16_t)in_v, vw16, vl);
                                }
                            }
                        }

                        /* ---- conv2d_s8's Q0.31 requantize tail. */
                        vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
                            vacc, conv_output_multiplier, __RISCV_VXRM_RNU, vl);

                        vint16m2_t vout16;
                        if (conv_output_shift < 0) {
                            vint32m4_t vshifted = __riscv_vsll_vx_i32m4(
                                vscaled, (size_t)(-conv_output_shift), vl);
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vshifted, 0, __RISCV_VXRM_RNU, vl);
                        } else if (conv_output_shift < 32) {
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vscaled, (size_t)conv_output_shift,
                                __RISCV_VXRM_RNU, vl);
                        } else {
                            int sa2 = conv_output_shift - 31;
                            if (sa2 > 31) sa2 = 31;
                            vint32m4_t vscaled2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vscaled2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
                        }

                        vout16 = __riscv_vadd_vx_i16m2(
                            vout16, (int16_t)conv_output_offset, vl);
                        vout16 = __riscv_vmax_vx_i16m2(
                            vout16, (int16_t)conv_activation_min, vl);
                        vout16 = __riscv_vmin_vx_i16m2(
                            vout16, (int16_t)conv_activation_max, vl);

                        vint8m1_t vout8 = __riscv_vnsra_wx_i8m1(vout16, 0, vl);

                        /* ---- BN epilogue, then the strided store. The
                         * output is NCHW and the lanes are the OC axis, so
                         * the store is elementwise with stride OH*OW either
                         * way -- the epilogue rides along in it. */
                        int8_t *op = output
                            + ((size_t)n * OC + oc_base) * OH * OW
                            + (size_t)oh * OW + ow;
                        int8_t _obuf[256];
                        __riscv_vse8_v_i8m1(_obuf, vout8, vl);
                        if (use_lut) {
                            const int lut_base = oc_base - oc_outer;
                            for (size_t _vi = 0; _vi < vl; _vi++)
                                op[_vi * (ptrdiff_t)spatial] =
                                    epi_lut[lut_base + (int)_vi]
                                           [(int)_obuf[_vi] + 128];
                        } else {
                            for (size_t _vi = 0; _vi < vl; _vi++) {
                                int oc = oc_base + (int)_vi;
                                op[_vi * (ptrdiff_t)spatial] = mb_cb_bn_stage(
                                    _obuf[_vi], bn_scale[oc], bn_bias[oc],
                                    bn_scale_in, bn_scale_out,
                                    bn_activation_min, bn_activation_max);
                            }
                        }

                        oc_base += (int)vl;
                    }
                }
            }
        }
    }
}
