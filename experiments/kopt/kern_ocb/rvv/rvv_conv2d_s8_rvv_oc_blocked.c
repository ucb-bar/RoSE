/* source: curated */
/* algorithm: rvv_oc_blocked */
/* origin: cache-aware variant of rvv_vsmul_vnclip. The un-blocked
 *         version walks the entire weight tensor for every (n, oh, ow)
 *         output position — for a 3x3 IC=128 OC=128 conv that's
 *         144 KB cycling through L1D once per output position
 *         (OH*OW=16 reloads from LLC per inference). This kernel
 *         tiles the OC dimension so a TILE_OC-slab of weights stays
 *         resident in L1D across the whole spatial sweep, then moves
 *         on to the next OC tile.
 *
 *         Reuse improvement (dronet 3x3 IC=128 OC=128 OH=OW=4):
 *           weight LLC traffic = OH*OW * IC*KH*KW * OC bytes
 *                              = 16 * 144 KB         (un-blocked)
 *                              = 2.3 MB per layer-pass
 *           with TILE_OC=16:    OC/TILE_OC tiles * tile_bytes
 *                              = 8 * 18 KB           (blocked)
 *                              = 144 KB
 *           ratio:             ~16x less LLC traffic on weights.
 *
 *         The inner reduction is identical to rvv_vsmul_vnclip — same
 *         vsmul/vnclip Q0.31 requantize tail, same OC-strided
 *         vector-load pattern. The only structural change is the
 *         outer oc_outer loop. Spike's flat-memory model won't reward
 *         the rewrite (no cache misses); FireSim does. */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;

    /* Tile size selection. Goal: keep TILE_OC * IC*KH*KW int8 weights
     * resident in L1D (32 KB on quad-rocket-saturn). Leave headroom
     * for input window, output write set, and stack frame.
     *
     * Constraints:
     *   - TILE_OC must be a multiple of LMUL=4 i32 lanes (vlmax_e32m4)
     *     so the inner vsetvl loop doesn't run a tail iteration that
     *     pays the per-tile overhead twice.
     *   - When IC*KH*KW already exceeds the budget (i.e. each OC slab
     *     is huge — IC=512, KH=3 etc.), shrink TILE_OC down to vlmax;
     *     the tile no longer fits L1D but the outer loop is a no-op
     *     (TILE_OC equals one inner iteration) and we degrade
     *     gracefully to the un-blocked behavior. */
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

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

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
                                /* Hoist row offset to size_t — see the
                                 * matching comment in rvv_vsmul_vnclip.
                                 * Without this, GCC's 32-bit index
                                 * arithmetic wraps when the BSS-placed
                                 * input buffer's low 32 bits + the row
                                 * partial sum cross the int32 sign
                                 * boundary. Triggered on V512D256 firesim. */
                                const size_t row_off =
                                    (((size_t)n * IC + ic) * IH + ih) * (size_t)IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int8_t in_byte = 0;
                                    if (row_in && iw >= 0 && iw < IW)
                                        in_byte = input[row_off + iw];
                                    int32_t in_v = (int32_t)in_byte + input_offset;

                                    /* IHWOC: weight[ic][kh][kw][oc] — OC contiguous.
                                     * With OC blocking, the [oc_outer..oc_end)
                                     * slab stays hot in L1D across the entire
                                     * (n, oh, ow) sweep. */
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

                        /* ---- Q0.31 requantize tail (identical to vsmul_vnclip). */
                        vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
                            vacc, output_multiplier, __RISCV_VXRM_RNU, vl);

                        vint16m2_t vout16;
                        if (output_shift < 0) {
                            vint32m4_t vshifted = __riscv_vsll_vx_i32m4(
                                vscaled, (size_t)(-output_shift), vl);
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vshifted, 0, __RISCV_VXRM_RNU, vl);
                        } else if (output_shift < 32) {
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vscaled, (size_t)output_shift, __RISCV_VXRM_RNU, vl);
                        } else {
                            int sa2 = output_shift - 31;
                            if (sa2 > 31) sa2 = 31;
                            vint32m4_t vscaled2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vscaled2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
                        }

                        vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
                        vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
                        vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);

                        vint8m1_t vout8 = __riscv_vnsra_wx_i8m1(vout16, 0, vl);

                        int8_t *op = output
                            + ((size_t)n * OC + oc_base) * OH * OW
                            + (size_t)oh * OW + ow;
                        int8_t _obuf[256];
                        __riscv_vse8_v_i8m1(_obuf, vout8, vl);
                        for (size_t _vi = 0; _vi < vl; _vi++)
                            op[_vi * (ptrdiff_t)(OH * OW)] = _obuf[_vi];

                        oc_base += (int)vl;
                    }
                }
            }
        }
    }
}
