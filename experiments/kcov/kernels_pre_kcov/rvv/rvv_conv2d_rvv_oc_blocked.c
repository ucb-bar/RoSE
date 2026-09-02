/* source: curated */
/* algorithm: rvv_oc_blocked */
/* accuracy_class: numeric_drift */   /* fp32 accumulation reorder */
/* origin: fp32 port of rvv_conv2d_s8_rvv_oc_blocked.c. Vectorizes across
 *         the OUTPUT-CHANNEL dimension: for each (n, oh, ow) output
 *         position we hold a vl-wide vector of fp32 accumulators (one per
 *         OC lane) and fold in each (ic, kh, kw) contribution with a
 *         single vfmacc — the input pixel is a scalar broadcast, the OC
 *         weights are a unit-stride vector load (weights are physically
 *         IHWOC, i.e. [IC][KH][KW][OC] with OC innermost/contiguous).
 *
 *         The outer oc_outer tile keeps a TILE_OC-slab of weights resident
 *         in L1D across the whole OH*OW spatial sweep, cutting weight LLC
 *         traffic ~OH*OW-fold on FireSim (spike's flat memory won't care).
 *         Output is NCHW so the store over OC lanes is strided by OH*OW. */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW,
                   int DH, int DW)
{
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    const ptrdiff_t oc_stride_elems = (ptrdiff_t)IC * KH * KW;   /* weight OC slab */
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * OW * sizeof(float);

    /* Tile the OC dimension so one weight slab fits in ~24 KB of L1D.
     * TILE_OC * IC*KH*KW * 4 bytes <= budget. Round to a multiple of the
     * fp32 m4 vlmax so the inner vsetvl loop has no extra tail iteration;
     * when a single OC slab already busts the budget, fall back to vlmax
     * (one inner pass = graceful degradation to un-blocked). */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const long oc_slab_bytes = (long)oc_stride_elems * (long)sizeof(float);
    int TILE_OC;
    if (oc_slab_bytes > 0 && oc_slab_bytes <= L1D_OC_BUDGET_BYTES) {
        TILE_OC = (int)((long)L1D_OC_BUDGET_BYTES / oc_slab_bytes);
        if (TILE_OC > vlmax_oc)
            TILE_OC = (TILE_OC / vlmax_oc) * vlmax_oc;
        else
            TILE_OC = vlmax_oc;
    } else {
        TILE_OC = vlmax_oc;
    }
    if (TILE_OC > OC) TILE_OC = OC;
    if (TILE_OC <= 0) TILE_OC = OC;

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

                        vfloat32m4_t vacc;
                        if (bias != NULL)
                            vacc = __riscv_vle32_v_f32m4(bias + oc_base, vl);
                        else
                            vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh * DH;
                                if (ih < 0 || ih >= IH) continue;
                                const size_t row_off =
                                    (((size_t)n * IC + ic) * IH + ih) * (size_t)IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw * DW;
                                    if (iw < 0 || iw >= IW) continue;
                                    float in_v = input[row_off + iw];

                                    /* IHWOC weight: [IC][KH][KW][OC], OC
                                     * contiguous. Unit-stride vector load of
                                     * the OC slab starting at oc_base. */
                                    const float *wp = weight
                                        + ((size_t)(ic * KH + kh) * KW + kw) * OC
                                        + oc_base;
                                    vfloat32m4_t vw = __riscv_vle32_v_f32m4(wp, vl);

                                    vacc = __riscv_vfmacc_vf_f32m4(vacc, in_v, vw, vl);
                                }
                            }
                        }

                        /* NCHW output: consecutive OC lanes are OH*OW apart. */
                        float *op = output
                            + ((size_t)n * OC + oc_base) * OH * OW
                            + (size_t)oh * OW + ow;
                        __riscv_vsse32_v_f32m4(op, out_c_stride_bytes, vacc, vl);

                        oc_base += (int)vl;
                    }
                }
            }
        }
    }
}
