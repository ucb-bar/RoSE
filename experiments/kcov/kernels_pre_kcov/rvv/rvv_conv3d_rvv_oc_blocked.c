/* source: curated */
/* algorithm: rvv_oc_blocked */
/* accuracy_class: numeric_drift */   /* fp32 accumulation reorder */
/* origin: fp32 RVV conv3d, modeled on rvv_conv2d_rvv_oc_blocked.c but for
 *         the NCDHW 3D-gather family. Vectorizes across the OUTPUT-CHANNEL
 *         dimension: for each (n, od, oh, ow) output voxel we hold a vl-wide
 *         vector of fp32 accumulators (one lane per OC) and fold in every
 *         (icg, kd, kh, kw) contribution with a single vfmacc.vf — the input
 *         voxel is a scalar broadcast, the OC weights are a vector load.
 *
 *         KEY LAYOUT DIFFERENCE vs conv2d: conv3d weights are 5D
 *         [OC][IC/G][KD][KH][KW] and are NOT backend-repacked
 *         (generate_skeleton._backend_pack_weight only permutes 4D tensors,
 *         so the ihwoc trick is unavailable). OC is therefore the OUTERMOST
 *         weight axis with stride IC/G*KD*KH*KW, so the per-(icg,kd,kh,kw) OC
 *         slab is non-contiguous; it is repacked into a small
 *         contiguous stack buffer and loaded UNIT-STRIDE (see below).
 *         Bias is OC-contiguous (unit-stride vle32). Output is NCDHW so the
 *         per-OC store is strided by OD*OH*OW (vsse32).
 *
 *         Groups: OC lanes are slabbed WITHIN a single group (a slab never
 *         crosses a group boundary) because the input-channel base ic depends
 *         on g = oc/OCpG. */

#include <stddef.h>
#include <riscv_vector.h>

/* Max e32,m4 vector length (elements) across supported VLEN configs; the OC
 * tile is repacked into this stack buffer so the vector load is unit-stride.
 * Sized generously (covers VLEN up to 2048) — a slab never exceeds vl <= this. */
#define KCONV3D_VLMAX 256

void kernel_conv3d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int ID, int IH, int IW,
                   int OC, int OD, int OH, int OW,
                   int KD, int KH, int KW, int SD, int SH, int SW,
                   int PD, int PH, int PW, int DD, int DH, int DW, int G)
{
    const int ICpG = IC / G;
    const int OCpG = OC / G;

    /* Weight OC stride (elements): for fixed (icg,kd,kh,kw), consecutive oc
     * are this far apart in the [OC][ICpG][KD][KH][KW] buffer. */
    const ptrdiff_t w_oc_stride_elems = (ptrdiff_t)ICpG * KD * KH * KW;
    /* NCDHW output: consecutive OC lanes are OD*OH*OW apart. */
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OD * OH * OW * sizeof(float);

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            const int oc_lo = g * OCpG;
            const int oc_hi = oc_lo + OCpG;
            const int ic_base = g * ICpG;

            for (int od = 0; od < OD; od++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int oc = oc_lo;
                        while (oc < oc_hi) {
                            size_t vl = __riscv_vsetvl_e32m4(
                                (size_t)(oc_hi - oc));

                            vfloat32m4_t vacc;
                            if (bias != NULL)
                                vacc = __riscv_vle32_v_f32m4(bias + oc, vl);
                            else
                                vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                            for (int kd = 0; kd < KD; kd++) {
                                int id = od * SD - PD + kd * DD;
                                if (id < 0 || id >= ID) continue;
                                for (int kh = 0; kh < KH; kh++) {
                                    int ih = oh * SH - PH + kh * DH;
                                    if (ih < 0 || ih >= IH) continue;
                                    for (int kw = 0; kw < KW; kw++) {
                                        int iw = ow * SW - PW + kw * DW;
                                        if (iw < 0 || iw >= IW) continue;
                                        for (int icg = 0; icg < ICpG; icg++) {
                                            int ic = ic_base + icg;
                                            float in_v = input[
                                                (((size_t)(n * IC + ic) * ID + id)
                                                 * IH + ih) * (size_t)IW + iw];

                                            /* OIDHW weight, OC-strided in memory.
                                             * base = weight[oc][icg][kd][kh][kw].
                                             * Repack the OC-strided slab into a
                                             * contiguous stack buffer so the
                                             * vector load is UNIT-STRIDE
                                             * instead of large-stride strided,
                                             * which the Saturn V256D128 vector
                                             * unit miscomputes. */
                                            const float *wp = weight
                                                + ((((size_t)oc * ICpG + icg) * KD
                                                    + kd) * KH + kh) * (size_t)KW
                                                + kw;
                                            float wtmp[KCONV3D_VLMAX];
                                            for (size_t l = 0; l < vl; l++)
                                                wtmp[l] = wp[(size_t)l
                                                             * w_oc_stride_elems];
                                            vfloat32m4_t vw =
                                                __riscv_vle32_v_f32m4(wtmp, vl);

                                            vacc = __riscv_vfmacc_vf_f32m4(
                                                vacc, in_v, vw, vl);
                                        }
                                    }
                                }
                            }

                            float *op = output
                                + (((size_t)(n * OC + oc) * OD + od) * OH + oh)
                                  * (size_t)OW + ow;
                            __riscv_vsse32_v_f32m4(op, out_c_stride_bytes,
                                                   vacc, vl);

                            oc += (int)vl;
                        }
                    }
                }
            }
        }
    }
}
