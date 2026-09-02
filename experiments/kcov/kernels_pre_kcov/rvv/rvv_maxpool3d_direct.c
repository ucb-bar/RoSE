/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 maxpool3d (vectorized over OW, strided loads). */

#include <float.h>
#include <riscv_vector.h>

void kernel_maxpool3d(const float *input, float *output,
                      int N, int C, int ID, int IH, int IW,
                      int OD, int OH, int OW,
                      int KD, int KH, int KW, int SD, int SH, int SW,
                      int PD, int PH, int PW, int DD, int DH, int DW) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *in_nc =
                input + ((long)(n*C + c)) * ID * IH * IW;
            float *out_nc =
                output + ((long)(n*C + c)) * OD * OH * OW;

            for (int od = 0; od < OD; od++) {
                for (int oh = 0; oh < OH; oh++) {
                    float *out_row = out_nc + ((long)od * OH + oh) * OW;
                    size_t vl;
                    for (int ow = 0; ow < OW; ow += vl) {
                        vl = __riscv_vsetvl_e32m8(OW - ow);
                        vfloat32m8_t vacc =
                            __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);

                        for (int kd = 0; kd < KD; kd++) {
                            int id = od*SD - PD + kd*DD;
                            if (id < 0 || id >= ID) continue;
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh*SH - PH + kh*DH;
                                if (ih < 0 || ih >= IH) continue;
                                const float *in_row =
                                    in_nc + ((long)id * IH + ih) * IW;

                                for (int kw = 0; kw < KW; kw++) {
                                    int iw_base = ow*SW - PW + kw*DW;
                                    int iw_last = iw_base + (int)(vl - 1) * SW;

                                    if (iw_base >= 0 && iw_last < IW) {
                                        vfloat32m8_t vdata =
                                            __riscv_vlse32_v_f32m8(
                                                in_row + iw_base,
                                                (ptrdiff_t)SW *
                                                    (ptrdiff_t)sizeof(float),
                                                vl);
                                        vacc = __riscv_vfmax_vv_f32m8(
                                            vacc, vdata, vl);
                                    } else {
                                        float tmp[512];
                                        for (size_t l = 0; l < vl; l++) {
                                            int iw = iw_base + (int)l * SW;
                                            tmp[l] = (iw >= 0 && iw < IW)
                                                         ? in_row[iw]
                                                         : -FLT_MAX;
                                        }
                                        vfloat32m8_t vdata =
                                            __riscv_vle32_v_f32m8(tmp, vl);
                                        vacc = __riscv_vfmax_vv_f32m8(
                                            vacc, vdata, vl);
                                    }
                                }
                            }
                        }

                        __riscv_vse32_v_f32m8(out_row + ow, vacc, vl);
                    }
                }
            }
        }
    }
}
