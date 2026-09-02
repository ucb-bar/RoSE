/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 maxpool2d (vectorized over OW, strided loads). */

#include <float.h>
#include <riscv_vector.h>

void kernel_maxpool2d(const float *input, float *output,
                     int N, int C, int IH, int IW,
                     int KH, int KW, int SH, int SW,
                     int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *in_nc = input + ((size_t)(n*C + c)) * IH * IW;
            float *out_nc = output + ((size_t)(n*C + c)) * OH * OW;

            for (int oh = 0; oh < OH; oh++) {
                size_t vl;
                for (int ow = 0; ow < OW; ow += vl) {
                    vl = __riscv_vsetvl_e32m8(OW - ow);

                    /* accumulator holds the running max over the pooling window,
                       one output column per lane (columns ow .. ow+vl-1). */
                    vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh * DH;
                        if (ih < 0 || ih >= IH) continue;
                        const float *in_row = in_nc + ih * IW;

                        for (int kw = 0; kw < KW; kw++) {
                            /* lane l -> input column (ow+l)*SW - PW + kw*DW */
                            int iw_base = ow * SW - PW + kw * DW;
                            int iw_last = iw_base + (int)(vl - 1) * SW;

                            if (iw_base >= 0 && iw_last < IW) {
                                /* all lanes in-bounds: strided gather */
                                vfloat32m8_t vdata = __riscv_vlse32_v_f32m8(
                                    in_row + iw_base,
                                    (ptrdiff_t)SW * (ptrdiff_t)sizeof(float),
                                    vl);
                                vacc = __riscv_vfmax_vv_f32m8(vacc, vdata, vl);
                            } else {
                                /* boundary chunk: build a lane vector, filling
                                   out-of-bounds lanes with -FLT_MAX so they can
                                   never win the max. */
                                float tmp[512];
                                for (size_t l = 0; l < vl; l++) {
                                    int iw = iw_base + (int)l * SW;
                                    tmp[l] = (iw >= 0 && iw < IW)
                                                 ? in_row[iw]
                                                 : -FLT_MAX;
                                }
                                vfloat32m8_t vdata =
                                    __riscv_vle32_v_f32m8(tmp, vl);
                                vacc = __riscv_vfmax_vv_f32m8(vacc, vdata, vl);
                            }
                        }
                    }

                    __riscv_vse32_v_f32m8(out_nc + oh * OW + ow, vacc, vl);
                }
            }
        }
    }
}
