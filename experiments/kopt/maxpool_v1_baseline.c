/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV maxpool2d_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_nc = input + (n*C + c)*IH*IW;
            int8_t *out_nc = output + (n*C + c)*OH*OW;

            for (int oh = 0; oh < OH; oh++) {
                int ow = 0;
                size_t vl;

                /* Vectorize over OW dimension */
                for (; ow < OW; ow += vl) {
                    vl = __riscv_vsetvl_e8m4(OW - ow);

                    /* Initialize accumulator with INT8_MIN */
                    vint8m4_t vacc = __riscv_vmv_v_x_i8m4((int8_t)(-128), vl);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;

                        const int8_t *in_row = in_nc + ih*IW;

                        for (int kw = 0; kw < KW; kw++) {
                            /* Base input column for first output in this vector chunk */
                            /* output ow+lane corresponds to input (ow+lane)*SW - PW + kw*DW */
                            /* For SW==1: iw = ow + kw*DW - PW, contiguous load */
                            /* For SW>1: strided load with stride SW bytes */

                            int iw_base = ow*SW - PW + kw*DW;

                            if (SW == 1) {
                                /* Check if any element in [iw_base, iw_base+vl) is valid */
                                /* For contiguous load, handle boundary with masking */
                                if (iw_base >= IW || iw_base + (int)vl <= 0) continue;

                                if (iw_base >= 0 && iw_base + (int)vl <= IW) {
                                    /* Fully in bounds: simple load */
                                    vint8m4_t vdata = __riscv_vle8_v_i8m4(in_row + iw_base, vl);
                                    vacc = __riscv_vmax_vv_i8m4(vacc, vdata, vl);
                                } else {
                                    /* Partial: scalar fallback for this kw */
                                    for (size_t lane = 0; lane < vl; lane++) {
                                        int iw = iw_base + (int)lane;
                                        if (iw >= 0 && iw < IW) {
                                            int8_t v = in_row[iw];
                                            /* We need to update lane 'lane' of vacc */
                                            /* Use scalar approach for boundary cases */
                                            /* Extract, compare, insert is expensive;
                                               just do scalar for the whole chunk */
                                        }
                                    }
                                    /* Fall back to scalar for this kw */
                                    for (size_t lane = 0; lane < vl; lane++) {
                                        int iw = iw_base + (int)lane;
                                        if (iw >= 0 && iw < IW) {
                                            /* We need per-lane update - use slide approach */
                                            /* Simpler: just do scalar accumulation */
                                        }
                                    }
                                    /* Actually handle with scalar for boundary kw */
                                    goto scalar_kw;
                                }
                                continue;
                            scalar_kw:;
                                /* scalar path for boundary */
                                for (size_t lane = 0; lane < vl; lane++) {
                                    int iw = iw_base + (int)lane;
                                    if (iw >= 0 && iw < IW) {
                                        int8_t cur = __riscv_vmv_x_s_i8m4_i8(
                                            __riscv_vslidedown_vx_i8m4(vacc, lane, 1));
                                        int8_t v = in_row[iw];
                                        if (v > cur) {
                                            /* Can't easily update single lane; use scalar acc */
                                        }
                                    }
                                }
                                /* This is getting complex - just do full scalar for boundary */
                                {
                                    /* Write back vacc to temp, update, reload */
                                    int8_t tmp[256];
                                    __riscv_vse8_v_i8m4(tmp, vacc, vl);
                                    for (size_t lane = 0; lane < vl; lane++) {
                                        int iw = iw_base + (int)lane;
                                        if (iw >= 0 && iw < IW) {
                                            int8_t v = in_row[iw];
                                            if (v > tmp[lane]) tmp[lane] = v;
                                        }
                                    }
                                    vacc = __riscv_vle8_v_i8m4(tmp, vl);
                                }
                            } else {
                                /* SW > 1: strided load */
                                if (iw_base >= IW) continue;
                                /* Check bounds for all lanes */
                                int iw_last = iw_base + (int)(vl-1)*SW;
                                if (iw_base >= 0 && iw_last < IW) {
                                    /* All in bounds */
                                    vint8m4_t vdata = __riscv_vlse8_v_i8m4(
                                        in_row + iw_base,
                                        (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t),
                                        vl);
                                    vacc = __riscv_vmax_vv_i8m4(vacc, vdata, vl);
                                } else {
                                    /* Boundary: scalar fallback */
                                    int8_t tmp[256];
                                    __riscv_vse8_v_i8m4(tmp, vacc, vl);
                                    for (size_t lane = 0; lane < vl; lane++) {
                                        int iw = iw_base + (int)lane*SW;
                                        if (iw >= 0 && iw < IW) {
                                            int8_t v = in_row[iw];
                                            if (v > tmp[lane]) tmp[lane] = v;
                                        }
                                    }
                                    vacc = __riscv_vle8_v_i8m4(tmp, vl);
                                }
                            }
                        }
                    }

                    __riscv_vse8_v_i8m4(out_nc + oh*OW + ow, vacc, vl);
                }
            }
        }
    }
}