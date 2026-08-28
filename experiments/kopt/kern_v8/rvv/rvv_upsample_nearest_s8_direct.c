/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV upsample_nearest_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                 int N, int C, int IH, int IW, int scale) {
    int OH = IH * scale, OW = IW * scale;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_plane  = input  + (n*C + c)*IH*IW;
            int8_t       *out_plane = output + (n*C + c)*OH*OW;
            for (int ih = 0; ih < IH; ih++) {
                const int8_t *in_row = in_plane + ih * IW;
                /* First, build one output row by replicating each input pixel
                 * 'scale' times. We'll write this into the first of the
                 * 'scale' output rows, then memcpy the rest. */
                int8_t *out_row0 = out_plane + ih * scale * OW;

                /* Build the replicated row using RVV */
                int ow = 0;
                for (int iw = 0; iw < IW; iw++) {
                    int8_t val = in_row[iw];
                    int rem = scale;
                    size_t vl;
                    /* Fill 'scale' output positions with val */
                    for (int s = 0; s < rem; ) {
                        vl = __riscv_vsetvl_e8m8(rem - s);
                        vint8m8_t vval = __riscv_vmv_v_x_i8m8(val, vl);
                        __riscv_vse8_v_i8m8(out_row0 + ow + s, vval, vl);
                        s += vl;
                    }
                    ow += scale;
                }

                /* Now copy out_row0 to the remaining (scale-1) output rows */
                for (int sr = 1; sr < scale; sr++) {
                    int8_t *out_rowN = out_row0 + sr * OW;
                    /* Copy OW bytes using RVV */
                    int rem = OW;
                    int off = 0;
                    size_t vl;
                    while (rem > 0) {
                        vl = __riscv_vsetvl_e8m8(rem);
                        vint8m8_t v = __riscv_vle8_v_i8m8(out_row0 + off, vl);
                        __riscv_vse8_v_i8m8(out_rowN + off, v, vl);
                        off += vl;
                        rem -= vl;
                    }
                }
            }
        }
    }
}