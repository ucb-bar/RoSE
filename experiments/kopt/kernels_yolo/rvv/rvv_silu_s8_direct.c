/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV silu_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    /* Precompute LUT for all 256 int8 values.
     * int8 values range from -128 to 127.
     * We index the LUT by unsigned byte (0..255), where value v maps to
     * int8 input (int8_t)v. */
    int8_t lut[256];
    for (int v = 0; v < 256; v++) {
        int8_t iv = (int8_t)(uint8_t)v;
        float f = (float)iv * scale_in;
        float y = f / (1.0f + expf(-f));
        int32_t q = (int32_t)roundf(y / scale_out);
        if (q < activation_min) q = activation_min;
        if (q > activation_max) q = activation_max;
        lut[v] = (int8_t)q;
    }

    /* Use RVV to do vectorized table lookup.
     * We treat input bytes as unsigned indices into the LUT. */
    int i = 0;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        /* Load input bytes as unsigned indices */
        vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
        /* Gather from LUT using unsigned byte indices */
        vuint8m8_t vout = __riscv_vluxei8_v_u8m8((const uint8_t *)lut, vidx, vl);
        __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
    }
}