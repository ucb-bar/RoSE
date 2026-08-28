/* source: curated */
/* algorithm: direct */
/* origin: RVV sigmoid_s8 via a range-limited int8 LUT + vluxei8 gather.
 *
 * sigmoid_s8 is a byte-in / byte-out map, so every distinct answer can be
 * tabulated once and then gathered at ~16 lanes/cycle. The catch is that
 * building the table costs one expf per DISTINCT input code, which is the
 * same price the scalar reference pays per ELEMENT — so the table only pays
 * for itself when the tensor has more elements than it has distinct codes.
 * This kernel measures that directly instead of guessing:
 *
 *   1. n below a floor: no scan, straight to the exact scalar loop. (dronet's
 *      single sigmoid is n=1 at the model head and lands here.)
 *   2. vector min/max scan -> the table only has to span the codes actually
 *      present, not all 256.
 *   3. entries >= n: the table can never amortize, so still use the scalar
 *      loop -- the scan was cheap.
 *   4. otherwise: build the range-limited table and gather.
 *
 * The scalar path is character-for-character the reference computation, so
 * every element it produces is bit-identical to it, and the LUT path is the
 * same computation evaluated once per distinct code. */

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max) {
    if (n <= 0) {
        return;
    }

    int lo_v = 127;
    int hi_v = -128;
    int entries = 256;

    if (n >= 16) {
        /* Vector min/max scan. The running extremes are carried in scalars,
         * so the short tail chunk needs no tail-undisturbed accumulator. */
        int i = 0;
        size_t vl;
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
            vint8m1_t rmin = __riscv_vredmin_vs_i8m8_i8m1(
                v, __riscv_vmv_v_x_i8m1((int8_t)lo_v, 1), vl);
            vint8m1_t rmax = __riscv_vredmax_vs_i8m8_i8m1(
                v, __riscv_vmv_v_x_i8m1((int8_t)hi_v, 1), vl);
            lo_v = __riscv_vmv_x_s_i8m1_i8(rmin);
            hi_v = __riscv_vmv_x_s_i8m1_i8(rmax);
        }
        entries = hi_v - lo_v + 1;
    }

    if (n < 16 || entries >= n) {
        for (int i = 0; i < n; i++) {
            float fv = (float)input[i] * scale_in;
            float sig = 1.0f / (1.0f + expf(-fv));
            int32_t v = (int32_t)roundf(sig / scale_out);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i] = (int8_t)v;
        }
        return;
    }

    /* Table over the observed codes only; (hi_v - lo_v) <= 255 so 256 entries
     * always suffice. */
    int8_t lut[256];
    for (int e = 0; e < entries; e++) {
        float fv = (float)(int8_t)(lo_v + e) * scale_in;
        float sig = 1.0f / (1.0f + expf(-fv));
        int32_t v = (int32_t)roundf(sig / scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        lut[e] = (int8_t)v;
    }

    /* Gather. (x - lo_v) is evaluated in wrapping int8 arithmetic and read
     * back as an unsigned byte, which is exactly the table offset across the
     * whole 0..255 span -- no widening to a 16-bit index needed. */
    {
        int i = 0;
        size_t vl;
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
            vint8m8_t voff = __riscv_vsub_vx_i8m8(v, (int8_t)lo_v, vl);
            vuint8m8_t vidx = __riscv_vreinterpret_v_i8m8_u8m8(voff);
            vuint8m8_t vout = __riscv_vluxei8_v_u8m8(
                (const uint8_t *)lut, vidx, vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
        }
    }
}
