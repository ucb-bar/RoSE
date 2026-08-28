/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV batchnorm2d_s8, per-channel affine.
 *
 * Four things matter on a Saturn-class decoupled vector unit (VLEN=256,
 * DLEN=128), where an instruction's cost tracks the BYTES it touches, so a
 * 32-bit-wide op moves 4 elements/cycle while an 8/16-bit one moves 16/8:
 *
 *   * i8 -> i16 -> f32 via vsext.vf2 + vfwcvt, so only the second widening
 *     step runs at 32-bit width.
 *   * The per-channel bias is broadcast ONCE per channel, so the inner loop
 *     is a single vfmadd (vd = vd*cs + vcb) rather than a broadcast plus a
 *     vfmacc. The broadcast is issued at min(hw, VLMAX) -- issuing it at
 *     VLMAX wastes most of a vector op on every channel of a late layer
 *     whose whole H*W plane is shorter than one vector (dronet's last BN is
 *     C=128, H*W=16).
 *   * vfncvt.x.f.w converts f32 -> i16 in ONE instruction, and RVV float ->
 *     int conversions already clip to the destination range, so it replaces
 *     the vfcvt + narrow pair AND provides the saturation for free.
 *   * That deliberately avoids vnclip. vnclip is a fixed-point op, so GCC
 *     emits a `csrwi vxrm` for it, and the mode-switching pass only hoists
 *     that out as far as the CHANNEL loop -- on this core a vxrm write costs
 *     ~50 cycles of vector-unit sync PER CHANNEL. Measured on F2: a vnclip
 *     tail beat this one by 18 cyc/chunk on the C=32,H*W=729 layer and lost
 *     by 30 cyc/channel on C=128,H*W=16, i.e. it bought the big layers by
 *     taxing every channel of the small ones.
 *
 * Folding: one FMA of x * (scale[c]*scale_in/scale_out) + bias[c]/scale_out,
 * identical to the previously curated kernel, which verifies bit-exact
 * against the dronet golden. The int16-domain clamp assumes the activation
 * range fits in int8 (what a quantizer emitting an int8 tensor produces);
 * the bounds are clamped into int8 so a wider nominal range saturates rather
 * than wraps. */

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max) {
    const float inv_scale_out = 1.0f / scale_out;
    const int hw = H * W;

    const int16_t lo = (int16_t)(activation_min < -128 ? -128
                                 : (activation_min > 127 ? 127
                                    : activation_min));
    const int16_t hi = (int16_t)(activation_max < -128 ? -128
                                 : (activation_max > 127 ? 127
                                    : activation_max));

    if (hw <= 0) {
        return;
    }
    /* Width of the bias broadcast: never more than one plane's worth. */
    const size_t vl_b = __riscv_vsetvl_e32m8((size_t)hw);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float cs = scale[c] * scale_in * inv_scale_out;
            const float cb = bias[c] * inv_scale_out;
            const int base = (n * C + c) * hw;
            const int8_t *in_ptr = input + base;
            int8_t *out_ptr = output + base;

            vfloat32m8_t vcb = __riscv_vfmv_v_f_f32m8(cb, vl_b);

            int i = 0;
            size_t vl;
            for (; i < hw; i += vl) {
                vl = __riscv_vsetvl_e8m2(hw - i);

                vint8m2_t v8 = __riscv_vle8_v_i8m2(in_ptr + i, vl);
                vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);

                /* vf = vf * cs + vcb */
                vf = __riscv_vfmadd_vf_f32m8(vf, cs, vcb, vl);

                /* f32 -> i16 with round-to-nearest and range clipping. */
                vint16m4_t o16 = __riscv_vfncvt_x_f_w_i16m4(vf, vl);
                o16 = __riscv_vmax_vx_i16m4(o16, lo, vl);
                o16 = __riscv_vmin_vx_i16m4(o16, hi, vl);

                vint8m2_t o8 = __riscv_vncvt_x_x_w_i8m2(o16, vl);
                __riscv_vse8_v_i8m2(out_ptr + i, o8, vl);
            }
        }
    }
}
