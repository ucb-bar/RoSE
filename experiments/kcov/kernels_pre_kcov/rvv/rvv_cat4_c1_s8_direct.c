/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV cat4_c1_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_cat4_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2, const int8_t *in3, int c3, float scale3,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    int total_c = c0 + c1 + c2 + c3;

    const int8_t *ins[4] = { in0, in1, in2, in3 };
    int cs[4] = { c0, c1, c2, c3 };
    float scales[4] = { scale0, scale1, scale2, scale3 };

    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 4; i++) {
            float ratio = scales[i] / scale_out;
            int ci = cs[i];
            const int8_t *in_base = ins[i] + n * ci * stride;
            int8_t *out_base = output + (n * total_c + out_c) * stride;

            /* Check if output channels are contiguous (they are: out_c..out_c+ci-1 are consecutive) */
            /* We can process all ci channels * stride pixels as a flat array */
            int total_elems = ci * stride;
            int hw = 0;
            size_t vl;
            for (; hw < total_elems; hw += vl) {
                    /* Width domains named explicitly.
                     *
                     * GCC 13.2 does not carry vtype across the mixed widths
                     * here: it leaves the e8m2 setting from the load in force
                     * and issues vsext/vfwcvt/vfcvt under SEW=8, which are
                     * illegal (a vf2 extend would imply a 4-bit source; there
                     * is no 8-bit float). The kernel SIGILLs on its first
                     * dispatch. scripts/check_rvv_vtype.py gates on this.
                     *
                     * Element COUNT is identical across e8m2 / e16m4 / e32m8
                     * (EMUL scales with SEW), so no arithmetic changes. */
                const size_t n_elem = (size_t)(total_elems - hw);
                /* Element count in its own variable, handed to every
                 * width. Chaining `vsetvl_e16m4(vl)` on a previous
                 * vsetvl's result is miscompiled by this GCC: it passes an
                 * ADDRESS register as the AVL operand, vl saturates to
                 * VLMAX, and the vl-preserving forms carry that to the
                 * store. See rvv_cat2_c1_s8_direct.c for the disassembly
                 * and the guard-page proof. Only bites when the count is
                 * not a whole multiple of VLMAX, i.e. on a partial tail. */
                vl = __riscv_vsetvl_e8m2(n_elem);
                size_t vl8 = vl;
                size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
                size_t vl32 = __riscv_vsetvl_e32m8(n_elem);
                (void)__riscv_vsetvl_e8m2(n_elem);
                vint8m2_t v8 = __riscv_vle8_v_i8m2(in_base + hw, vl8);
                /* sign-extend i8 -> i32 (4x widen) */
                vint32m8_t v32 = __riscv_vsext_vf4_i32m8(v8, vl32);
                /* convert to float */
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl32);
                /* multiply by ratio */
                vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl32);
                /* round to nearest int32 */
                vint32m8_t vr = __riscv_vfcvt_x_f_v_i32m8(vf, vl32);
                /* clamp */
                vr = __riscv_vmax_vx_i32m8(vr, activation_min, vl32);
                vr = __riscv_vmin_vx_i32m8(vr, activation_max, vl32);
                /* narrow i32 -> i16 -> i8 */
                vint16m4_t vr16 = __riscv_vncvt_x_x_w_i16m4(vr, vl16);
                vint8m2_t vr8 = __riscv_vncvt_x_x_w_i8m2(vr16, vl8);
                __riscv_vse8_v_i8m2(out_base + hw, vr8, vl8);
            }

            out_c += ci;
        }
    }
}
