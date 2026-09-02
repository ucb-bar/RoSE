/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV cat2_c1_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    float ratio0 = scale0 / scale_out;
    float ratio1 = scale1 / scale_out;

    const int8_t *ins[2] = { in0, in1 };
    int cs[2] = { c0, c1 };
    float ratios[2] = { ratio0, ratio1 };
    int8_t amin = (int8_t)activation_min;
    int8_t amax = (int8_t)activation_max;

    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 2; i++) {
            float ratio = ratios[i];
            int ci = cs[i];
            const int8_t *in = ins[i];
            for (int c = 0; c < ci; c++) {
                const int8_t *src = in + ((n * ci) + c) * stride;
                int8_t *dst = output + ((n * (c0 + c1) + out_c + c) * stride);
                int hw = 0;
                size_t vl;
                for (; hw < stride; hw += (int)vl) {
                    /* Each width domain is entered EXPLICITLY.
                     *
                     * GCC 13.2 does not carry vtype across the mixed widths in
                     * this loop: it left `vsetvli e8,m2` standing and emitted
                     * `vsext.vf2 v8, v24` under it, which is illegal at SEW=8.
                     * The kernel SIGILLs on its first dispatch. Same defect,
                     * same fix, as rvv_batchnorm2d_s8_direct.c -- the
                     * intrinsics are used correctly and the compiler owes the
                     * vtype change, so name each transition instead.
                     *
                     * Element COUNT is identical throughout (EMUL scales with
                     * SEW: e8m2, e16m4 and e32m8 all hold the same number of
                     * elements), so this changes no arithmetic. */
                    /* The element count for THIS iteration, in its own
                     * variable, passed to every width's vsetvl.
                     *
                     * Chaining them (`vsetvl_e16m4(vl)` on the previous
                     * vsetvl's result) is what GCC miscompiles here. It
                     * emitted, verbatim:
                     *
                     *     subw    a3,a1,a4          ; a3 = stride - hw = 6
                     *     add     a5,a6,a4          ; a5 = src + hw  (POINTER)
                     *     vsetvli a3,a3,e8,m2       ; vl = 6, correct
                     *     vle8.v  v2,(a5)
                     *     vsetvli zero,a5,e16,m4    ; AVL = a5 -- THE POINTER
                     *
                     * A pointer as AVL is astronomically larger than VLMAX, so
                     * vl saturates to VLMAX = 64, and the vl-preserving
                     * `vsetvli zero,zero` forms that follow carry 64 all the
                     * way to the store. The tail then writes 64 bytes where it
                     * owes 6.
                     *
                     * It needs a small stride to bite: yolov8's cat_15 is
                     * stride = H*W = 2*3 = 6 over 66 channels, so the last
                     * store runs 58 bytes past the output buffer and lands in
                     * whatever .bss follows. Proven with a PROT_NONE guard
                     * page end-aligned to the output: SEGV_ACCERR at the guard
                     * with epc on this store.
                     *
                     * Giving each vsetvl the same plain integer removes the
                     * chain GCC mis-allocated. */
                    const size_t n_elem = (size_t)(stride - hw);
                    vl = __riscv_vsetvl_e8m2(n_elem);
                    /* Load int8 */
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src + hw, vl);
                    /* Sign-extend i8 -> i16 */
                    size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl16);
                    /* Sign-extend i16 -> i32 */
                    size_t vl32 = __riscv_vsetvl_e32m8(n_elem);
                    vint32m8_t v32 = __riscv_vsext_vf2_i32m8(v16, vl32);
                    /* Convert int32 -> float32 */
                    vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl32);
                    /* Multiply by ratio */
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl32);
                    /* Round to nearest int32 (round-to-nearest-even via vfcvt) */
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl32);
                    /* Narrow i32 -> i16 with saturation */
                    vl16 = __riscv_vsetvl_e16m4(n_elem);
                    vint16m4_t vi16 = __riscv_vnclip_wx_i16m4(vi, 0, __RISCV_VXRM_RDN, vl16);
                    /* Narrow i16 -> i8 with saturation */
                    size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
                    vint8m2_t vi8 = __riscv_vnclip_wx_i8m2(vi16, 0, __RISCV_VXRM_RDN, vl8);
                    /* Clamp to activation range */
                    vi8 = __riscv_vmax_vx_i8m2(vi8, amin, vl8);
                    vi8 = __riscv_vmin_vx_i8m2(vi8, amax, vl8);
                    /* Store */
                    __riscv_vse8_v_i8m2(dst + hw, vi8, vl8);
                }
            }
            out_c += ci;
        }
    }
}