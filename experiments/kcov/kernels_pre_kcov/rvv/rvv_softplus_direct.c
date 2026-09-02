/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 softplus = log(1+exp(x)), computed in the
   overflow-safe form  m + log(1+exp(-|x|))  with m=max(x,0). Cephes-style
   minimax expf and logf helpers. */

static inline vfloat32m8_t rvv_exp_ps_softplus(vfloat32m8_t x, size_t vl) {
    x = __riscv_vfmin_vf_f32m8(x, 88.3762626647949f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -88.3762626647949f, vl);
    vfloat32m8_t fx = __riscv_vfmul_vf_f32m8(x, 1.44269504088896341f, vl);
    vint32m8_t n = __riscv_vfcvt_x_f_v_i32m8(fx, vl);
    fx = __riscv_vfcvt_f_x_v_f32m8(n, vl);
    vfloat32m8_t t = __riscv_vfmul_vf_f32m8(fx, 0.693359375f, vl);
    x = __riscv_vfsub_vv_f32m8(x, t, vl);
    t = __riscv_vfmul_vf_f32m8(fx, -2.12194440e-4f, vl);
    x = __riscv_vfsub_vv_f32m8(x, t, vl);
    vfloat32m8_t z = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(1.9875691500E-4f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 1.3981999507E-3f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 8.3334519073E-3f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 4.1665795894E-2f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 1.6666665459E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 5.0000001201E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, z, vl);
    y = __riscv_vfadd_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, 1.0f, vl);
    n = __riscv_vadd_vx_i32m8(n, 127, vl);
    n = __riscv_vsll_vx_i32m8(n, 23, vl);
    vfloat32m8_t pow2n = __riscv_vreinterpret_v_i32m8_f32m8(n);
    return __riscv_vfmul_vv_f32m8(y, pow2n, vl);
}

static inline vfloat32m8_t rvv_log_ps_softplus(vfloat32m8_t x, size_t vl) {
    x = __riscv_vfmax_vf_f32m8(x, 1.17549435e-38f, vl);
    vuint32m8_t xu = __riscv_vreinterpret_v_f32m8_u32m8(x);
    vuint32m8_t eu = __riscv_vsrl_vx_u32m8(xu, 23, vl);
    xu = __riscv_vand_vx_u32m8(xu, 0x007fffffu, vl);
    xu = __riscv_vor_vx_u32m8(xu, 0x3f000000u, vl);
    x = __riscv_vreinterpret_v_u32m8_f32m8(xu);
    vint32m8_t ei = __riscv_vreinterpret_v_u32m8_i32m8(eu);
    ei = __riscv_vsub_vx_i32m8(ei, 0x7f, vl);
    vfloat32m8_t e = __riscv_vfcvt_f_x_v_f32m8(ei, vl);
    e = __riscv_vfadd_vf_f32m8(e, 1.0f, vl);
    vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(x, 0.707106781186547524f, vl);
    vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t tmp = __riscv_vmerge_vvm_f32m8(zero, x, mask, vl);
    x = __riscv_vfsub_vf_f32m8(x, 1.0f, vl);
    vfloat32m8_t onem = __riscv_vfmerge_vfm_f32m8(zero, 1.0f, mask, vl);
    e = __riscv_vfsub_vv_f32m8(e, onem, vl);
    x = __riscv_vfadd_vv_f32m8(x, tmp, vl);
    vfloat32m8_t z = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(7.0376836292E-2f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.1514610310E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  1.1676998740E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.2420140846E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  1.4249322787E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.6668057665E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  2.0000714765E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -2.4999993993E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  3.3333331174E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfmul_vv_f32m8(y, z, vl);
    tmp = __riscv_vfmul_vf_f32m8(e, -2.12194440e-4f, vl);
    y = __riscv_vfadd_vv_f32m8(y, tmp, vl);
    tmp = __riscv_vfmul_vf_f32m8(z, 0.5f, vl);
    y = __riscv_vfsub_vv_f32m8(y, tmp, vl);
    x = __riscv_vfadd_vv_f32m8(x, y, vl);
    tmp = __riscv_vfmul_vf_f32m8(e, 0.693359375f, vl);
    x = __riscv_vfadd_vv_f32m8(x, tmp, vl);
    return x;
}

void kernel_softplus(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m8_t m = __riscv_vfmax_vf_f32m8(v, 0.0f, vl);
        vfloat32m8_t ax = __riscv_vfabs_v_f32m8(v, vl);
        vfloat32m8_t t = rvv_exp_ps_softplus(__riscv_vfneg_v_f32m8(ax, vl), vl);
        vfloat32m8_t onept = __riscv_vfadd_vf_f32m8(t, 1.0f, vl); /* 1 + exp(-|x|) */
        vfloat32m8_t s = rvv_log_ps_softplus(onept, vl);
        vfloat32m8_t out = __riscv_vfadd_vv_f32m8(m, s, vl);
        __riscv_vse32_v_f32m8(output + i, out, vl);
    }
}
