/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 exact GELU = 0.5*x*(1+erf(x/sqrt2)).
   erf via Abramowitz&Stegun 7.1.26 (|err|<1.5e-7) with sign folding;
   the required exp(-t^2) uses a Cephes-style minimax expf. */

static inline vfloat32m8_t rvv_exp_ps_gelu(vfloat32m8_t x, size_t vl) {
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

static inline vfloat32m8_t rvv_erf_ps_gelu(vfloat32m8_t x, size_t vl) {
    vbool4_t neg = __riscv_vmflt_vf_f32m8_b4(x, 0.0f, vl);
    vfloat32m8_t ax = __riscv_vfabs_v_f32m8(x, vl);
    /* t = 1 / (1 + 0.3275911*ax) */
    vfloat32m8_t d = __riscv_vfmul_vf_f32m8(ax, 0.3275911f, vl);
    d = __riscv_vfadd_vf_f32m8(d, 1.0f, vl);
    vfloat32m8_t t = __riscv_vfrdiv_vf_f32m8(d, 1.0f, vl);
    /* poly = ((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t */
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(1.061405429f, vl);
    y = __riscv_vfmul_vv_f32m8(y, t, vl); y = __riscv_vfadd_vf_f32m8(y, -1.453152027f, vl);
    y = __riscv_vfmul_vv_f32m8(y, t, vl); y = __riscv_vfadd_vf_f32m8(y, 1.421413741f, vl);
    y = __riscv_vfmul_vv_f32m8(y, t, vl); y = __riscv_vfadd_vf_f32m8(y, -0.284496736f, vl);
    y = __riscv_vfmul_vv_f32m8(y, t, vl); y = __riscv_vfadd_vf_f32m8(y, 0.254829592f, vl);
    y = __riscv_vfmul_vv_f32m8(y, t, vl);
    /* erf = 1 - poly*exp(-ax*ax) */
    vfloat32m8_t axx = __riscv_vfmul_vv_f32m8(ax, ax, vl);
    vfloat32m8_t e = rvv_exp_ps_gelu(__riscv_vfneg_v_f32m8(axx, vl), vl);
    vfloat32m8_t ye = __riscv_vfmul_vv_f32m8(y, e, vl);
    vfloat32m8_t erf = __riscv_vfrsub_vf_f32m8(ye, 1.0f, vl); /* 1 - ye */
    vfloat32m8_t nerf = __riscv_vfneg_v_f32m8(erf, vl);
    return __riscv_vmerge_vvm_f32m8(erf, nerf, neg, vl);
}

void kernel_gelu(const float *input, float *output, int n) {
    const float inv_sqrt2 = 0.7071067811865475f;
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t x = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m8_t a = __riscv_vfmul_vf_f32m8(x, inv_sqrt2, vl);
        vfloat32m8_t erf = rvv_erf_ps_gelu(a, vl);
        erf = __riscv_vfadd_vf_f32m8(erf, 1.0f, vl);
        vfloat32m8_t out = __riscv_vfmul_vv_f32m8(x, erf, vl);
        out = __riscv_vfmul_vf_f32m8(out, 0.5f, vl);
        __riscv_vse32_v_f32m8(output + i, out, vl);
    }
}
