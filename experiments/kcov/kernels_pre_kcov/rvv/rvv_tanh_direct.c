/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 tanh via tanh(x)=sign(x)*(e2-1)/(e2+1),
   e2=exp(2|x|). |x| saturated at 15 (tanh already ==1 in fp32).
   Cephes-style minimax expf, <1e-6 rel. */

static inline vfloat32m8_t rvv_exp_ps_tanh(vfloat32m8_t x, size_t vl) {
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

static inline vfloat32m8_t rvv_tanh_ps_tanh(vfloat32m8_t x, size_t vl) {
    vbool4_t neg = __riscv_vmflt_vf_f32m8_b4(x, 0.0f, vl);
    vfloat32m8_t ax = __riscv_vfabs_v_f32m8(x, vl);
    ax = __riscv_vfmin_vf_f32m8(ax, 15.0f, vl);
    vfloat32m8_t e = rvv_exp_ps_tanh(__riscv_vfmul_vf_f32m8(ax, 2.0f, vl), vl);
    vfloat32m8_t num = __riscv_vfsub_vf_f32m8(e, 1.0f, vl);
    vfloat32m8_t den = __riscv_vfadd_vf_f32m8(e, 1.0f, vl);
    vfloat32m8_t r = __riscv_vfdiv_vv_f32m8(num, den, vl);
    vfloat32m8_t nr = __riscv_vfneg_v_f32m8(r, vl);
    return __riscv_vmerge_vvm_f32m8(r, nr, neg, vl);
}

void kernel_tanh(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        v = rvv_tanh_ps_tanh(v, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
