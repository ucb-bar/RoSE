/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 natural log (Cephes single-precision logf):
   frexp via exponent bits, degree-8 minimax poly on the reduced mantissa.
   x<=0 -> NaN. Accurate to <1e-6 rel. */

static inline vfloat32m8_t rvv_log_ps_log(vfloat32m8_t x, size_t vl) {
    vbool4_t invalid = __riscv_vmfle_vf_f32m8_b4(x, 0.0f, vl);
    x = __riscv_vfmax_vf_f32m8(x, 1.17549435e-38f, vl); /* min normal */
    vuint32m8_t xu = __riscv_vreinterpret_v_f32m8_u32m8(x);
    vuint32m8_t eu = __riscv_vsrl_vx_u32m8(xu, 23, vl);
    xu = __riscv_vand_vx_u32m8(xu, 0x007fffffu, vl);
    xu = __riscv_vor_vx_u32m8(xu, 0x3f000000u, vl); /* mantissa in [0.5,1) */
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
    x = __riscv_vfmerge_vfm_f32m8(x, NAN, invalid, vl);
    return x;
}

void kernel_log(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        v = rvv_log_ps_log(v, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
