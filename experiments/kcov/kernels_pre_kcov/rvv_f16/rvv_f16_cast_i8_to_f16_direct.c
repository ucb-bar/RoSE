/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV int8->fp16 dequantize cast. The reference scalar
 *   loop (`out[i] = (_Float16)((float)in[i] * scale)`) is a per-element
 *   scalar int->float convert + float multiply + float->half narrow -- ~18
 *   cyc/element measured on this core (fused_full: n=1536 costs 28,382 cyc).
 *   This kernel widens int8 -> int16 (vsext, pure-integer, always
 *   available), converts int16 -> _Float16 (vfcvt.f.x.v -- EXACT for the
 *   whole int8 range [-128,127], fp16 has an 11-bit mantissa), then
 *   multiplies by a fp16-rounded copy of `scale` (vfmul.vf). The whole
 *   vector hot loop stays SEW=16 with NO fp32 vector datapath, matching the
 *   fp16-tail design constraint used by linear_f16/lstm_f16 (curated notes:
 *   the FPGA target has no LUT area for a full fp32 vector ALU).
 *   Divergence from the reference: the int8->fp16 widen is exact, so the
 *   only extra error vs the reference is from rounding `scale` to fp16 ONCE
 *   up front instead of multiplying in fp32 and narrowing at the end --
 *   ~2^-11 relative, well inside the fp16 tail's existing numeric_drift
 *   envelope (same class as the reduction-order drift already declared by
 *   linear_f16/lstm_f16 "narrow"/"direct"). */

#include <stddef.h>
#include <riscv_vector.h>

void kernel_cast_i8_to_f16(const int8_t *in, _Float16 *out,
                           int n, float scale) {
    const _Float16 hscale = (_Float16)scale;
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m2((size_t)(n - i));
        vint8m2_t vi8 = __riscv_vle8_v_i8m2(in + i, vl);
        vint16m4_t vi16 = __riscv_vsext_vf2_i16m4(vi8, vl);
        vfloat16m4_t vf = __riscv_vfcvt_f_x_v_f16m4(vi16, vl);
        vf = __riscv_vfmul_vf_f16m4(vf, hscale, vl);
        __riscv_vse16_v_f16m4(out + i, vf, vl);
        i += (int)vl;
    }
}
