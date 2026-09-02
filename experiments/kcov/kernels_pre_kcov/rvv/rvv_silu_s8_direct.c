/* source: curated */
/* algorithm: direct */
/* origin: RVV silu_s8 — register-resident LUT via vrgather.vv instead of a
 *   memory-indexed vluxei8 gather.
 *
 *   Measured on this machine (Saturn V256/D128, fq jobs 35/36): the
 *   __riscv_vluxei8_v_u8m8 gather costs 11.9 cyc/elem while the plain f32
 *   requantize chain costs 1.7 — a memory-indexed gather is ~7x an
 *   arithmetic op stream here, because every one of the 256 lanes issues an
 *   independent byte load. silu_s8 sits at 17.2 cyc/elem for that reason.
 *
 *   The table is exactly 256 bytes and int8 inputs address exactly 256
 *   entries. At VLEN=256 an e8m8 register group holds exactly 256 bytes
 *   (vlmax_e8m8 == 256), so the WHOLE LUT fits in one register group and
 *   vrgather.vv can permute out of it — a register-internal crossbar with
 *   no memory traffic at all.
 *
 *   Guarded: vrgather.vv returns 0 for any index >= vlmax, so if the part
 *   has VLEN < 256 (vlmax_e8m8 < 256) the table would not fit and we fall
 *   back to the original vluxei8 path. Output is bit-identical either way —
 *   same LUT contents, same expf/roundf expression as the stock kernel.
 *
 *   HARDWARE-MEASURED on yolov8_nano (the only model that dispatches this
 *   op; 57 instances, 868,000 elements total, fq job 118 on
 *   f2_dual_small_norose_tacit_q31_60mhz): silu_s8's OWN per-dispatch
 *   hardware cycle counter reads 4,796,189 cycles = 5.53 cyc/elem, vs. the
 *   stock vluxei8 kernel's repeatedly-measured 17.16 cyc/elem (ids
 *   406/425/429 in kernel_opt_log.jsonl) = ~3.1x. Per-op spike verify
 *   against golden at all 13 of yolov8_nano's real silu shapes (plus
 *   extra_shapes) is max_abs_err=0.
 *
 *   IMPORTANT CAVEAT, not about this kernel: yolov8_nano's shared
 *   conv2d_s8/rvv_vsmul_vnclip is CONFIRMED BROKEN on real hardware for
 *   this model (coordinator bisect: the oc_blocked algorithm gives err=0,
 *   the untiled vsmul_vnclip gives err=3, and the currently-shipped tiled
 *   vsmul_vnclip causes an outright Store/AMO access fault — see
 *   kernel_opt_log.jsonl ids 37/904 and the conv owner's own log). That is
 *   almost certainly why this same FPGA run's whole-model self-check read
 *   max_abs_err=1 instead of 0, and why two attempts to run the STOCK
 *   silu_s8 baseline for comparison (same conv) each hung on real FPGA
 *   hardware without reaching HTIF exit (id 902) rather than completing.
 *   ids 702/704 already showed reverting add_s8 or cat*_c1_s8 alone (not
 *   silu, not conv) also zeroes this class of error against a frozen
 *   buggy base, so per-instruction it is not attributable to this file.
 *   Per the coordinator's explicit instruction, NO yolov8_nano figure is
 *   being banked as a validated result until the conv owner lands a fix
 *   and this gets re-measured on a clean rebuild — that includes the
 *   3.1x/5.53-cyc/elem numbers above, which are the best currently
 *   available evidence but not yet a clean-conv confirmation. */

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    static int8_t lut[256];
    static float  c_si = 0.0f, c_so = 0.0f;
    static int    c_lo = 0, c_hi = 0, c_valid = 0;

    if (!c_valid || scale_in != c_si || scale_out != c_so
                 || activation_min != c_lo || activation_max != c_hi) {
        for (int v = 0; v < 256; v++) {
            int8_t iv = (int8_t)(uint8_t)v;
            float f = (float)iv * scale_in;
            float y = f / (1.0f + expf(-f));
            int32_t q = (int32_t)roundf(y / scale_out);
            if (q < activation_min) q = activation_min;
            if (q > activation_max) q = activation_max;
            lut[v] = (int8_t)q;
        }
        c_si = scale_in; c_so = scale_out;
        c_lo = activation_min; c_hi = activation_max; c_valid = 1;
    }

    size_t vlmax = __riscv_vsetvlmax_e8m8();
    int i = 0;
    size_t vl;

    if (vlmax >= 256) {
        /* Whole 256-byte table resident in one e8m8 register group. */
        size_t tvl = __riscv_vsetvl_e8m8(256);
        vuint8m8_t vlut = __riscv_vle8_v_u8m8((const uint8_t *)lut, tvl);
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
            vuint8m8_t vout = __riscv_vrgather_vv_u8m8(vlut, vidx, vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
        }
    } else {
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
            vuint8m8_t vout = __riscv_vluxei8_v_u8m8((const uint8_t *)lut, vidx, vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
        }
    }
}
