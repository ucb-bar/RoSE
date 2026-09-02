/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV depthwise_conv2d_s8.
 *
 * The scalar reference is a six-deep NCHW loop nest that pays an int64
 * requantize, two bounds tests and four index multiplies per OUTPUT ELEMENT
 * for only KH*KW MACs -- measured at 324.6 cycles per output element (36
 * cyc/MAC) on ViNT's EfficientNet-B0 depthwise stages, which made it the
 * single largest line item in the whole model.
 *
 * Depthwise has no reduction over input channels, so the only dimension with
 * both length and unit stride is OW: for a fixed (n, c, oh, kh, kw) the taps
 * input[.., ih, ow*SW - PW + kw] are contiguous when SW == 1. So vectorize
 * over ow and accumulate KH*KW widening MACs into one i32 vector:
 *
 *   i8 row  --vwadd.vx(input_offset)-->  i16  --vwmacc.vx(w+filter_offset)--> i32
 *
 * Two structural choices, both deliberate:
 *
 * 1. The requantize tail is SCALAR, draining the accumulator vector through a
 *    static buffer -- the same shape as the shipped rvv_f16_conv2d_s8_pc_direct
 *    kernel. It is one requant per KH*KW MACs, so it is not the bottleneck, and
 *    doing it scalar makes this kernel BIT-EXACT against the reference. The
 *    vector alternative (vsmul + vnclip) is explicitly retired in this tree:
 *    kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c records it measuring err=3 on
 *    yolov8n on FPGA where the scalar-drain kernel measured err=0.
 *
 * 2. Border columns are peeled to the scalar path rather than masked. The
 *    interior [PW, IW+PW-KW+1) is the ow range where every one of the KW taps
 *    is in bounds, so the vector body needs no per-tap predication; with the
 *    EfficientNet pad of 1 or 2 that leaves at most KW-1 columns per row on the
 *    slow path.
 *
 * SW != 1 keeps the scalar path (ViNT: 12 of 16 depthwise ops and 80% of the
 * depthwise MACs are SW==1). A strided vlse8 would cover it in one load but
 * this tree has a recorded Saturn vlse8 corruption on yolov8, so it is not
 * worth the risk for the remaining 20%. */

#define MB_DWS8_TILE 256
/* 64-byte aligned so introducing this TU cannot shift the alignment of the
 * static buffers other kernels in the same translation unit declare. */
static int32_t mb_dws8_acc[MB_DWS8_TILE] __attribute__((aligned(64)));

static inline int32_t mb_dws8_requant(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return (scaled + round) >> shift;
    }
    return scaled << (-shift);
}

void kernel_depthwise_conv2d_s8(const int8_t *input, const int8_t *weight,
                                const int32_t *bias, int8_t *output,
                                int N, int C, int IH, int IW,
                                int KH, int KW, int SH, int SW, int PH, int PW,
                                int input_offset, int filter_offset,
                                int output_offset,
                                int output_multiplier, int output_shift,
                                int activation_min, int activation_max) {
    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;

    /* ow range over which every tap kw in [0,KW) lands inside the row. */
    int ow_v_lo = 0, ow_v_hi = 0;
    if (SW == 1) {
        ow_v_lo = PW;                       /* kw=0     -> iw = ow-PW   >= 0 */
        ow_v_hi = IW + PW - (KW - 1);       /* kw=KW-1  -> iw <  IW        */
        if (ow_v_lo < 0) ow_v_lo = 0;
        if (ow_v_hi > OW) ow_v_hi = OW;
        if (ow_v_hi < ow_v_lo) ow_v_hi = ow_v_lo;
    }

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int32_t bias_c = bias ? bias[c] : 0;
            const int8_t *in_ch = input + (size_t)(n * C + c) * IH * IW;
            const int8_t *w_ch  = weight + (size_t)c * KH * KW;
            int8_t *out_ch = output + (size_t)(n * C + c) * OH * OW;

            for (int oh = 0; oh < OH; oh++) {
                int8_t *out_row = out_ch + (size_t)oh * OW;

                /* ---- scalar edges (and the whole row when SW != 1) ---- */
                int s_lo_end = (SW == 1) ? ow_v_lo : OW;
                int s_hi_beg = (SW == 1) ? ow_v_hi : OW;
                for (int ow = 0; ow < OW; ow++) {
                    if (ow >= s_lo_end && ow < s_hi_beg) continue;
                    int32_t acc = bias_c;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;
                            int32_t iv = (int32_t)in_ch[(size_t)ih * IW + iw]
                                       + input_offset;
                            int32_t wv = (int32_t)w_ch[kh * KW + kw]
                                       + filter_offset;
                            acc += iv * wv;
                        }
                    }
                    int32_t v = mb_dws8_requant(acc, output_multiplier,
                                                output_shift) + output_offset;
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    out_row[ow] = (int8_t)v;
                }
                if (SW != 1) continue;

                /* ---- vector interior ---- */
                for (int ow0 = ow_v_lo; ow0 < ow_v_hi; ) {
                    size_t want = (size_t)(ow_v_hi - ow0);
                    if (want > MB_DWS8_TILE) want = MB_DWS8_TILE;
                    size_t vl = __riscv_vsetvl_e8m1(want);

                    vint32m4_t acc = __riscv_vmv_v_x_i32m4(bias_c, vl);
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        const int8_t *in_row = in_ch + (size_t)ih * IW;
                        for (int kw = 0; kw < KW; kw++) {
                            int16_t wv = (int16_t)((int32_t)w_ch[kh * KW + kw]
                                                   + filter_offset);
                            vint8m1_t v8 = __riscv_vle8_v_i8m1(
                                in_row + (ow0 - PW + kw), vl);
                            vint16m2_t v16 = __riscv_vwadd_vx_i16m2(
                                v8, (int16_t)input_offset, vl);
                            acc = __riscv_vwmacc_vx_i32m4(acc, wv, v16, vl);
                        }
                    }
                    __riscv_vse32_v_i32m4(mb_dws8_acc, acc, vl);
                    for (size_t lane = 0; lane < vl; lane++) {
                        int32_t v = mb_dws8_requant(mb_dws8_acc[lane],
                                                    output_multiplier,
                                                    output_shift)
                                  + output_offset;
                        if (v < activation_min) v = activation_min;
                        if (v > activation_max) v = activation_max;
                        out_row[ow0 + (int)lane] = (int8_t)v;
                    }
                    ow0 += (int)vl;
                }
            }
        }
    }
}
