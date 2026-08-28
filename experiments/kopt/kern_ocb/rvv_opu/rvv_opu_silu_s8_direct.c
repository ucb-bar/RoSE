/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV silu_s8 for the Saturn OPU tile. Pre-this
 *   kernel rvv_opu's silu_s8 fell back to the scalar reference, costing
 *   ~38 ms total across yolov8_nano_64's ~30 silu calls. The Saturn
 *   OPU has identical RVV scalar/vector unit width to plain `rvv`, so
 *   we use the same LUT-gather strategy:
 *     - 256-entry LUT precomputed at kernel entry (256 expf + roundf =
 *       ~12 µs; negligible vs the 100K-element gather it amortizes).
 *     - vluxei8 with LMUL=8 issues one indirect-byte-load per
 *       vector-cycle; on Saturn this is the back-pressure of the
 *       memory subsystem on random gather (no cache miss because the
 *       LUT lives in L1 hot from the precompute pass).
 *   Bit-exact against the gemmini scalar reference because both
 *   compute (input * scale_in -> silu -> /scale_out -> round -> clip)
 *   in single-precision float with identical rounding.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
#include "saturn_opu.h"  /* unused but present for backend signature parity */

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    /* Precompute LUT for all 256 int8 values. */
    int8_t lut[256];
    for (int v = 0; v < 256; v++) {
        int8_t iv = (int8_t)(uint8_t)v;
        float f = (float)iv * scale_in;
        float y = f / (1.0f + expf(-f));
        int32_t q = (int32_t)roundf(y / scale_out);
        if (q < activation_min) q = activation_min;
        if (q > activation_max) q = activation_max;
        lut[v] = (int8_t)q;
    }

    /* Vectorized table lookup via vluxei8 gather. */
    int i = 0;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
        vuint8m8_t vout = __riscv_vluxei8_v_u8m8((const uint8_t *)lut, vidx, vl);
        __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
    }
}
