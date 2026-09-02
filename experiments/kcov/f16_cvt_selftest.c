#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* --- candidate inline conversions --- */
static inline float mb_h2f(uint16_t h)
{
    union { uint32_t u; float f; } o, magic;
    magic.u = 113u << 23;
    const uint32_t shifted_exp = 0x7c00u << 13;
    o.u = (uint32_t)(h & 0x7fffu) << 13;
    uint32_t exp = shifted_exp & o.u;
    o.u += (uint32_t)(127 - 15) << 23;
    if (exp == shifted_exp) {
        o.u += (uint32_t)(128 - 16) << 23;
    } else if (exp == 0) {
        o.u += 1u << 23;
        o.f -= magic.f;
    }
    o.u |= (uint32_t)(h & 0x8000u) << 16;
    return o.f;
}

static inline uint16_t mb_f2h(float ff)
{
    union { uint32_t u; float f; } f, f32infty, f16max, denorm_magic;
    f.f = ff;
    f32infty.u = 255u << 23;
    f16max.u   = (uint32_t)(127 + 16) << 23;
    denorm_magic.u = (uint32_t)((127 - 15) + (23 - 10) + 1) << 23;
    uint32_t sign = f.u & 0x80000000u;
    uint16_t o;
    f.u ^= sign;
    if (f.u >= f16max.u) {
        o = (f.u > f32infty.u) ? 0x7e00u : 0x7c00u;
    } else {
        if (f.u < (113u << 23)) {
            f.f += denorm_magic.f;
            o = (uint16_t)(f.u - denorm_magic.u);
        } else {
            uint32_t mant_odd = (f.u >> 13) & 1u;
            f.u += ((uint32_t)(15 - 127) << 23) + 0xfffu;
            f.u += mant_odd;
            o = (uint16_t)(f.u >> 13);
        }
    }
    o |= (uint16_t)(sign >> 16);
    return o;
}

int main(void)
{
    /* 1. h2f exhaustive over all 65536 half bit patterns. */
    long bad = 0;
    for (uint32_t i = 0; i < 65536u; i++) {
        uint16_t h = (uint16_t)i;
        _Float16 hv; memcpy(&hv, &h, 2);
        float ref = (float)hv;             /* toolchain / libgcc conversion */
        float got = mb_h2f(h);
        uint32_t rb, gb; memcpy(&rb,&ref,4); memcpy(&gb,&got,4);
        int is_nan = ((h & 0x7c00u) == 0x7c00u) && (h & 0x3ffu);
        if (rb != gb && !is_nan) { if (bad<10) printf("h2f MISMATCH h=%04x ref=%08x got=%08x\n",h,rb,gb); bad++; }
    }
    printf("h2f: %ld mismatches out of 65536 (NaN patterns excluded)\n", bad);

    /* 2. f2h round-trip over every half: h -> float -> half must be identity. */
    long bad2 = 0;
    for (uint32_t i = 0; i < 65536u; i++) {
        uint16_t h = (uint16_t)i;
        if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu)) continue;  /* NaN */
        uint16_t back = mb_f2h(mb_h2f(h));
        /* -0.0 and +0.0 are distinct patterns and must both round-trip. */
        if (back != h) { if (bad2<10) printf("rt MISMATCH h=%04x back=%04x\n",h,back); bad2++; }
    }
    printf("h2f->f2h round trip: %ld mismatches\n", bad2);

    /* 3. f2h vs toolchain over a dense sweep of float bit patterns:
     *    every 512th of the whole float32 space + all halfway cases. */
    long bad3 = 0, n3 = 0;
    for (uint64_t u = 0; u < 0x100000000ull; u += 251) {
        uint32_t bits = (uint32_t)u;
        float x; memcpy(&x,&bits,4);
        if (x != x) continue;                      /* NaN */
        _Float16 r = (_Float16)x;
        uint16_t rb; memcpy(&rb,&r,2);
        uint16_t gb = mb_f2h(x);
        n3++;
        if (rb != gb) { if (bad3<10) printf("f2h MISMATCH x=%08x ref=%04x got=%04x\n",bits,rb,gb); bad3++; }
    }
    printf("f2h: %ld mismatches out of %ld float32 patterns\n", bad3, n3);

    /* 4. exact ties (RNE) around every representable half step. */
    long bad4 = 0, n4 = 0;
    for (uint32_t i = 0; i < 65535u; i++) {
        uint16_t h = (uint16_t)i;
        if ((h & 0x7c00u) == 0x7c00u) continue;
        float lo = mb_h2f(h), hi = mb_h2f((uint16_t)(h + 1));
        if (lo != lo || hi != hi) continue;
        float mid = lo + (hi - lo) * 0.5f;
        _Float16 r = (_Float16)mid; uint16_t rb; memcpy(&rb,&r,2);
        uint16_t gb = mb_f2h(mid);
        n4++;
        if (rb != gb) { if (bad4<10) printf("tie MISMATCH h=%04x mid=%.9g ref=%04x got=%04x\n",h,mid,rb,gb); bad4++; }
    }
    printf("f2h ties: %ld mismatches out of %ld midpoints\n", bad4, n4);
    return (bad|bad2|bad3|bad4) ? 1 : 0;
}
