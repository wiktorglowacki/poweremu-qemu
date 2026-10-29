/* AArch64 lane-wise AltiVec integer helpers. */
#ifndef PPC_VMX_NEON_H
#define PPC_VMX_NEON_H

#if defined(__aarch64__)
#include <arm_neon.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static inline bool vmx_neon_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PPC_NATIVE_VINT");
        on = !(e && e[0] == '0');
    }
    return on;
}

/* The one product 32768 needs an extra one after NEON saturates it. */
static inline bool vmhaddshs_neon(int16_t *r, const int16_t *ap,
                                  const int16_t *bp, const int16_t *cp,
                                  bool round)
{
    int16x8_t a = vld1q_s16(ap), b = vld1q_s16(bp), c = vld1q_s16(cp);
    int16x8_t min = vdupq_n_s16(INT16_MIN), max = vdupq_n_s16(INT16_MAX);
    uint16x8_t special = vandq_u16(vceqq_s16(a, min), vceqq_s16(b, min));
    int16x8_t p = round ? vqrdmulhq_s16(a, b) : vqdmulhq_s16(a, b);
    int16x8_t sum = vqaddq_s16(p, c);
    uint16x8_t sat = vmvnq_u16(vceqq_s16(sum, vaddq_s16(p, c)));
    sat = vorrq_u16(sat, vandq_u16(special, vceqq_s16(sum, max)));
    sum = vqaddq_s16(sum, vreinterpretq_s16_u16(vandq_u16(special, vdupq_n_u16(1))));
    vst1q_s16(r, sum);
    return vmaxvq_u16(sat) != 0;
}
#endif
#endif
