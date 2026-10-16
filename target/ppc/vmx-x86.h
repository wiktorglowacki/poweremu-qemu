/*
 * x86 fast paths for AltiVec integer helpers that MorphOS's video code
 * leans on.  Lane order is irrelevant: every op is lane-wise and the
 * whole 16-byte register is loaded and stored.
 */
#ifndef PPC_VMX_X86_H
#define PPC_VMX_X86_H

#if defined(__x86_64__)
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

static inline bool vmx_x86_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PPC_NATIVE_VINT");
        on = !(e && e[0] == '0') && __builtin_cpu_supports("sse4.1");
    }
    return on;
}

/*
 * vmhraddshs: r = sat16(((a * b + 0x4000) >> 15) + c).  pmulhrsw is that
 * product except for -32768 * -32768, where it wraps to -32768 while the
 * PPC keeps 32768; those lanes are patched (32768 + c saturated).  Returns
 * whether any lane saturated (VSCR[SAT]).
 */
__attribute__((target("ssse3,sse4.1")))
static inline bool vmhraddshs_x86(void *rp, const void *ap, const void *bp,
                                  const void *cp)
{
    __m128i a = _mm_loadu_si128((const __m128i *)ap);
    __m128i b = _mm_loadu_si128((const __m128i *)bp);
    __m128i c = _mm_loadu_si128((const __m128i *)cp);
    __m128i min16 = _mm_set1_epi16((short)0x8000);
    __m128i m = _mm_mulhrs_epi16(a, b);
    __m128i e = _mm_and_si128(_mm_cmpeq_epi16(a, min16), _mm_cmpeq_epi16(b, min16));
    __m128i adds = _mm_adds_epi16(m, c), add = _mm_add_epi16(m, c);
    __m128i sat_lane = _mm_xor_si128(_mm_cmpeq_epi16(adds, add), _mm_set1_epi16(-1));
    /* patched lanes: c < 0 -> c ^ 0x8000 (= 32768 + c), else 32767 (saturated) */
    __m128i cneg = _mm_cmpgt_epi16(_mm_setzero_si128(), c);
    __m128i fixed = _mm_blendv_epi8(_mm_set1_epi16(0x7FFF), _mm_xor_si128(c, min16), cneg);
    __m128i r = _mm_blendv_epi8(adds, fixed, e);
    __m128i sat = _mm_or_si128(_mm_andnot_si128(e, sat_lane),
                               _mm_andnot_si128(cneg, e));
    _mm_storeu_si128((__m128i *)rp, r);
    return !_mm_testz_si128(sat, sat);
}
#endif /* __x86_64__ */
#endif
