/*
 * PowerPC single <-> double conversions behind lfs and stfs.
 *
 * These are the "non-arithmetic" conversions of the Power ISA: pure bit
 * manipulation, no rounding, no FPSCR and no softfloat flags.  They live
 * here so that the helper (fpu_helper.c) and the inline translator path
 * (translate/fp-impl.c.inc) are defined against one algorithm, and so
 * that tests/ppc-vmx/lfs-conv-test.c can compare the inline path with the
 * helper over every input using a plain host compiler.
 *
 * Self-contained on purpose: no qemu headers.
 */
#ifndef PPC_FP32_CONV_H
#define PPC_FP32_CONV_H

#include <stdbool.h>
#include <stdint.h>

#ifndef likely
#define likely(x)   (x)
#define unlikely(x) (x)
#define PPC_FP32_CONV_UNDEF_HINTS
#endif

/*
 * This is the non-arithmatic conversion that happens e.g. on loads.
 * In the Power ISA pseudocode, this is called DOUBLE.
 */
static inline uint64_t ppc_todouble(uint32_t arg)
{
    uint32_t abs_arg = arg & 0x7fffffff;
    uint64_t ret;

    if (likely(abs_arg >= 0x00800000)) {
        if (unlikely(((arg >> 23) & 0xff) == 0xff)) {
            /* Inf or NAN.  */
            ret  = (uint64_t)(arg >> 31) << 63;
            ret |= (uint64_t)0x7ff << 52;
            ret |= (uint64_t)(arg & 0x7fffff) << 29;
        } else {
            /* Normalized operand.  */
            ret  = (uint64_t)((arg >> 30) & 3) << 62;
            ret |= ((uint64_t)(((arg >> 30) & 1) ^ 1) * 7) << 59;
            ret |= (uint64_t)(arg & 0x3fffffff) << 29;
        }
    } else {
        /* Zero or Denormalized operand.  */
        ret = (uint64_t)(arg >> 31) << 63;
        if (unlikely(abs_arg != 0)) {
            /*
             * Denormalized operand.
             * Shift fraction so that the msb is in the implicit bit position.
             * Thus, shift is in the range [1:23].
             */
            int shift = __builtin_clz(abs_arg) - 8;
            /*
             * The first 3 terms compute the float64 exponent.  We then bias
             * this result by -1 so that we can swallow the implicit bit below.
             */
            int exp = -126 - shift + 1023 - 1;

            ret |= (uint64_t)exp << 52;
            ret += (uint64_t)abs_arg << (52 - 23 + shift);
        }
    }
    return ret;
}

/*
 * This is the non-arithmatic conversion that happens e.g. on stores.
 * In the Power ISA pseudocode, this is called SINGLE.
 */
static inline uint32_t ppc_tosingle(uint64_t arg)
{
    int exp = (arg >> 52) & 0x7ff;
    uint32_t ret;

    if (likely(exp > 896)) {
        /* No denormalization required (includes Inf, NaN).  */
        ret  = ((arg >> 62) & 3) << 30;
        ret |= (arg >> 29) & 0x3fffffff;
    } else {
        /*
         * Zero or Denormal result.  If the exponent is in bounds for
         * a single-precision denormal result, extract the proper
         * bits.  If the input is not zero, and the exponent is out of
         * bounds, then the result is undefined; this underflows to
         * zero.
         */
        ret = (uint32_t)(arg >> 63) << 31;
        if (unlikely(exp >= 874)) {
            /* Denormal result.  */
            ret |= ((1ULL << 52) | (arg & 0xfffffffffffffULL)) >>
                   (896 + 30 - exp);
        }
    }
    return ret;
}

#ifdef PPC_FP32_CONV_UNDEF_HINTS
#undef likely
#undef unlikely
#undef PPC_FP32_CONV_UNDEF_HINTS
#endif

#endif /* PPC_FP32_CONV_H */
