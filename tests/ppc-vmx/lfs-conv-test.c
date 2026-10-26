/*
 * Self-check of the inline lfs/stfs conversion against the helper.
 *
 * The translator emits the arithmetic of gen_native_todouble() and
 * gen_native_tosingle() (target/ppc/translate/fp-impl.c.inc) when
 * ppc_todouble_inline()/ppc_tosingle_inline() below say they apply, and
 * calls helper_todouble()/helper_tosingle() -- which are ppc_todouble()
 * and ppc_tosingle() from target/ppc/fp32-conv.h -- otherwise.  This
 * compares the two over every single-precision bit pattern (the load
 * direction is exhaustive) and over a wide sweep of doubles (the store
 * direction: every single as a double, every exponent with edge
 * mantissas, and random patterns).
 *
 * gcc -O2 -o /tmp/lfs-conv-test lfs-conv-test.c && /tmp/lfs-conv-test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "../../target/ppc/fp32-conv.h"

/*
 * The inline path, one statement per TCG op, in the order the translator
 * emits them.  false is the branch to the helper.
 */
static bool ppc_todouble_inline(uint32_t arg, uint64_t *out)
{
    uint64_t x = arg;
    uint64_t a, t;

    a = x & 0x7fffffff;
    t = a - 0x00800000;
    if (t >= 0x7f000000) {
        *out = x << 32;
        return a == 0;
    }
    *out = a << 29;
    *out += (uint64_t)896 << 52;
    t = x & 0x80000000;
    t <<= 32;
    *out |= t;
    return true;
}

static bool ppc_tosingle_inline(uint64_t arg, uint32_t *out)
{
    uint64_t t, u;

    t = arg >> 52;
    t &= 0x7ff;
    if (t <= 896) {
        *out = (uint32_t)(arg >> 32) & 0x80000000;
        return t < 874;
    }
    u = arg >> 29;
    u &= 0x3fffffff;
    t = arg >> 62;
    t <<= 30;
    u |= t;
    *out = (uint32_t)u;
    return true;
}

static unsigned long n_inline, n_slow, bad;

static void check_double(uint32_t x)
{
    uint64_t got, want = ppc_todouble(x);

    if (ppc_todouble_inline(x, &got)) {
        n_inline++;
        if (got != want) {
            if (bad++ < 10) {
                printf("lfs  0x%08x: inline 0x%016llx helper 0x%016llx\n",
                       x, (unsigned long long)got, (unsigned long long)want);
            }
        }
    } else {
        n_slow++;
    }
}

static void check_single(uint64_t x)
{
    uint32_t got, want = ppc_tosingle(x);

    if (ppc_tosingle_inline(x, &got)) {
        n_inline++;
        if (got != want) {
            if (bad++ < 10) {
                printf("stfs 0x%016llx: inline 0x%08x helper 0x%08x\n",
                       (unsigned long long)x, got, want);
            }
        }
    } else {
        n_slow++;
    }
}

int main(int argc, char **argv)
{
    static const uint64_t edge_mant[] = {
        0, 1, 2, 3, 7, 8, 0x7ffffffffffffULL, 0x8000000000000ULL,
        0x8000000000001ULL, 0xfffffffffffffULL, 0xffffffffffffeULL,
        0x4000000000000ULL,
    };
    unsigned long long n_rand = argc > 1 ? strtoull(argv[1], NULL, 0) : 40000000;
    uint64_t rng = 0x9e3779b97f4a7c15ull;
    unsigned long long i;

    /* loads: every single-precision bit pattern */
    for (i = 0; i < (1ULL << 32); i++) {
        check_double((uint32_t)i);
    }
    printf("lfs:  2^32 patterns, %lu inline, %lu helper, %lu mismatches\n",
           n_inline, n_slow, bad);
    if (bad) {
        return 1;
    }

    n_inline = n_slow = 0;

    /* stores: every single as a double, plus the edge doubles */
    for (i = 0; i < (1ULL << 32); i++) {
        check_single(ppc_todouble((uint32_t)i));
    }
    for (int e = 0; e < 2048; e++) {
        for (size_t m = 0; m < sizeof(edge_mant) / sizeof(edge_mant[0]); m++) {
            uint64_t x = (uint64_t)e << 52 | edge_mant[m];

            check_single(x);
            check_single(x | (1ULL << 63));
        }
    }
    /* exponents around 896 (normal results) and 874 (denormals) */
    for (int e = 860; e < 940; e++) {
        for (uint64_t m = 0; m < (1ULL << 20); m += 0x9e37) {
            check_single((uint64_t)e << 52 | m);
        }
    }

    /* stores: random patterns */
    for (i = 0; i < n_rand; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        check_single(rng);
    }
    printf("stfs: 2^32 singles as doubles + edges + %llu random, "
           "%lu inline, %lu helper, %lu mismatches\n",
           n_rand, n_inline, n_slow, bad);
    return bad != 0;
}
