/* clang -O2 -o /tmp/vmh-neon-test vmh-neon-test.c && /tmp/vmh-neon-test */
#include <stdio.h>
#include <string.h>
#include "../../target/ppc/vmx-neon.h"

static int check(const int16_t *aa, const int16_t *bb, const int16_t *cc,
                 bool round, int alias)
{
    int16_t a[8], b[8], c[8], out[8], expected[8];
    bool sat = false;
    memcpy(a, aa, sizeof(a));
    memcpy(b, bb, sizeof(b));
    memcpy(c, cc, sizeof(c));
    for (int i = 0; i < 8; i++) {
        int32_t p = (int32_t)a[i] * b[i] + (round ? 0x4000 : 0);
        int32_t x = (p >> 15) + c[i];
        if (x > INT16_MAX) { x = INT16_MAX; sat = true; }
        if (x < INT16_MIN) { x = INT16_MIN; sat = true; }
        expected[i] = x;
    }
    int16_t *dst = alias == 1 ? a : alias == 2 ? b : alias == 3 ? c : out;
    bool actual_sat = vmhaddshs_neon(dst, a, b, c, round);
    if (actual_sat != sat || memcmp(dst, expected, sizeof(expected))) {
        fprintf(stderr, "mismatch: round=%d alias=%d sat=%d/%d\n",
                round, alias, actual_sat, sat);
        return 1;
    }
    return 0;
}

int main(void)
{
    static const int16_t edge[] = { INT16_MIN, -32767, -16385, -16384,
                                    -1, 0, 1, 16383, 16384, 32766, INT16_MAX };
    int16_t a[8], b[8], c[8];
    uint32_t seed = 1;
    unsigned long n = 0;
    for (unsigned i = 0; i < sizeof(edge) / sizeof(edge[0]); i++) {
        for (unsigned j = 0; j < sizeof(edge) / sizeof(edge[0]); j++) {
            for (unsigned k = 0; k < sizeof(edge) / sizeof(edge[0]); k++) {
                for (int lane = 0; lane < 8; lane++) {
                    a[lane] = edge[i]; b[lane] = edge[j]; c[lane] = edge[k];
                }
                for (int round = 0; round < 2; round++) {
                    for (int alias = 0; alias < 4; alias++) {
                        if (check(a, b, c, round, alias)) { return 1; }
                        n++;
                    }
                }
            }
        }
    }
    for (int it = 0; it < 100000; it++) {
        for (int lane = 0; lane < 8; lane++) {
            seed = seed * 1664525 + 1013904223; a[lane] = seed >> 16;
            seed = seed * 1664525 + 1013904223; b[lane] = seed >> 16;
            seed = seed * 1664525 + 1013904223; c[lane] = seed >> 16;
        }
        for (int round = 0; round < 2; round++) {
            if (check(a, b, c, round, it % 4)) { return 1; }
            n++;
        }
    }
    printf("%lu vectors passed\n", n);
    return 0;
}
