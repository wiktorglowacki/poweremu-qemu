/* Self-check of vmhraddshs_x86 against the scalar definition. gcc -O2 -o t vmhraddshs-test.c && ./t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../target/ppc/vmx-x86.h"
static int16_t sat16(int32_t t, int *sat) { if (t > 32767) { *sat = 1; return 32767; } if (t < -32768) { *sat = 1; return -32768; } return t; }
int main(void)
{
    static const int16_t edge[] = { -32768, -32767, -16385, -16384, -1, 0, 1, 16383, 16384, 32766, 32767, 0x4000, -0x4000 };
    srand(1); unsigned long n = 0, bad = 0;
    for (int it = 0; it < 400000; it++) {
        int16_t a[8], b[8], c[8], r[8], ref[8]; int sat = 0;
        for (int i = 0; i < 8; i++) {
            int pick = rand() % 4;
            a[i] = pick == 0 ? edge[rand() % 13] : (int16_t)rand();
            b[i] = pick == 1 ? edge[rand() % 13] : (int16_t)rand();
            c[i] = pick == 2 ? edge[rand() % 13] : (int16_t)rand();
            int32_t prod = a[i] * b[i] + 0x4000;
            ref[i] = sat16((int32_t)c[i] + (prod >> 15), &sat);
        }
        bool s = vmhraddshs_x86(r, a, b, c); n++;
        if (memcmp(r, ref, 16) || s != (bool)sat) {
            if (bad++ < 5) for (int i = 0; i < 8; i++) printf("a=%d b=%d c=%d got %d want %d (sat %d/%d)\n", a[i], b[i], c[i], r[i], ref[i], s, sat);
        }
    }
    printf("%lu vectors, %lu mismatches\n", n, bad);
    return bad != 0;
}
