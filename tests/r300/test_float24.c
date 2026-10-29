#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
/* r300_float24 (r300_us.c) against the ldexpf form it replaced, all 2^24 inputs:
   cc -O2 -o test_float24 test_float24.c -lm && ./test_float24 */
static float old24(uint32_t v)
{
    uint32_t mant = v & 0xFFFF, exp = (v >> 16) & 0x7F, sign = (v >> 23) & 1;
    float f;
    if (!exp && !mant) return sign ? -0.0f : 0.0f;
    f = ldexpf(1.0f + mant / 65536.0f, (int)exp - 63);
    return sign ? -f : f;
}
static float new24(uint32_t v)
{
    uint32_t bits = (v & 0xFFFFFF) == 0 || (v & 0xFFFFFF) == 0x800000
        ? (v & 0x800000) << 8
        : ((v & 0x800000) << 8) | ((((v >> 16) & 0x7F) + 64) << 23) | ((v & 0xFFFF) << 7);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
int main(void)
{
    for (uint32_t v = 0; v < (1u << 24); v++) {
        float a = old24(v), b = new24(v);
        assert(memcmp(&a, &b, 4) == 0);
    }
    puts("float24: all 2^24 values identical");
    return 0;
}
