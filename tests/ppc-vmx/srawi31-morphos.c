/* PPC32/G4 regression for SRAWI31 carry, aliases, XER and CR fields.
 * Build: ppc-morphos-gcc -O2 -Wall -Wextra -noixemul srawi31-morphos.c -o srawi31
 * Run inside mac99/G4 MorphOS. PPC64/ISA300 is outside this native probe.
 */
#include <stdint.h>
#include <stdio.h>

#define PRE "mfcr %[before]\n\tmtcrf 128,%[crseed]\n\tmtxer %[seed]\n\t"
#define POST "\n\tmfxer %[xer]\n\tmfcr %[cr]"
#define DISTINCT(insn) __asm__ volatile(PRE insn " %[out],%[in],31" POST \
 : [out] "=&r"(out), [xer] "=&r"(xer), [cr] "=&r"(cr), [before] "=&r"(before) \
 : [in] "r"(input), [seed] "r"(seed), [crseed] "r"(crseed) : "xer", "cr0")
#define ALIASED(insn) __asm__ volatile(PRE insn " %[out],%[out],31" POST \
 : [out] "+&r"(out), [xer] "=&r"(xer), [cr] "=&r"(cr), [before] "=&r"(before) \
 : [seed] "r"(seed), [crseed] "r"(crseed) : "xer", "cr0")

static int check(uint32_t input, uint32_t seed, unsigned alias, unsigned record)
{
    uint32_t out = input, xer, cr, before, crseed = 0x50000000u;
    uint32_t expected, carry, nibble, expected_cr;
    if (alias) {
        if (record) { ALIASED("srawi."); } else { ALIASED("srawi"); }
    } else {
        if (record) { DISTINCT("srawi."); } else { DISTINCT("srawi"); }
    }
    expected = 0u - (input >> 31);
    carry = input > 0x80000000u;
    nibble = record ? ((input >> 31) ? 8u : 2u) | (seed >> 31) : 5u;
    expected_cr = (before & 0x0fffffffu) | (nibble << 28);
    if (out != expected || xer != ((seed & ~0x20000000u) | (carry << 29)) ||
        cr != expected_cr) {
        printf("SRAWI31_FAIL input=%08lx seed=%08lx alias=%u record=%u out=%08lx xer=%08lx cr=%08lx expected_cr=%08lx\n",
            (unsigned long)input, (unsigned long)seed, alias, record,
            (unsigned long)out, (unsigned long)xer, (unsigned long)cr,
            (unsigned long)expected_cr);
        return 1;
    }
    return 0;
}

int main(void)
{
    static const uint32_t directed[] = {0,1,2,0x7ffffffeu,0x7fffffffu,
        0x80000000u,0x80000001u,0x80000002u,0xfffffffeu,0xffffffffu};
    uint32_t random = 0x5a17c3e9u;
    unsigned count = 0;
    for (unsigned i = 0; i < 2010; i++) {
        uint32_t v;
        random = random * 1664525u + 1013904223u;
        v = i < 10 ? directed[i] : random;
        for (unsigned flags = 0; flags < 16; flags++) {
            uint32_t seed = (flags & 7u) << 29 | ((flags & 8u) ? 127u : 0u);
            for (unsigned alias = 0; alias < 2; alias++) {
                for (unsigned record = 0; record < 2; record++) {
                    if (check(v, seed, alias, record)) return 20;
                    count++;
                }
            }
        }
    }
    printf("SRAWI31_PASS checks=%u alias=both record=both xer=SO_OV_CA_BC cr=all\n", count);
    return 0;
}
