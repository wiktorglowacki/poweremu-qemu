/*
 * The expected values for tests/ppc-vmx/lfs-qtest.py.
 *
 * Reads 4096 single bit patterns and 4096 double bit patterns from
 * stdin (raw, host byte order) and writes what lfs and stfs must produce
 * for them: ppc_todouble() of each single, then ppc_tosingle() of each
 * double.  Those are the functions the helpers are defined by, so this
 * is the reference the guest run is compared against.
 *
 * gcc -O2 -o /tmp/lfs-conv-ref lfs-conv-ref.c
 */
#include <stdio.h>
#include <stdint.h>
#include "../../target/ppc/fp32-conv.h"

#define N 4096

int main(void)
{
    static uint32_t s[N], so[N];
    static uint64_t d[N], do_[N];
    size_t i;

    if (fread(s, sizeof(s[0]), N, stdin) != N ||
        fread(d, sizeof(d[0]), N, stdin) != N) {
        return 1;
    }
    for (i = 0; i < N; i++) {
        do_[i] = ppc_todouble(s[i]);
        so[i] = ppc_tosingle(d[i]);
    }
    fwrite(do_, sizeof(do_[0]), N, stdout);
    fwrite(so, sizeof(so[0]), N, stdout);
    return 0;
}
