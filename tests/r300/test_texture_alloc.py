#!/usr/bin/env python3
"""Fail real texture-converter allocations, then check a clean retry."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

r300 = Path(__file__).resolve().parents[2] / "hw/display/r300"
check = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned calls, fail_at, live;
static void *checked_malloc(size_t size)
{
    if (++calls == fail_at) return NULL;
    void *p = malloc(size);
    if (p) live++;
    return p;
}
static void checked_free(void *p)
{
    if (p) { assert(live); live--; }
    free(p);
}
#define malloc checked_malloc
#define free checked_free
#include "r300_draw.c"
#undef malloc
#undef free

static void retry(const R300TexDesc *td, const uint8_t *src,
                  const uint8_t *expected, uint32_t pitch, size_t bytes,
                  unsigned allocations)
{
    for (unsigned failure = 1; failure <= allocations; failure++) {
        uint32_t bpr = 0xdeadbeef;
        calls = 0; fail_at = failure;
        uint8_t *out = r300_tex_level_bytes(td, src, 0, td->width, td->height, &bpr);
        assert(!out && !bpr && calls == failure && !live);
        calls = 0; fail_at = 0;
        out = r300_tex_level_bytes(td, src, 0, td->width, td->height, &bpr);
        assert(out && bpr == pitch && calls == allocations && live == 1);
        assert(!memcmp(out, expected, bytes));
        checked_free(out);
        assert(!live);
    }
}

int main(void)
{
    const uint8_t rgb565[] = { 0, 0xf8, 0xe0, 7 };
    const uint8_t decoded[] = { 0, 0, 255, 255, 0, 255, 0, 255 };
    R300TexDesc td = { .kind = R300_TEXK_CONVERT16, .format = 6,
        .card_bytes = true, .width = 2, .height = 1, .levels = 1,
        .lvl_pitch = { 4 }, .lvl_rows = { 1 }, .size_bytes = 4 };
    retry(&td, rgb565, decoded, 8, sizeof(decoded), 1);

    uint8_t src[32];
    for (unsigned i = 0; i < sizeof(src); i++) src[i] = i;
    const uint8_t dxt[] = { 3, 2, 1, 0, 7, 6, 5, 4 };
    td = (R300TexDesc){ .kind = R300_TEXK_DXT1, .card_bytes = true, .endian = 2,
        .width = 4, .height = 4, .levels = 1, .lvl_pitch = { 32 },
        .lvl_rows = { 1 }, .size_bytes = 32 };
    retry(&td, src, dxt, 8, sizeof(dxt), 1);

    const uint8_t rgba[] = { 0, 1, 2, 3, 4, 5, 6, 7,
                            12, 13, 14, 15, 16, 17, 18, 19 };
    td = (R300TexDesc){ .kind = R300_TEXK_RGBA8, .card_bytes = true,
        .width = 2, .height = 2, .levels = 1, .lvl_pitch = { 12 },
        .lvl_rows = { 2 }, .size_bytes = 24 };
    retry(&td, src, rgba, 8, sizeof(rgba), 1);

    /* One full microtile: fail the untile buffer and the converter buffer. */
    td.width = 4; td.tiling = 2; td.lvl_pitch[0] = 16; td.size_bytes = 32;
    retry(&td, src, src, 16, sizeof(src), 2);
    puts("PASS: texture allocation failures, zero pitch, cleanup and byte-exact retries");
}
'''

with tempfile.TemporaryDirectory(prefix="r300-texture-alloc-") as tmp:
    path = Path(tmp)
    (path / "check.c").write_text(check)
    subprocess.run([*shlex.split(os.environ.get("CC", "cc")), "-std=c11", "-D_GNU_SOURCE",
                    "-O1", "-g", "-Wall", "-Wextra", "-Wno-unused-parameter",
                    "-Wno-unused-function", "-Wno-sign-compare",
                    *shlex.split(os.environ.get("CFLAGS", "")),
                    "-I" + str(r300), str(path / "check.c"),
                    *[str(r300 / n) for n in ("r300_state.c", "r300_pvs.c", "r300_us.c")],
                    "-lm", "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)
