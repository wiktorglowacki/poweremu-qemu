#!/usr/bin/env python3
"""Check the production cursor cache without a guest or GPU.

An optional source filename lets the same assertions check an older revision.
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[2] / "hw/display/ppc_mac_gpu.c")
text = source.read_text()
match = re.search(r"^static void ppc_mac_gpu_radeon_cursor\([^;]+?\)\n\{", text, re.M)
assert match
start = match.start()
end = text.index("{", start) + 1
depth = 1
while depth:
    depth += (text[end] == "{") - (text[end] == "}")
    end += 1
function = text[start:end]

prefix = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
typedef struct {
    unsigned hot_x, hot_y;
    uint32_t data[64 * 64];
} QEMUCursor;
typedef struct {
    bool r300, r300_card_bytes, hwc_visible, r300_cursor_dirty;
    void *con, *vram;
    struct { uint32_t cur_offset, cur_horz_vert_posn, cur_horz_vert_off,
        crtc_gen_cntl, cur_clr0, cur_clr1; } regs;
    int hwc_x, hwc_y, r300_cursor_y0, r300_cursor_y1;
    uint32_t hwc_w, hwc_h, hwc_pix[64 * 64];
    uint8_t r300_cursor_source[64 * 64 * 4];
    uint32_t r300_cursor_source_len;
} PPCMacGPUState;
static uint8_t vram[65536];
static unsigned allocations, defines, syncs;
static bool pending_write, move_during_sync;
static QEMUCursor *cursor_alloc(unsigned w, unsigned h)
{ allocations++; return calloc(1, sizeof(QEMUCursor)); }
static void cursor_unref(QEMUCursor *c) { free(c); }
static void dpy_cursor_define(void *con, QEMUCursor *c) { defines++; }
static void dpy_mouse_set(void *con, int x, int y, bool visible) {}
static void dpy_refresh_soon(void *con) {}
static void cursor_set_mono(QEMUCursor *c, uint32_t fg, uint32_t bg,
                            const uint8_t *bits, int pitch, const uint8_t *mask)
{ abort(); }
static void *memory_region_get_ram_ptr(void *region) { return vram; }
static bool r300_to_vram(PPCMacGPUState *s, uint32_t *off, uint32_t len)
{ return *off + (uint64_t)len <= sizeof(vram); }
static uint32_t ldl_le_p(const uint8_t *p)
{ return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint32_t ldl_be_p(const uint8_t *p)
{ return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static void r200_vram_access(PPCMacGPUState *s, uint64_t lo, uint64_t hi,
                             bool write, uint32_t tag)
{
    syncs++;
    if (pending_write) { vram[lo + 2] = 77; pending_write = false; }
    if (move_during_sync) { s->regs.cur_offset = 32768; move_during_sync = false; }
}
'''

tests = r'''
int main(void)
{
    PPCMacGPUState s = { .r300 = true, .r300_card_bytes = true, .con = (void *)1,
        .regs = { .cur_offset = 128, .cur_horz_vert_posn = 2u << 16 | 4,
                  .crtc_gen_cntl = 3u << 20 | 1u << 16 } };
    vram[130] = 255; vram[131] = 255;
    ppc_mac_gpu_radeon_cursor(&s, true);
    assert(s.hwc_pix[0] == 0xffff0000u && allocations == 1);
    s.r300_cursor_dirty = false;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(allocations == 1 && !s.r300_cursor_dirty);

    vram[130] = 140;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(s.hwc_pix[0] == 0xff8c0000u && allocations == 2 && s.r300_cursor_dirty);
    s.r300_cursor_dirty = false;
    s.regs.cur_horz_vert_posn |= 1u << 31;
    vram[130] = 100;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(s.hwc_pix[0] == 0xff8c0000u && allocations == 2 && !s.r300_cursor_dirty);
    s.regs.cur_horz_vert_posn &= ~(1u << 31);
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(s.hwc_pix[0] == 0xff640000u && allocations == 3);

    pending_write = true;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(syncs && !pending_write && s.hwc_pix[0] == 0xff4d0000u);
    unsigned before = allocations;
    move_during_sync = true;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(allocations == before);
    vram[32770] = 33; vram[32771] = 255;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(s.hwc_pix[0] == 0xff210000u && allocations == before + 1);

    s.regs.cur_offset = 128;
    s.regs.crtc_gen_cntl = 1u << 16;
    s.regs.cur_clr0 = 0x112233;
    s.regs.cur_clr1 = 0xaabbcc;
    memset(vram + 128, 0xff, 1024);
    for (unsigned y = 0; y < 64; y++) memset(vram + 128 + y * 16 + 8, 0, 8);
    vram[128] = 0x3f; vram[136] = 0x60;
    ppc_mac_gpu_radeon_cursor(&s, true);
    assert(s.hwc_pix[0] == 0xff112233u && s.hwc_pix[1] == 0xffaabbccu);
    assert(s.hwc_pix[2] == 0x01000000u && s.hwc_pix[3] == 0);
    vram[136] = 0x20;
    ppc_mac_gpu_radeon_cursor(&s, false);
    assert(s.hwc_pix[1] == 0xff112233u);
    puts("PASS: in-place ARGB/mono updates, unchanged cache, lock, source synchronization and BQL revalidation");
}
'''

with tempfile.TemporaryDirectory(prefix="radeon-cursor-source-") as temporary:
    path = Path(temporary)
    c = path / "test.c"
    c.write_text(prefix + function + tests)
    binary = path / "test"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
                    "-Wno-unused-parameter", str(c), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
