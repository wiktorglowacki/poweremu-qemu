#!/usr/bin/env python3
"""Compile the actual generation sampling, source selection and final guard."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / "hw/display/ppc_mac_gpu.c").read_text()
start = source.index("    uint64_t vram_generation = 0;")
sample = source[start:source.index("    if (s->disp.stride && s->disp.height) {", start)]
start = source.index("    m->offset = s->regs.crtc_offset")
offset = source[start:source.index("    /*\n     * The refresh the driver programmed:", start)]
start = source.index("    use_generation &= generation_lo == s->disp.offset")
guard = source[start:source.index(";", start) + 1]
bias = re.search(r"^#define R200_CRTC_OFFSET_BIAS\s+\S+", (root / "include/hw/display/ppc_mac_gpu.h").read_text(), re.M).group()
fixture = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
typedef struct { uint32_t offset, stride, height; } PPCMacGPUDisplayMode;
typedef struct {
    uint64_t (*vram_generation)(void *, uint64_t, uint64_t);
} Renderer;
typedef struct {
    bool r300, r300_card_bytes, rv250;
    uint64_t vram_size;
    struct { uint32_t crtc_offset; } regs;
    PPCMacGPUDisplayMode disp;
    Renderer *renderer;
    void *renderer_opaque;
} PPCMacGPUState;
static bool ppc_gpu_dirty_scanout = true;
static uint64_t sampled_lo, sampled_hi;
static uint32_t yield_offset;
static uint64_t generation(void *opaque, uint64_t lo, uint64_t hi)
{
    PPCMacGPUState *s = opaque;
    assert(lo <= hi && hi <= s->vram_size);
    sampled_lo = lo; sampled_hi = hi;
    if (yield_offset) s->regs.crtc_offset = yield_offset;
    return 17;
}
'''
wrapper = """
static bool refresh(PPCMacGPUState *s, uint32_t next_stride, uint32_t next_height)
{
""" + sample + """
    /* Mode tracking runs after harvesting, which may have yielded the BQL. */
    s->disp.stride = next_stride;
    s->disp.height = next_height;
    PPCMacGPUDisplayMode *m = &s->disp;
""" + offset + """
    uint32_t stride = s->disp.stride, height = s->disp.height;
""" + guard + """
    assert(vram_generation == 17);
    return use_generation;
}
"""
main = r'''
int main(void)
{
    Renderer renderer = { generation };
    PPCMacGPUState s = { .r300 = true, .r300_card_bytes = true,
        .vram_size = 256u << 20, .regs = { 0x960000 },
        .disp = { 0x10000, 8192, 1152 }, .renderer = &renderer };
    s.renderer_opaque = &s;
    assert(refresh(&s, 8192, 1152));
    assert(sampled_lo == 0x960000 && sampled_hi == 0x960000 + 8192 * 1152);
    assert(s.disp.offset == 0x960000);

    s.disp.offset = 0x10000;
    assert(!refresh(&s, 16384, 1152));
    assert(sampled_hi == 0x960000 + 8192 * 1152);
    s.disp = (PPCMacGPUDisplayMode){ 0x10000, 8192, 1152 };
    assert(!refresh(&s, 8192, 1080));

    s.disp = (PPCMacGPUDisplayMode){ 0x10000, 8192, 1152 };
    yield_offset = 0x1300000;
    assert(!refresh(&s, 8192, 1152));
    assert(sampled_lo == 0x960000 && s.disp.offset == yield_offset);
    yield_offset = 0;

    s.disp.offset = 0x10000;
    s.regs.crtc_offset = s.vram_size - 4096;
    assert(!refresh(&s, 8192, 1152));
    assert(sampled_lo == 0x10000 && s.disp.offset == 0);

    s.disp.offset = 0x10000; s.regs.crtc_offset = 0x960000;
    s.r300_card_bytes = false;
    assert(!refresh(&s, 8192, 1152) && sampled_lo == 0x10000);
    s.disp.offset = 0x10000; s.r300_card_bytes = true; s.r300 = false;
    assert(!refresh(&s, 8192, 1152) && sampled_lo == 0x10000);
    s.disp.offset = 0x10000; s.r300 = true; s.rv250 = true;
    assert(!refresh(&s, 8192, 1152) && sampled_lo == 0x10000);
    assert(s.disp.offset == 0x960000 + R200_CRTC_OFFSET_BIAS);
    puts("PASS: native flip, geometry changes, yielding source change and legacy guards");
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="r300-pageflip-generation-") as tmp:
    c, exe = Path(tmp) / "test.c", Path(tmp) / "test"
    c.write_text(fixture + bias + "\n" + wrapper + main)
    subprocess.run(["cc", "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", str(c), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
