#!/usr/bin/env python3
"""Check the renderer's actual layout policy, bounds and GLSL address math."""
import ast
from pathlib import Path
import re
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[2] /
          "hw/display/ppc_mac_gpu_vulkan.c").read_text()


def function(text, name):
    match = re.search(r"(?:static\s+)?(?:inline\s+)?(?:void|bool|uint|uint32_t|uint64_t)\s+" +
                      name + r"\s*\([^;]+?\)\s*\{", text)
    assert match, name
    start = text.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[match.start():end]


shader = source.split("static const char vk_tile_glsl[] =", 1)[1].split(";\n", 1)[0]
shader = "".join(ast.literal_eval(line.strip()) for line in shader.splitlines()
                 if line.strip().startswith('"'))
address = re.sub(r"\buint\b", "uint32_t", function(shader, "tiled"))
helpers = "\n".join(function(source, name) for name in
                    ("vk_vram_tiling", "vk_tile_band", "vk_span_hi",
                     "vk_img_hi", "vk_img_written"))
prefix = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { IMG_RT, IMG_TEX };
static int g_strcmp0(const char *a, const char *b) { return !a || !b ? (a != b) : strcmp(a, b); }
#define QEMU_ALIGN_UP(n, a) (((n) + (a) - 1) / (a) * (a))
#define QEMU_ALIGN_DOWN(n, a) ((n) / (a) * (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static bool opt_in;
static bool vk_tiled_rt_enabled(void) { return opt_in; }
static struct { uint32_t addr, pitch, bpp, mode; } p;
typedef struct {
    int cls;
    uint32_t addr, width, height, pitch, bpp, tiling;
    uint32_t dr[1][4], ndr;
    bool dirty, valid;
    uint64_t gen;
    bool sp_ok;
    uint32_t sp_seq;
    uint64_t sp_p0, sp_e;
} VkImg;
static struct {
    uint64_t gen, npages, page_wgen[4], page_gen[4], page_gprev[4], page_wr[4][2], blk_gen[1];
    uint32_t page_wseq[4], blk_wseq[1];
    uint8_t page_src[4];
} V;
static struct Batch { uint32_t seq; } batch = { 7 };
static struct Batch *vk_batch(void) { return &batch; }
static uint64_t vk_pg(uint64_t addr) { return addr / 4096; }
static void vk_img_dirty(VkImg *im, bool d) { im->dirty = d; }
static void vk_stamp_drop(uint64_t lo, uint64_t hi) { (void)lo; (void)hi; }
/* Isolate the production page-stamping loop for a nonempty first draw. */
static int vk_dr_slot(VkImg *im, uint32_t *in, uint32_t *out)
{ (void)im; (void)in; (void)out; return 0; }
'''
checks = r'''
int main(void)
{
    for (unsigned mode = 0; mode < 4; mode++) {
        opt_in = false;
        assert(vk_vram_tiling(IMG_RT, mode) == (mode & 2));
        assert(vk_vram_tiling(IMG_TEX, mode) == mode);
        opt_in = true;
        assert(vk_vram_tiling(IMG_RT, mode) == mode);
    }
    opt_in = false;
    assert(vk_span_hi(IMG_RT, 0, 7, 3, 32, 4, 2) == 128);
    assert(vk_span_hi(IMG_RT, 0, 7, 3, 32, 4, 3) == 128);
    assert(vk_span_hi(IMG_RT, 0, 7, 3, 32, 4, 1) == 92);
    /* A linear-sized allocation at VRAM's end must no longer be accepted. */
    assert(vk_span_hi(IMG_RT, 4096 - 96, 7, 3, 32, 4, 2) > 4096);
    assert(vk_span_hi(IMG_RT, 0xfffffff0u, 7, 3, 32, 4, 2) == 0x100000070ull);
    /* Row zero's microtiles cross page 1; linear/macro still use page 0. */
    for (unsigned mode = 0; mode < 4; mode++) {
        memset(&V, 0, sizeof(V));
        V.npages = 4;
        VkImg im = { .cls = IMG_RT, .width = 1024, .height = 4,
            .pitch = 4096, .bpp = 4, .tiling = mode };
        vk_img_written(&im, 0, 0, 1024, 1);
        bool micro = mode & 2;
        assert(V.page_gen[0] == 1 && V.page_gen[1] == micro);
        assert(V.page_gen[2] == 0 && V.page_gen[3] == 0);
        assert(V.page_wseq[0] == 7 && V.page_src[0] == 2);
        assert(!micro || (V.page_wseq[1] == 7 && V.page_src[1] == 2));
        assert(im.gen == 1 && im.dirty && im.valid && im.ndr == 1);
        /* Odd row within the next band must stamp both later pages. */
        vk_img_written(&im, 0, 3, 1024, 4);
        assert(V.page_gen[2] == (micro ? 2 : 0) && V.page_gen[3] == 2);
        assert(im.gen == 2);
        /* The same pages again in the same batch: nothing to stamp. */
        vk_img_written(&im, 0, 3, 1024, 4);
        assert(V.gen == 2 && im.gen == 2 && V.page_gen[3] == 2);
        batch.seq = 8;
        vk_img_written(&im, 0, 3, 1024, 4);
        assert(V.gen == 3 && im.gen == 3 && V.page_gen[3] == 3 && V.page_wseq[3] == 8);
        im.sp_ok = false;
        vk_img_written(&im, 0, 3, 1024, 4);
        assert(V.gen == 4 && V.page_gen[3] == 4);
        batch.seq = 7;
    }
    unsigned pixels = 0;
    for (unsigned w = 1; w <= 17; w++) {
        for (unsigned h = 1; h <= 9; h++) {
            p.addr = 0; p.pitch = QEMU_ALIGN_UP(w, 4) * 4;
            p.bpp = 4; p.mode = 2;
            unsigned extent = vk_span_hi(IMG_RT, 0, w, h, p.pitch, 4, 2);
            uint32_t words[256];
            bool touched[256] = { false };
            assert(extent / 4 + 1 < 256);
            for (unsigned i = 0; i < 256; i++) words[i] = 0xdeadbeef;
            for (unsigned y = 0; y < h; y++) for (unsigned x = 0; x < w; x++) {
                unsigned offset = tiled(x, y);
                /* Four-by-two tiles, row-major within each 32-byte tile. */
                unsigned reference = ((y / 2) * (p.pitch / 16) + x / 4) * 32 +
                                     ((y % 2) * 4 + x % 4) * 4;
                assert(offset == reference && offset + 4 <= extent);
                assert(!touched[offset / 4]);
                touched[offset / 4] = true;
                words[offset / 4] = 0xff000000 | (y << 16) | x;
                pixels++;
            }
            for (unsigned y = 0; y < h; y++) for (unsigned x = 0; x < w; x++)
                assert(words[tiled(x, y) / 4] == (0xff000000 | (y << 16) | x));
            /* An interior update crosses tile edges without changing neighbours. */
            for (unsigned y = 1; y + 1 < h; y++) for (unsigned x = 1; x + 1 < w; x++)
                words[tiled(x, y) / 4] ^= 0x00008000;
            for (unsigned y = 0; y < h; y++) for (unsigned x = 0; x < w; x++) {
                uint32_t value = 0xff000000 | (y << 16) | x;
                if (x && x + 1 < w && y && y + 1 < h) value ^= 0x00008000;
                assert(words[tiled(x, y) / 4] == value);
            }
            for (unsigned i = 0; i < 256; i++)
                if (!touched[i]) assert(words[i] == 0xdeadbeef);
        }
    }
    printf("PASS: micro-only policy, macro exclusion, page stamps, odd-height bounds, "
           "%u unique-pixel tile roundtrips and padding guards\n", pixels);
}
'''
# The same policy must govern range accounting, upload and writeback.
assert "!vk_vram_tiling(im->cls, im->tiling)" in source
assert "vk_tile_band(cls, tiling, bpp)" in source
# Software depth uses colour-aspect integer images: never opt it in.
assert "zd->tiling & (vk_tiled_rt_enabled() ? 3 : 1)" in source
with tempfile.TemporaryDirectory(prefix="r300-microtiles-") as tmp:
    path = Path(tmp)
    (path / "check.c").write_text(prefix + helpers + "\n" + address + checks)
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-Wno-sign-compare", "-O2",
                    str(path / "check.c"), "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)
