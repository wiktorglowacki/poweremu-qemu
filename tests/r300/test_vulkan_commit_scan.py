#!/usr/bin/env python3
"""Compare actual Vulkan commit scan prefixes, without a GPU or QEMU build."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(text, name):
    start = re.search(r"^static [^\n]*\b" + name + r"\([^;]*?\)\n\{", text, re.M).start()
    return text[start:text.index("\n}\n", start) + 3]


def typedef(text, name):
    start = text.index("typedef struct " + name + " {")
    return text[start:text.index("} " + name + ";", start) + len(name) + 3]


def source(path):
    text = path.read_text()
    prefix = function(text, "vk_commit_ex").split("    VkMemoryBarrier mb = {")[0]
    prefix = prefix.replace("static uint32_t", "static __attribute__((noinline)) uint32_t", 1)
    return PRELUDE + typedef(text, "VkImg") + typedef(text, "VkBatch") + STATE + function(text, "vk_img_dirty") + function(text, "vk_img_z") + STUBS + function(text, "vk_img_free") + prefix + "    return b->seq;\n}\n" + MAIN


PRELUDE = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef uint32_t VkFormat;
typedef uint64_t VkImage, VkDeviceMemory, VkImageView, VkCommandBuffer, VkFence;
typedef void GArray, GPtrArray;
#define VK_MAX_IMG 1024
#define VK_MAX_DR 8
#define VK_MAX_ST 16
#define ctz64(x) __builtin_ctzll(x)
#define g_assert assert
'''
STATE = r'''
static struct {
    VkImg img[VK_MAX_IMG];
    uint64_t dimg[VK_MAX_IMG / 64], zimg[VK_MAX_IMG / 64];
    VkBatch *cur;
    void *chunk;
    uint64_t stat_writebacks, stat_wb_z, stat_wb_commit, stat_commits;
    uint32_t pass_n;
    VkImg *pass_img[8];
    bool in_pass;
    uint16_t img_next[VK_MAX_IMG];
} V;
static VkBatch batch;
static bool lazy, harvest_clear, trace_on = true;
static uint64_t digest = 0xcbf29ce484222325ull;
static uint16_t free_head;
static void note(uint64_t x) { if (trace_on) { digest ^= x; digest *= 0x100000001b3ull; } }
'''
STUBS = r'''
enum { TRASH_IMAGE, TRASH_VIEW, TRASH_MEM };
static bool vk_lazy_z(void) { return lazy; }
static void vk_end_pass(void) { note(0x4000); }
static void vk_full_barrier(VkCommandBuffer cb) { note(0x5000 + cb); }
static void vk_harvest(void) { note(0x6000); if (harvest_clear) V.cur = NULL; }
static void vk_copy_to_vram_nb(VkImg *im)
{
    note(0x2000 + (im - V.img));
    V.stat_writebacks += im->ndr > 0;
    vk_img_dirty(im, false);
    im->ndr = 0;
}
static void vk_copy_to_vram(VkImg *im)
{
    note(0x1000 + (im - V.img));
    if (im->ndr && !V.cur) V.cur = &batch;
    vk_copy_to_vram_nb(im);
}
static uint16_t *vk_img_chain(uint32_t a, uint32_t w) { return &free_head; }
static void vk_fb_forget(VkImageView v) { note(0x7000 + v); }
static void vk_trash(VkBatch *b, int t, uint64_t h) { note(0x8000 + t + h); }
'''
MAIN = r'''
static uint32_t rng = 42;
static uint32_t random32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static double seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void check_masks(void)
{
    for (unsigned i = 0; i < VK_MAX_IMG; i++) {
        assert(!!(V.dimg[i / 64] & (1ull << (i % 64))) == V.img[i].dirty);
        assert(!!(V.zimg[i / 64] & (1ull << (i % 64))) == V.img[i].z);
    }
}
int main(int argc, char **argv)
{
    /* Exercise empty/full/sparse scans, word boundaries, zero rectangles,
     * lazy depth, and another commit during dirty-log synchronization. */
    for (unsigned test = 0; test < 10000; test++) {
        memset(&V, 0, sizeof(V));
        memset(&batch, 0, sizeof(batch));
        batch.state = 1; batch.seq = test + 1; batch.cb = 12;
        batch.draws = test % 173;
        V.cur = test & 1 ? &batch : NULL;
        lazy = test & 2;
        harvest_clear = test % 13 == 0;
        for (unsigned i = 0; i < VK_MAX_IMG; i++) {
            VkImg *im = &V.img[i];
            uint32_t r = random32();
            im->live = !!(r & 8);
            vk_img_z(im, r & 16);
            im->ndr = r % 4;
            vk_img_dirty(im, test % 5 == 0 || (test % 5 != 1 && r % 71 == 0));
        }
        note(vk_commit_ex(NULL, NULL, test & 4));
        note(V.stat_writebacks); note(V.stat_wb_z); note(V.stat_wb_commit);
        note(V.stat_commits); note(batch.wb); note(batch.wbz);
        check_masks();
        for (unsigned w = 0; w < VK_MAX_IMG / 64; w++) { note(V.dimg[w]); note(V.zimg[w]); }
    }
    /* Real dirty/depth setter and eviction functions clear reused slots. */
    memset(&V, 0, sizeof(V));
    for (unsigned i = 0; i < VK_MAX_IMG; i++) {
        VkImg *im = &V.img[i];
        im->live = true; im->ndr = i & 1;
        vk_img_z(im, true); vk_img_dirty(im, true);
        free_head = i + 1;
        vk_img_free(im);
        assert(!im->live && !im->dirty && !im->z);
        check_masks();
        im->live = true;
        vk_img_dirty(im, true); vk_img_z(im, false);
        vk_img_dirty(im, false);
        check_masks();
    }
    printf("differential_digest %016llx\n", (unsigned long long)digest);
    if (argc == 1) return 0;
    trace_on = false;
    /* This measures only CPU commit selection overhead; driver calls are
     * stubbed equally in both binaries, so it is not a guest speedup. */
    for (unsigned density = 0; density < 3; density++) {
        const unsigned count[] = {0, 2, 16};
        memset(&V, 0, sizeof(V));
        memset(&batch, 0, sizeof(batch));
        batch.state = 1; batch.seq = 1;
        lazy = true; harvest_clear = false;
        double t = seconds();
        for (unsigned n = 0; n < 200000; n++) {
            V.cur = &batch;
            for (unsigned j = 0; j < count[density]; j++) {
                unsigned i = (j * 67 + n) % VK_MAX_IMG;
                V.img[i].live = true; V.img[i].ndr = 1;
                vk_img_z(&V.img[i], j & 1);
                vk_img_dirty(&V.img[i], true);
            }
            vk_commit_ex(NULL, NULL, true);
        }
        printf("scan_benchmark %u_dirty %.3f ns/commit\n", count[density], (seconds() - t) * 1e9 / 200000);
    }
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", type=Path, default=Path(__file__).resolve().parents[2] / "hw/display/ppc_mac_gpu_vulkan.c")
    p.add_argument("--reference", type=Path)
    p.add_argument("--sanitize", action="store_true")
    p.add_argument("--bench", action="store_true")
    a = p.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="vk-commit-scan-") as tmp:
        for label, path in [("candidate", a.source), ("reference", a.reference)]:
            if not path:
                continue
            c, exe = Path(tmp) / (label + ".c"), Path(tmp) / label
            c.write_text(source(path))
            flags = ["-O1", "-g", "-fsanitize=address,undefined"] if a.sanitize else ["-O3", "-march=native"]
            subprocess.run([os.environ.get("CC", "cc"), "-std=c11", *flags, str(c), "-o", str(exe)], check=True)
            result = subprocess.run([str(exe)] + (["--bench"] if a.bench else []), text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            results[label] = result.stdout.strip().splitlines()
        assert results["candidate"][0] == "differential_digest 4263fb96cd18be48", results
        if a.reference:
            assert results["candidate"][0] == results["reference"][0], results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
