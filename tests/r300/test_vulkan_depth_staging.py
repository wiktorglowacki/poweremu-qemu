#!/usr/bin/env python3
"""Inject allocation failures into the renderer's depth staging transaction."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from test_vulkan_commit_scan import function


source = (Path(__file__).resolve().parents[2] / "hw/display/ppc_mac_gpu_vulkan.c").read_text()
# Keep these invariants at the callers: reserving late loses dirty depth on eviction.
get_image = function(source, "vk_img_get")
assert get_image.index("if (!vk_tile_staging(bytes))") < get_image.index("vk_img_free(")
assert "vk_tile_staging(" not in function(source, "vk_zconv")

prelude = r"""
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ROUND_UP(x,a) (((x) + (a) - 1) / (a) * (a))
enum { TRASH_BUF, TRASH_MEM };
static struct {
    VkDevice dev;
    VkBuffer tile_buf, zs_buf, vram_buf;
    VkDeviceMemory tile_mem, zs_mem;
    VkDescriptorSetLayout tile_dsl;
    VkDescriptorPool tile_pool;
    VkDescriptorSet tile_set;
    VkDeviceSize tile_size;
} V;
static int step, fail, live, retired;
static uintptr_t serial;
static VkDeviceSize pow2ceil(VkDeviceSize n) {
    VkDeviceSize v = 1;
    while (v < n) v *= 2;
    return v;
}
static bool vk_zstage(void) { return true; }
#define HANDLE(T) ((T)++serial)
static bool vk_buffer(VkDeviceSize n, VkBufferUsageFlags u, int f, int p,
                      VkBuffer *b, VkDeviceMemory *m, void *map) {
    *b = VK_NULL_HANDLE;
    *m = VK_NULL_HANDLE;
    if (++step == fail) return false;
    *b = HANDLE(VkBuffer);
    *m = HANDLE(VkDeviceMemory);
    live += 2;
    return true;
}
static void vk_trash(void *batch, int type, uint64_t h) {
    if (h) { retired++; live--; }
}
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) {
    if (b) live--;
}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) {
    if (m) live--;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo *i,
        const VkAllocationCallbacks *a, VkDescriptorPool *pool) {
    if (++step == fail) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *pool = HANDLE(VkDescriptorPool);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo *i,
                                                VkDescriptorSet *set) {
    if (++step == fail) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *set = HANDLE(VkDescriptorSet);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet *w,
                                                  uint32_t c, const VkCopyDescriptorSet *p) {
    assert(n == 2 && w[0].dstSet == w[1].dstSet);
}
"""
main = r"""
int main(void) {
    /* Tight and padded pitches, full and narrow dirty regions, both tile modes. */
    for (uint32_t band = 1; band <= 32; band *= 32) {
        for (uint32_t pitch = 6144; pitch <= 24576; pitch *= 2) {
            uint32_t width = 1533, height = 4097;
            uint32_t rows = vk_zconv_rows(width, pitch);
            assert(rows && rows < height);
            VkDeviceSize reserve = MAX((VkDeviceSize)ROUND_UP(width, 4) * 5 * rows,
                                       (VkDeviceSize)(rows + 2 * (band - 1)) * pitch);
            for (uint32_t y = 0; y < height; y++) {
                uint32_t end = y + rows < height ? y + rows : height;
                VkDeviceSize copy = (VkDeviceSize)(ROUND_UP(end, band) - y / band * band) * pitch;
                assert(copy <= reserve);
            }
        }
    }
    for (fail = 1; fail <= 4; fail++) {
        step = 0;
        assert(!vk_tile_staging(1));
        assert(!V.tile_size && !V.tile_set && !live);
    }
    fail = 0;
    assert(vk_tile_staging(1));
    assert(live == 4);
    VkDescriptorSet old = V.tile_set;
    VkBuffer tile = V.tile_buf, zs = V.zs_buf;
    VkDeviceSize old_size = V.tile_size;
    for (fail = 1; fail <= 3; fail++) {
        step = 0;
        assert(!vk_tile_staging(old_size + 1));
        assert(vk_tile_staging(old_size));
        assert(live == 4 && !retired && V.tile_size == old_size);
        assert(V.tile_buf == tile && V.zs_buf == zs && V.tile_set == old);
    }
    fail = 0;
    assert(vk_tile_staging(old_size + 1));
    assert(live == 4 && retired == 4 && V.tile_set != old);
    return 0;
}
"""
cc = shlex.split(os.getenv("CC", "cc")) + shlex.split(os.getenv("CPPFLAGS", ""))
if subprocess.run([*cc, "-E", "-x", "c", "-"], input=b"#include <vulkan/vulkan.h>\n",
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    print("Skipping Vulkan depth staging test (Vulkan headers unavailable).")
    raise SystemExit(0)

with tempfile.TemporaryDirectory() as tmp:
    program = Path(tmp) / "depth-staging"
    subprocess.run([*cc, "-std=c11", "-x", "c", "-", "-o", program],
                   input=prelude + function(source, "vk_zconv_rows") + function(source, "vk_tile_staging") + main, text=True, check=True)
    subprocess.run([program], check=True)
print("Depth staging initial failure, failed growth, old reservation and retry passed.")
