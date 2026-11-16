#!/usr/bin/env python3
"""Exercise actual batch acquisition, reaping and unlocked fence polling."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / 'hw/display/ppc_mac_gpu_vulkan.c').read_text()

def function(start):
    begin = source.index(start)
    brace = source.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]

batch = source[source.index('typedef struct VkBatch {'):source.index('} VkBatch;') + len('} VkBatch;')]
count = re.search(r'^#define VK_NBATCH\s+\d+', source, re.M).group()
header = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uint64_t VkDevice, VkFence, VkCommandBuffer;
typedef int32_t VkResult;
typedef void GArray, GPtrArray;
enum { VK_SUCCESS = 0, VK_NOT_READY = 1, VK_TIMEOUT = 2, VK_ERROR_DEVICE_LOST = -4,
       VK_TRUE = 1,
       VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 1,
       VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 1 };
typedef struct { int sType; unsigned flags; } VkCommandBufferBeginInfo;
#define qatomic_read(p) (*(p))
#define qatomic_cmpxchg(p, old, value) __sync_val_compare_and_swap((p), (old), (value))
#define g_assert(c) assert(c)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) MIN((hi), ((v) < (lo) ? (lo) : (v)))
#define g_ascii_strtoll strtoll
'''
fixture = r'''
static struct {
    VkDevice dev;
    VkBatch batch[VK_NBATCH], *cur, *poll_batch;
    void *chunk;
    uint32_t seq, done_seq, worker_seq;
    uint64_t stat_batch_waits;
    int lock, idle;
    void *waitq;
} V;
static struct {
    uint32_t wait_seq;
    int64_t wait_start;
    uint64_t commits, commit_us, commit_draws;
} vkw;
static bool held, vcpu;
static int64_t clock_us;
static unsigned queries, blocking_waits, resets[VK_NBATCH], queue_items, spin_env_reads;
static uint64_t last_timeout_ns;
static VkResult wait_result;
static int64_t unlock_advance;
static bool worker_on_unlock;
static VkResult result;
static void (*query_hook)(void);
static VkBatch *allocated;
static void bql_lock(void) { assert(!held); held = true; }
static void bql_unlock(void) {
    assert(held); held = false;
    clock_us += unlock_advance;
    if (worker_on_unlock) V.done_seq = VK_NBATCH;
}
static bool bql_locked(void) { return held; }
static bool qemu_in_vcpu_thread(void) { return vcpu; }
static int64_t g_get_monotonic_time(void) { return clock_us++; }
static void cpu_relax(void) { }
static void qemu_mutex_lock(void *p) { }
static void qemu_mutex_unlock(void *p) { }
static bool g_queue_is_empty(void *p) { return queue_items == 0; }
static void qemu_cond_wait(void *c, void *m) { assert(!"unexpected allocator wait"); }
static void vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) { }
static void vk_full_barrier(VkCommandBuffer cb) { }
static void vk_recycle(VkBatch *b) {
    assert(held && b != V.poll_batch);
    resets[b - V.batch]++;
    b->state = 0;
}
static bool vk_wait_stats(void) { return false; }
static uint32_t vk_commit_ex(void (*done)(void *, uint32_t), void *arg, bool z) { return 0; }
static char *test_getenv(const char *name) {
    assert(held && !strcmp(name, "PPCGPU_VK_SPIN_US"));
    spin_env_reads++;
    return NULL;
}
#define getenv test_getenv
static VkResult vkWaitForFences(VkDevice dev, unsigned count, const VkFence *fence,
                                bool all, uint64_t timeout_ns) {
    assert(dev == 42 && count == 1 && all);
    assert(*fence == 1000 + VK_NBATCH - 1);
    assert(timeout_ns > 0 && timeout_ns != UINT64_MAX);
    if (!held) assert(V.poll_batch == &V.batch[VK_NBATCH - 1]);
    blocking_waits++;
    last_timeout_ns = timeout_ns;
    clock_us += timeout_ns / 1000;
    return wait_result;
}
static void vk_wait_note(uint32_t seq, int64_t submitted, uint32_t draws,
                         uint32_t wb, uint32_t wbz, int64_t now) { }
static VkResult vkGetFenceStatus(VkDevice dev, VkFence fence) {
    assert(dev == 42 && fence == 1000 + VK_NBATCH - 1);
    if (!held) assert(V.poll_batch == &V.batch[VK_NBATCH - 1]);
    queries++;
    if (query_hook) {
        void (*hook)(void) = query_hook;
        query_hook = NULL;
        hook();
    }
    assert(resets[VK_NBATCH - 1] == 0);
    return result;
}
'''
main = r'''
static void setup(void) {
    memset(&V, 0, sizeof(V));
    memset(&vkw, 0, sizeof(vkw));
    memset(resets, 0, sizeof(resets));
    held = vcpu = true;
    clock_us = 100;
    queries = blocking_waits = 0;
    last_timeout_ns = 0;
    wait_result = VK_TIMEOUT;
    unlock_advance = 0;
    worker_on_unlock = false;
    queue_items = 1;
    query_hook = NULL;
    allocated = NULL;
    V.dev = 42;
    V.seq = VK_NBATCH;
    for (unsigned i = 0; i < VK_NBATCH; i++) {
        V.batch[i].seq = i + 1;
        V.batch[i].fence = 1000 + i;
    }
    V.batch[VK_NBATCH - 1].state = 2;
    V.batch[VK_NBATCH - 1].submitted = clock_us;
}
static void concurrent_caller(void) {
    VkBatch *owner = V.poll_batch;
    unsigned before = queries;
    bql_lock();
    /* A second caller sees a newer batch, but must leave the reservation alone. */
    V.batch[0].state = 2;
    V.batch[0].seq = ++V.seq;
    queue_items++;
    assert(vk_wait_busy_r200(NULL, 3));
    assert(held && V.poll_batch == owner && queries == before);
    bql_unlock();
}
static void allocator_pressure(void) {
    bql_lock();
    /* The waiter has finished all six, including its callbacks. */
    vk_set_done(VK_NBATCH);
    V.worker_seq = VK_NBATCH;
    queue_items = 0;
    allocated = vk_batch();
    assert(allocated != V.poll_batch);
    assert(V.poll_batch == &V.batch[VK_NBATCH - 1]);
    for (unsigned i = 0; i < VK_NBATCH - 1; i++) assert(resets[i] == 1);
    assert(resets[VK_NBATCH - 1] == 0);
    bql_unlock();
}
int main(void) {
    setup(); result = VK_SUCCESS;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(held && queries == 1 && V.done_seq == VK_NBATCH);
    assert(V.worker_seq == 0 && queue_items == 1);
    assert(!V.poll_batch && V.batch[VK_NBATCH - 1].state == 2);

    setup(); result = VK_NOT_READY;
    assert(vk_wait_busy_r200(NULL, 20));
    assert(held && queries && clock_us >= 120 && !V.done_seq && !V.worker_seq);
    assert(!V.poll_batch && queue_items == 1);

    setup(); result = VK_ERROR_DEVICE_LOST;
    assert(vk_wait_busy_r200(NULL, 20));
    assert(held && queries == 1 && !V.done_seq && !V.worker_seq);
    assert(!V.poll_batch && queue_items == 1);

    setup(); result = VK_SUCCESS; query_hook = concurrent_caller;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(queries == 1 && !V.poll_batch && V.done_seq == VK_NBATCH);
    assert(!vk_seq_done(V.batch[0].seq) && queue_items == 2);

    setup(); result = VK_SUCCESS; query_hook = allocator_pressure;
    for (unsigned i = 0; i < VK_NBATCH; i++) V.batch[i].state = 2;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(!V.poll_batch && allocated && V.stat_batch_waits == 0);
    assert(resets[VK_NBATCH - 1] == 0);
    vk_reap();
    assert(resets[VK_NBATCH - 1] == 1 && allocated->state == 1);

    /* A young batch spins only until its submission-age cap, then blocks. */
    setup(); result = VK_NOT_READY;
    assert(vk_wait_busy_r200(NULL, 250));
    assert(queries > 100 && queries < 205 && blocking_waits == 1);
    assert(last_timeout_ns > 0 && last_timeout_ns <= 52000);
    assert(!V.done_seq && !V.worker_seq && !V.poll_batch && held);

    /* Repeated waits on that aged batch do not renew a 250us CPU spin. */
    unsigned previous_queries = queries;
    assert(vk_wait_busy_r200(NULL, 250));
    assert(queries == previous_queries + 1 && blocking_waits == 2);
    assert(last_timeout_ns <= 250000 && !V.done_seq && held);

    setup(); result = VK_NOT_READY; clock_us = 400; wait_result = VK_SUCCESS;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(queries == 1 && blocking_waits == 1 && V.done_seq == VK_NBATCH);
    assert(!V.worker_seq && queue_items == 1 && !V.poll_batch && held);

    setup(); result = VK_NOT_READY; clock_us = 400; wait_result = VK_ERROR_DEVICE_LOST;
    assert(vk_wait_busy_r200(NULL, 20));
    assert(queries == 1 && blocking_waits == 1 && !V.done_seq && held);

    /* Expiry while dropping BQL must never turn into an infinite wait. */
    setup(); result = VK_SUCCESS; unlock_advance = 30;
    assert(vk_wait_busy_r200(NULL, 20));
    assert(!queries && !blocking_waits && !V.done_seq && !V.poll_batch && held);

    setup(); result = VK_SUCCESS; worker_on_unlock = true;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(!queries && !blocking_waits && V.done_seq == VK_NBATCH && held);

    setup(); result = VK_SUCCESS; vcpu = false;
    assert(!vk_wait_busy_r200(NULL, 20));
    assert(held && queries == 1 && !blocking_waits && !V.poll_batch);

    setup(); result = VK_SUCCESS;
    assert(!vk_wait_busy_r200(NULL, 0));
    assert(held && queries == 1 && !blocking_waits && !V.poll_batch);
    assert(spin_env_reads == 1);

    printf("PASS: actual spin/fence helpers, young/aged cap, bounded remaining timeout, expiry, completion/error, worker/fallback, six-slot pin and zero-time poll\n");
    return 0;
}
'''
functions = '\n\n'.join(function(s) for s in [
    'static bool vk_seq_done(', 'static void vk_set_done(',
    'static void vk_reap(', 'static VkBatch *vk_batch(',
    'static VkBatch *vk_newest_pending(', 'static int64_t vk_spin_us(',
    'static bool vk_fence_wait(', 'static bool vk_wait_busy_r200(',
])
with tempfile.TemporaryDirectory(prefix='r300-fence-poll-') as tmp:
    c = Path(tmp) / 'test.c'
    exe = Path(tmp) / 'test'
    c.write_text(header + count + '\n' + batch + fixture + functions + main)
    subprocess.run(['cc', '-O2', '-Wall', '-Wextra', '-Wno-unused-parameter',
                    str(c), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
