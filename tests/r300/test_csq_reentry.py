#!/usr/bin/env python3
"""Exercise the production PIO parser/drain under a yielding renderer callback.

Optional argument: another ppc_mac_gpu.c (the old source must fail). No VM/GPU.
"""
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'hw/display/ppc_mac_gpu.c'
text = source.read_text()


def function(name, required=True):
    match = re.search(r'^static [^\n]+\b' + name + r'\([^;]*?\)\n\{', text, re.M)
    if not match:
        assert not required, name
        return ''
    start = text.index('{', match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[match.start():end]


read = function('ppc_mac_gpu_mmio_read')
read = read[:read.index('    /*\n     * POWEREMU_STALL_TRACE:')] + \
       '    (void)val; (void)async_status; return 64;\n}\n'
reset = function('ppc_mac_gpu_reset')
reset = reset[reset.index('    qemu_spin_lock(&s->csq_ring_lock);'):]
reset = reset[:reset.index('    memset(&s->regs')]

prefix = r'''
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define qatomic_read(p) (*(p))
#define qatomic_load_acquire(p) (*(p))
#define qatomic_set(p,v) (*(p) = (v))
#define qatomic_store_release(p,v) (*(p) = (v))
#define qemu_spin_lock(p) ((void)(p))
#define qemu_spin_unlock(p) ((void)(p))
#define unlikely(v) (v)
#define BQL_LOCK_GUARD() ((void)0)
#define gpu_debug_log(...) ((void)0)
#define cps_count(...) ((void)0)
#define PPC_MAC_GPU_HWC_BASE 0xff00
#define PPC_MAC_GPU_HWC_END 0xff20
typedef uint64_t hwaddr;
typedef struct {
    bool (*busy_r200)(void *opaque);
} Renderer;
typedef struct {
    uint32_t csq_ring[4096], pm4_fifo[16384];
    unsigned csq_ring_count, csq_ring_tail, csq_ring_lock, csq_ring_epoch;
    bool csq_ring_active, pm4_pkt_one_reg, r300;
    unsigned csq_ring_deferred_tail;
    GArray *csq_ring_deferred;
    uint32_t pm4_pkt_count, pm4_pkt_reg, pm4_pkt_opcode, pm4_fifo_idx;
    unsigned r3_read_clean_n;
    struct {
        uint64_t stall_pio_dwords, stall_ring_dwords;
        bool csq_just_submitted;
        unsigned cp_csq_cntl, cp_rb_wptr, cp_rb_rptr, cp_me_cntl, cp_csq_stat_counter;
    } regs;
    Renderer *renderer;
} PPCMacGPUState;
static struct { unsigned hdr, type, opcode, count; } pm4_recent[32];
static unsigned pm4_recent_n;
static bool locked, r200_in_pm4;
static int g_in_pm4;
static bool cp_in_ring;
static bool bql_locked(void) { return locked; }
static void bql_lock(void) { locked = true; }
static void bql_unlock(void) { locked = false; }
static bool gpu_debug_enabled(void) { return false; }
static bool r200_traffic_on(void) { return false; }
static bool r300_type0_fast(PPCMacGPUState *s, unsigned a, unsigned v) { return false; }
static unsigned syncs;
static struct { unsigned epoch, kick; bool started; } cp;
static struct { unsigned ring_dwords; } r200_traffic;
static unsigned dma_runs, dma_pio_seen;
static void qemu_event_set(unsigned *event) { (*event)++; }
static bool ppc_mac_gpu_process_ring_buffer(PPCMacGPUState *s, unsigned r, unsigned w)
{ dma_runs++; dma_pio_seen = s->regs.stall_pio_dwords; return true; }
static void ppc_mac_gpu_rptr_writeback(PPCMacGPUState *s) {}
static void ppc_mac_gpu_cp_sync(PPCMacGPUState *s, hwaddr a);
static bool ppc_mac_gpu_cp_yield(PPCMacGPUState *s, unsigned i) { return false; }
static void pm4_packet_body(PPCMacGPUState *s, unsigned hdr);
static void ppc_mac_gpu_mmio_write(void *, hwaddr, uint64_t, unsigned);
static void ppc_mac_gpu_pm4_fifo_push(PPCMacGPUState *, uint32_t);
static void ppc_mac_gpu_process_pm4(PPCMacGPUState *, const uint32_t *, uint32_t);
static void ppc_mac_gpu_pm4_process_type0(PPCMacGPUState *, uint32_t, bool, uint32_t, uint32_t *);
'''
constants = ('R200_PM4_FIFO_DATA_BASE', 'R200_PM4_FIFO_DATA_END', 'R200_RBBM_STATUS',
             'R200_GUI_STAT', 'R200_RBBM_ACTIVE', 'R200_CP_CSQ_STAT', 'R300_CP_CSQ_AVAIL',
             'R200_CP_CSQ_CNTL', 'R200_CP_RB_RPTR', 'R200_CP_RB_WPTR')
header = (root / 'include/hw/display/ppc_mac_gpu.h').read_text()
for name in constants:
    prefix += re.search(r'^#define\s+' + name + r'\s+[^\n]+', header, re.M)[0] + '\n'

enqueue = function('ppc_mac_gpu_csq_enqueue', False)
if enqueue:
    start = text.index('typedef struct PPCMacGPUCSQWrite')
    prefix += text[start:text.index('} PPCMacGPUCSQWrite;', start) + len('} PPCMacGPUCSQWrite;')]

sync = 'static void ppc_mac_gpu_cp_sync(PPCMacGPUState *s, hwaddr a) ' \
       '{ syncs++; ppc_mac_gpu_cp_run(s); }\n'
# Keep the production loop/epoch/yield checks; replace only packet dispatch.
loop = function('ppc_mac_gpu_process_pm4')
loop = loop[:loop.index('        FILE *rd = r300_ringdump();')]
loop = loop.replace('ppc_mac_gpu_process_pm4', 'pm4_packet_loop', 1) + r'''
        (void)type;
        i++;
        unsigned n = MIN(((hdr >> 16) & 0x3fff) + 1, size_dw - i);
        pm4_packet_body(s, hdr);
        i += n;
    }
    g_in_pm4--;
}
'''
body = '\n'.join((function('ppc_mac_gpu_cp_run'), function('ppc_mac_gpu_cp_pending'), sync,
                  enqueue, function('ppc_mac_gpu_csq_ring_drain'),
                  function('ppc_mac_gpu_csq_ring_write'),
                  function('ppc_mac_gpu_pm4_process_type0'),
                  function('ppc_mac_gpu_pm4_fifo_push'), read,
                  function('ppc_mac_gpu_mmio_read_guest'),
                  function('ppc_mac_gpu_mmio_write_guest'),
                  function('ppc_mac_gpu_pre_save'), loop))

tests = r'''
static unsigned failed, scenario, events[16], nevents, hook_count;
static void check(bool ok, const char *why) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", why); failed++; }
}
static void reset_queue(PPCMacGPUState *s) {
RESET_QUEUE
    s->pm4_pkt_count = s->pm4_fifo_idx = 0;
    memset(&s->regs, 0, sizeof(s->regs));
}
static void word(PPCMacGPUState *s, uint32_t value) {
    ppc_mac_gpu_csq_ring_write(s, 0, value, 4);
    check(s->csq_ring_count <= ARRAY_SIZE(s->csq_ring), "ring capacity exceeded");
}
static void hook(PPCMacGPUState *s) {
    if (hook_count++) { return; }
    s->regs.cp_rb_wptr = 2;
    ppc_mac_gpu_cp_run(s);
    check(dma_runs == 0, "DMA processor reentered the PIO owner's renderer");
    unsigned old_syncs = syncs;
    check(ppc_mac_gpu_mmio_read_guest(s, R200_RBBM_STATUS, 4) & R200_RBBM_ACTIVE,
          "active copied words reported idle");
    check(ppc_mac_gpu_mmio_read_guest(s, R200_CP_CSQ_STAT, 4) != 0,
          "active CSQ reported drained");
    check(syncs == old_syncs, "nested status entered the DMA command processor");
    if (scenario == 1) {
        /* BQL-held fallback must queue, not enter the parser again. */
        word(s, 0xc0000300); word(s, 30);
        ppc_mac_gpu_mmio_write_guest(s, 0x5000, 40, 4);
        word(s, 0xc0000500);         /* partial next packet */
    } else if (scenario == 2) {
        /* Lock-free producer fills the fixed queue while the owner waits. */
        locked = false;
        for (unsigned i = 0; i < 4100; i++) { word(s, 0x80000000); }
        locked = true;
        ppc_mac_gpu_mmio_write_guest(s, 0x5000, 40, 4);
        check((ppc_mac_gpu_mmio_read_guest(s, R200_RBBM_STATUS, 4) & 0x7f) == 0,
              "spilled queue still advertises room");
    } else {
        reset_queue(s);
        word(s, 0xc0000900); word(s, 90);
    }
}
static void ppc_mac_gpu_mmio_write(void *opaque, hwaddr addr, uint64_t val, unsigned size) {
    PPCMacGPUState *s = opaque;
    if (addr >= R200_PM4_FIFO_DATA_BASE && addr < R200_PM4_FIFO_DATA_END) {
        ppc_mac_gpu_pm4_fifo_push(s, val);
    } else {
        events[nevents++] = val;
        if (scenario == 4 && val == 10) { hook(s); }
    }
}
static void ppc_mac_gpu_process_pm4(PPCMacGPUState *s, const uint32_t *d, uint32_t n) {
    unsigned op = (d[0] >> 8) & 255;
    events[nevents++] = d[1];
    if (op == 1) { hook(s); }
    if (op == 2) {
        check(n == 3 && d[1] == 20 && d[2] == 21, "packet body corrupted by nested parser");
        if (scenario == 2) {
            check(s->regs.stall_pio_dwords == 5, "full-ring words overtook the local suffix");
        }
    }
}
static void test(unsigned mode) {
    PPCMacGPUState s = {0};
    unsigned expected[] = {10, 20, 30, 40, 50};
    scenario = mode == 5 ? 1 : mode; nevents = hook_count = syncs = dma_runs = 0;
    memset(&cp, 0, sizeof(cp)); cp.started = mode == 5;
    locked = false;
    word(&s, mode == 4 ? (1u << 16) | (0x5000 / 4) : 0xc0000100);
    word(&s, 10);
    if (mode == 4) { word(&s, 11); }
    word(&s, 0xc0010200); word(&s, 20); word(&s, 21);
    locked = true;
    ppc_mac_gpu_csq_ring_drain(&s);
    if (mode == 5) {
        check(cp.kick != 0 && dma_runs == 0, "PIO owner did not wake the DMA worker");
        ppc_mac_gpu_cp_run(&s);
    }
    if (scenario == 1) {
        check(s.pm4_pkt_count == 1 && s.pm4_fifo_idx == 0, "partial packet state lost");
        check(ppc_mac_gpu_mmio_read_guest(&s, R200_CP_CSQ_STAT, 4) != 0,
              "partial packet reported complete");
        word(&s, 50);
        check(nevents == 5 && !memcmp(events, expected, sizeof(expected)),
              "reentrant writes violated FIFO order");
    } else if (mode == 2) {
        check(nevents == 3 && events[0] == 10 && events[1] == 20 && events[2] == 40,
              "full-ring direct write overtook pending words");
        check(s.regs.stall_pio_dwords == 4105, "full-ring path lost or duplicated words");
    } else {
        check(nevents == 2 && events[0] == 10 && events[1] == 90,
              "reset executed a copied pre-reset command");
    }
    check(!s.csq_ring_active && !s.csq_ring_count && !s.csq_ring_deferred,
          "owner or deferred queue survived drain");
    check(dma_runs == ((scenario == 1 || scenario == 2) ? 1 : 0),
          "PIO completion stranded pending DMA commands");
    g_clear_pointer(&s.csq_ring_deferred, g_array_unref);
}
static void dma_owner(void) {
    PPCMacGPUState s = {0};
    nevents = dma_runs = syncs = 0; scenario = 0;
    memset(&cp, 0, sizeof(cp));
    locked = true; g_in_pm4 = 1; s.regs.cp_rb_wptr = 2;
    ppc_mac_gpu_mmio_write_guest(&s, 0x5000, 42, 4);
    word(&s, 0x80000000);
    ppc_mac_gpu_cp_run(&s);
    check(nevents == 0 && dma_runs == 0 && syncs == 0,
          "guest/PIO/DMA work reentered a yielding DMA packet");
    check(ppc_mac_gpu_mmio_read_guest(&s, R200_RBBM_STATUS, 4) & R200_RBBM_ACTIVE,
          "yielding DMA packet reported idle");
    g_in_pm4 = 0;
    ppc_mac_gpu_csq_ring_drain(&s);
    check(nevents == 1 && events[0] == 42 && dma_runs == 1 &&
          s.regs.stall_pio_dwords == 1, "DMA owner stranded deferred writes");
}
static void dma_before_fifo(void) {
    PPCMacGPUState s = {0};
    nevents = dma_runs = syncs = 0; scenario = 0;
    memset(&cp, 0, sizeof(cp)); cp.started = true;
    locked = true; s.regs.cp_rb_wptr = 2;
    ppc_mac_gpu_mmio_write_guest(&s, R200_PM4_FIFO_DATA_BASE, 0xc0000600, 4);
    ppc_mac_gpu_mmio_write_guest(&s, R200_PM4_FIFO_DATA_BASE, 60, 4);
    check(dma_runs == 1 && dma_pio_seen == 0 && syncs == 2 &&
          nevents == 1 && events[0] == 60,
          "direct FIFO path overtook older DMA (CSQ_RING=0)");
}
static void partial_save(void) {
    PPCMacGPUState s = {0};
    nevents = dma_runs = syncs = 0; scenario = 0;
    memset(&cp, 0, sizeof(cp)); locked = false;
    word(&s, 0xc0010700);
    locked = true;
    check(ppc_mac_gpu_pre_save(&s) == -EBUSY,
          "migration accepted an unsaved partial PIO packet");
    word(&s, 70); word(&s, 71);
    check(ppc_mac_gpu_pre_save(&s) == 0 && nevents == 1 && events[0] == 70,
          "migration did not recover after completing the PIO packet");
}
static unsigned ib_draws;
static void pm4_packet_body(PPCMacGPUState *s, unsigned hdr) {
    if (((hdr >> 8) & 255) == 0x3f) {
        uint32_t ib[] = { 0xc0000100, 10, 0xc0000200, 20 };
        pm4_packet_loop(s, ib, ARRAY_SIZE(ib));
    } else {
        ib_draws++;
        if (ib_draws == 1) { reset_queue(s); }
    }
}
static void reset_indirect_buffer(void) {
    PPCMacGPUState s = {0};
    uint32_t outer[] = { 0xc0003f00, 0, 0xc0000300, 30 };
    ib_draws = 0;
    pm4_packet_loop(&s, outer, ARRAY_SIZE(outer));
    check(ib_draws == 1 && g_in_pm4 == 0,
          "reset inside IB executed old inner/outer packet suffixes");
}
int main(void) {
    dma_owner();
    dma_before_fifo();
    partial_save();
    reset_indirect_buffer();
    for (unsigned i = 1; i <= 5; i++) { test(i); }
    printf("CSQ reentry: %u failures (FIFO, partial packet, busy, full ring, reset, DMA, IB, migration)\n", failed);
    return !!failed;
}
'''.replace('RESET_QUEUE', reset)

with tempfile.TemporaryDirectory(prefix='csq-reentry-') as tmp:
    path = Path(tmp)
    (path / 'check.c').write_text(prefix + body + tests)
    flags = shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', '--libs', 'glib-2.0'], text=True))
    subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Wno-unused-parameter',
                    '-Wno-unused-function', str(path / 'check.c'), '-o', str(path / 'check'),
                    *flags], check=True)
    subprocess.run([str(path / 'check')], check=True)
