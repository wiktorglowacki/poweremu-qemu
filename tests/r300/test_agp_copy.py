#!/usr/bin/env python3
"""Exercise actual copy recognition/pixel conversion and checked AGP writes."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
device = (root / "hw/display/ppc_mac_gpu.c").read_text()


def function(name):
    match = re.search(r"static (?:uint8_t \*|bool )" + name + r"\([^;]+?\)\n\{", device)
    assert match, name
    start = device.index("{", match.start())
    end, depth = start + 1, 1
    while depth:
        depth += (device[end] == "{") - (device[end] == "}")
        end += 1
    return device[match.start():end]


prefix = r'''
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>
#define MiB (1024u * 1024u)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define g_try_malloc malloc
#define g_free free
typedef uint64_t hwaddr;
typedef struct {
    unsigned char vram[32768]; bool r3_zpass_active;
    struct { uint32_t mc_agp_location, aic_pt_base; } regs;
} PPCMacGPUState;
static unsigned char ram[32768];
static unsigned maps[8], writes, syncs, strict_pte_reads, legacy_pte_reads;
static int failed_page, invalid_page, failed_write, nonram_page, readonly_page,
           short_page, address_space_memory;
typedef int MemTxResult;
typedef struct { bool memory, unspecified; } MemTxAttrs;
#define MEMTX_OK 0
#define MEMTXATTRS_UNSPECIFIED ((MemTxAttrs){.unspecified=true})
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define le32_to_cpu(x) (x)
#else
#define le32_to_cpu(x) __builtin_bswap32(x)
#endif
#define RCU_READ_LOCK_GUARD() ((void)0)
typedef struct { bool ram, readonly; } MemoryRegion;
static bool memory_region_is_ram(MemoryRegion *mr) { return mr->ram; }
static bool memory_access_is_direct(MemoryRegion *mr, bool wr, MemTxAttrs attr)
{ assert(wr && attr.memory); return mr->ram && !mr->readonly; }
#define gpu_debug_log(...) ((void)0)
static hwaddr uninorth_get_agp_gart_base(void) { return 0x100000; }
static int address_space_read(void *as, hwaddr a, MemTxAttrs attrs, void *out, unsigned n)
{
    (void)as;
    if (attrs.memory) { assert(!attrs.unspecified); strict_pte_reads++; }
    else { assert(attrs.unspecified); legacy_pte_reads++; }
    if (n!=4 || a<0x100000 || a>=0x100020 || a%4) return 1;
    unsigned page=(a-0x100000)/4;
    if ((int)page==failed_page) return 1;
    uint32_t pte=maps[page]*4096+((int)page!=invalid_page);
    for (unsigned i=0;i<4;i++) ((unsigned char *)out)[i]=pte>>(8*i);
    return MEMTX_OK;
}
static MemoryRegion *address_space_translate(void *as, hwaddr a, hwaddr *xlat,
                                              hwaddr *n, bool wr, MemTxAttrs attr)
{
    static MemoryRegion mr;
    (void)as; assert(wr && attr.memory);
    mr.ram=a+*n<=sizeof(ram) && (int)(a/4096)!=nonram_page;
    mr.readonly=(int)(a/4096)==readonly_page;
    if ((int)(a/4096)==short_page) *n=1;
    *xlat=a; return &mr;
}
static int address_space_write(void *as, hwaddr a, MemTxAttrs attr, const void *src, unsigned n)
{
    (void)as; assert(attr.memory);
    assert((a & 4095) + n <= 4096);
    if ((int)writes++ == failed_write) return 1;
    memcpy(ram + a, src, n);
    return MEMTX_OK;
}
static bool r300_to_vram(PPCMacGPUState *s, uint32_t *a, uint64_t n)
{
    if (*a < 0xa0000000 || (uint64_t)*a - 0xa0000000 + n > sizeof(s->vram)) return false;
    *a -= 0xa0000000;
    return true;
}
static void r200_vram_access(PPCMacGPUState *s, uint64_t a, uint64_t b, bool wr, int why)
{ (void)s; (void)why; assert(!wr && a < b && b <= sizeof(s->vram)); syncs++; }
static const void *memory_region_get_ram_ptr(void *p) { return p; }
'''

checks = r'''
static R300DrawPacket packet(void)
{
    R300DrawPacket p = {0};
    p.copy_w = p.rt_width = p.rt_pitch = 8;
    p.copy_h = p.rt_height = 4;
    p.rt_gpu_addr = 0xcf000ff0;
    p.rt_bpp = 4; p.rt_view = R300_RTV_RGBA8;
    p.num_cb = p.aa_samples = 1;
    p.uniforms.chanmask = 15; p.uniforms.clip_rule = 0xffff;
    p.tex[0] = (R300TexDesc){ .bound=true, .card_bytes=true,
        .gpu_addr=0xa0000000, .width=8, .height=4, .tiling=2,
        .pitch_bytes=32, .lvl_pitch={32}, .size_bytes=128,
        .levels=1, .filter0=0xa92, .kind=R300_TEXK_RGBA8 };
    for (unsigned c=0; c<4; c++) {
        p.uniforms.tex_swz[0][c] = c;
        p.uniforms.out_sel[c] = (c+1)%4;
    }
    return p;
}

int main(void)
{
    PPCMacGPUState s = {0};
    s.regs.mc_agp_location=0xcf00cf00;
    R300DrawPacket p = packet();
    for (unsigned i=0; i<8; i++) maps[i] = 7-i;
    failed_page = invalid_page = failed_write = nonram_page = readonly_page = short_page = -1;
    /* Independent tile writer, all channels/coordinates nonuniform. */
    unsigned char linear[128];
    for (unsigned y=0; y<4; y++) for (unsigned x=0; x<8; x++) {
        unsigned tile = (y/2)*2+x/4;
        for (unsigned c=0; c<4; c++) {
            unsigned char v = (y*8+x)*5+c*37;
            linear[(y*8+x)*4+c] = v;
            s.vram[tile*32+(y%2)*16+(x%4)*4+c] = v;
        }
    }
    unsigned cases=0;
    for (unsigned e=0; e<4; e++) for (unsigned decode=0; decode<2; decode++)
    for (unsigned swap=0; swap<2; swap++) for (unsigned rot=0; rot<4; rot++)
    for (unsigned outrot=0; outrot<4; outrot++) {
        p.tex[0].endian=e; p.uniforms.tex_info[0][1]=decode;
        p.uniforms.rt_endian=swap*2;
        for (unsigned c=0; c<4; c++) {
            p.uniforms.tex_swz[0][c]=(c+rot)%4;
            p.uniforms.out_sel[c]=(c+1+outrot)%4;
        }
        unsigned char *out=r300_readback_pixels(&p, s.vram);
        assert(out);
        unsigned lane=e==1?1:e==2?3:e==3?2:0;
        for (unsigned i=0; i<32; i++) for (unsigned c=0; c<4; c++) {
            unsigned k=(c+outrot+rot)%4;
            unsigned from=(decode?3-k:k)^lane;
            assert(out[i*4+(swap?3-c:c)]==linear[i*4+from]);
        }
        free(out); cases++;
    }
    p=packet(); p.uniforms.tex_swz[0][0]=4; p.uniforms.tex_swz[0][1]=5;
    unsigned char *out=r300_readback_pixels(&p, s.vram);
    assert(out && out[0]==0 && out[1]==255); free(out);
    /* Offset crop and padding must not read unrelated texels. */
    p=packet(); p.copy_w=3; p.copy_h=2; p.copy_tx=3; p.copy_ty=1;
    out=r300_readback_pixels(&p, s.vram); assert(out);
    for (unsigned y=0; y<2; y++) assert(!memcmp(out+y*12,linear+((y+1)*8+3)*4,12));
    free(out);
    #define REJECT(change) do { p=packet(); change; assert(!r300_readback_pixels(&p,s.vram)); cases++; } while(0)
    REJECT(p.copy_w=0); REJECT(p.copy_h=0); REJECT(p.copy_tx=-1);
    REJECT(p.copy_w=32*MiB); REJECT(p.copy_h=32*MiB);
    REJECT(p.copy_ty=4); REJECT(p.copy_x=UINT32_MAX); REJECT(p.copy_y=UINT32_MAX);
    REJECT(p.tex[0].size_bytes=127); REJECT(p.tex[0].lvl_pitch[0]=28);
    REJECT(p.tex[0].tiling=1); REJECT(p.tex[0].tiling=3); REJECT(p.tex[0].levels=2);
    REJECT(p.tex[0].card_bytes=false); REJECT(p.tex[0].kind=R300_TEXK_RAW);
    REJECT(p.uniforms.tex_dim[0][3]=16); REJECT(p.uniforms.tex_dim[0][3]=1);
    REJECT(p.uniforms.cblend=1); REJECT(p.uniforms.cblend=8);
    REJECT(p.uniforms.fog_blend=1); REJECT(p.uniforms.rop=4);
    REJECT(p.uniforms.rt_endian=1); REJECT(p.uniforms.rt_endian=3);
    REJECT(p.depth.attach=true); REJECT(p.aa_samples=2); REJECT(p.num_cb=2);
    REJECT(p.warn=1); REJECT(p.tex[0].filter0=0x1412);
    REJECT(p.rt_tiling=2); REJECT(p.uniforms.chanmask=7);
    REJECT(p.uniforms.clip_rule=0); REJECT(p.uniforms.alpha_func=1<<11);
    REJECT(p.uniforms.zpass_count=1);
    /* Real source recognizer: two exact triangles, micro allowed, macro denied. */
    p=packet(); p.copy_w=p.copy_h=0;
    R300Vertex v[6]={0}; p.verts=v; p.num_verts=6;
    p.scissor[2]=8; p.scissor[3]=4;
    const unsigned xy[6][2]={{0,0},{8,0},{8,4},{0,0},{8,4},{0,4}};
    for (unsigned i=0;i<6;i++) {
        v[i].pos[0]=xy[i][0]/4.0f-1; v[i].pos[1]=1-xy[i][1]/2.0f;
        v[i].pos[3]=1; v[i].v[0][0]=xy[i][0]/8.0f; v[i].v[0][1]=xy[i][1]/4.0f;
    }
    R300FSDesc desc={0}; Route route={0}; route.vary[0].is_tex=true;
    copy_detect(&desc,&route,0,&p); assert(p.copy_w==8 && p.copy_h==4);
    p.copy_w=0; p.tex[0].tiling=1; copy_detect(&desc,&route,0,&p); assert(!p.copy_w);
    p.tex[0].tiling=2; v[2].v[0][0]=0.75f;
    copy_detect(&desc,&route,0,&p); assert(!p.copy_w);
    /* Actual shader gate accepts MOV(texture0), rejects an altered ALU. */
    R300State st; r300_state_reset(&st);
    r300_state_write(&st,0x4104,1); r300_state_write(&st,0x4e0c,15);
    r300_state_write(&st,0x43d0,0xffff); r300_state_write(&st,0x4600,8);
    r300_state_write(&st,0x4620,1<<15); r300_state_write(&st,0x46c0,7<<26);
    r300_state_write(&st,0x47c0,1<<24);
    r300_state_write(&st,0x48c0,(21<<7)|(20<<14));
    r300_state_write(&st,0x49c0,9|(17<<7)|(16<<14));
    unsigned coord; assert(copy_state(&st,&coord) && coord==0);
    r300_state_write(&st,0x48c0,0); assert(!copy_state(&st,&coord));
    /* Device path: source flush, page crossings, noncontiguous mapped pages. */
    p=packet(); memset(ram,0x5a,sizeof(ram)); writes=syncs=0;
    assert(r300_agp_copy(&s,&p) && writes==5 && syncs==1);
    assert(strict_pte_reads && !legacy_pte_reads);
    for(unsigned i=0;i<128;i++) {
        unsigned a=0xff0+i, physical=maps[a/4096]*4096+(a%4096);
        assert(ram[physical]==linear[i]);
    }
    assert(ram[maps[0]*4096+0xfef]==0x5a && ram[maps[1]*4096+0x70]==0x5a);
    /* Readback targets can have padded rows (the retained 4x4 target pitch is 32). */
    p.rt_pitch=p.rt_width=32; writes=0; memset(ram,0x5a,sizeof(ram));
    assert(r300_agp_copy(&s,&p));
    for(unsigned y=0;y<4;y++) for(unsigned x=0;x<128;x++) {
        unsigned a=0xff0+y*128+x, physical=maps[a/4096]*4096+(a%4096);
        assert(ram[physical]==(x<32?linear[y*32+x]:0x5a));
    }
    p=packet();
    failed_page=1; writes=0; memset(ram,0x5a,sizeof(ram));
    assert(!r300_agp_copy(&s,&p) && !writes);
    for(unsigned i=0;i<sizeof(ram);i++) assert(ram[i]==0x5a);
    failed_page=-1;
    nonram_page=maps[1]; assert(!r300_agp_copy(&s,&p) && !writes); nonram_page=-1;
    readonly_page=maps[1]; assert(!r300_agp_copy(&s,&p) && !writes); readonly_page=-1;
    short_page=maps[1]; assert(!r300_agp_copy(&s,&p) && !writes); short_page=-1;
    for(unsigned i=0;i<sizeof(ram);i++) assert(ram[i]==0x5a);
    failed_write=1; writes=0;
    assert(!r300_agp_copy(&s,&p) && writes==2);
    failed_write=-1; invalid_page=1; writes=0;
    assert(!r300_agp_copy(&s,&p) && !writes);
    hwaddr phys;
    assert(ppc_mac_gpu_agp_translate(&s,0xcf001000,&phys)); /* legacy policy unchanged */
    assert(legacy_pte_reads==1);
    invalid_page=-1; p.rt_gpu_addr=0xfffffff0;
    assert(!r300_agp_copy(&s,&p) && !writes);
    p=packet(); p.copy_y=UINT32_MAX; p.rt_pitch=UINT32_MAX;
    assert(!r300_agp_copy(&s,&p));
    assert(!r300_agp_span(&s,0xfffffff0,linear,32,false));
    printf("PASS: %u pixel-policy cases, actual shader/geometry recognition, source sync, page-split AGP writes, guards and failures\n",cases);
}
'''

helpers = "\n".join(function(name) for name in
                    ("ppc_mac_gpu_agp_translate_checked", "ppc_mac_gpu_agp_translate",
                     "r300_readback_pixels", "r300_agp_span", "r300_agp_copy"))
assert "if (!w || td->tiling ||" in device  # old CPU memcpy must stay linear-only
assert "if (r300_agp_copy(s, &pkt))" in device
with tempfile.TemporaryDirectory(prefix="r300-agp-copy-") as tmp:
    path = Path(tmp)
    (path / "check.c").write_text('#include "r300_draw.c"\n' + prefix + helpers + checks)
    r300 = root / "hw/display/r300"
    subprocess.run([os.environ.get("CC", "clang"), "-std=c11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
                    "-Wno-unused-parameter", "-Wno-sign-compare", "-O1", "-g",
                    "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                    "-I" + str(r300), str(path / "check.c"),
                    *[str(r300 / n) for n in ("r300_state.c", "r300_pvs.c", "r300_us.c")],
                    "-lm", "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)
