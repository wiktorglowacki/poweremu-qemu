/* Assemble the first Quartz Extreme draw captured from Tiger. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

static bool no_read(void *o, uint32_t a, void *d, uint32_t l) { return false; }
static uint8_t mem[192];
static bool mem_read(void *o, uint32_t a, void *d, uint32_t l)
{
    if (a < 0x1000 || a - 0x1000 + l > sizeof(mem)) return false;
    memcpy(d, mem + a - 0x1000, l);
    return true;
}

int main(void)
{
    static R300State st;
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

    r300_state_reset(&st);
    while (fgets(line, sizeof(line), f)) {
        unsigned a, v, i, d[4];
        float c[4];
        if (strstr(line, "PVS code")) { mode = 1; continue; }
        if (strstr(line, "PVS constants")) { mode = 2; continue; }
        if (mode == 0 && sscanf(line, " %x %x", &a, &v) == 2) r300_state_write(&st, a, v);
        if (mode == 1 && sscanf(line, " %u: %x %x %x %x", &i, &d[0], &d[1], &d[2], &d[3]) == 5)
            memcpy(&st.pvs_mem[i * 4], d, 16);
        if (mode == 2 && sscanf(line, " c%u %f %f %f %f", &i, &c[0], &c[1], &c[2], &c[3]) == 5)
            memcpy(&st.pvs_mem[(512 + i) * 4], c, 16);
    }
    /* First draw: VF_CNTL + 4 vertices x 12 dwords (pos, colour, texcoord). */
    uint32_t pkt3[49] = { 0x0004003D };
    float verts[4][12] = {
        { 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1 },
        { 256, 0, 0, 1, 1, 1, 1, 1, 256, 0, 0, 1 },
        { 256, 256, 0, 1, 1, 1, 1, 1, 256, 256, 0, 1 },
        { 0, 256, 0, 1, 1, 1, 1, 1, 0, 256, 0, 1 },
    };
    memcpy(&pkt3[1], verts, sizeof(verts));
    R300Arrays arr = { 0 };
    R300DrawPacket p;
    const char *err;
    bool ok = r300_draw_build(&st, &arr, 0x35, pkt3, 49, no_read, NULL, &p, &err);
    if (!ok) { printf("build failed: %s\n", err); return 1; }
    printf("rt %08x pitch %u %ux%u fmt %u scissor %u,%u-%u,%u warn %x verts %u cls %u\n",
           p.rt_gpu_addr, p.rt_pitch, p.rt_width, p.rt_height, p.rt_format,
           p.scissor[0], p.scissor[1], p.scissor[2], p.scissor[3], p.warn,
           p.num_verts, p.prim_class);
    for (unsigned i = 0; i < p.num_verts; i++) {
        R300Vertex *v = &p.verts[i];
        float wx = (v->pos[0] / v->pos[3] + 1) * p.rt_width / 2;
        float wy = (1 - v->pos[1] / v->pos[3]) * p.rt_height / 2;
        printf(" v%u win(%.1f,%.1f) col(%.2f %.2f %.2f %.2f) tc(%.3f %.3f %.3f %.3f)\n", i, wx, wy,
               v->v[0][0], v->v[0][1], v->v[0][2], v->v[0][3],
               v->v[1][0], v->v[1][1], v->v[1][2], v->v[1][3]);
    }
    printf("tex0 bound %d addr %08x %ux%u pitch %u swz %u%u%u%u swap %u out_sel %u%u%u%u rtswap %u\n",
           p.tex[0].bound, p.tex[0].gpu_addr, p.tex[0].width, p.tex[0].height,
           p.tex[0].pitch_bytes, p.uniforms.tex_swz[0][0], p.uniforms.tex_swz[0][1],
           p.uniforms.tex_swz[0][2], p.uniforms.tex_swz[0][3], p.uniforms.tex_info[0][1],
           p.uniforms.out_sel[0], p.uniforms.out_sel[1], p.uniforms.out_sel[2],
           p.uniforms.out_sel[3], p.uniforms.rt_swap32);
    r300_draw_free(&p);

    /* The same vertices from arrays: a host window must fetch what the
     * read callback does, for FLOAT_4 and for a D3DCOLOR element, with
     * the last vertex outside the window. */
    for (unsigned i = 0; i < 48; i++) {
        memcpy(&mem[4 * i], &pkt3[1 + i], 4);
        mem[4 * i] ^= mem[4 * i + 3];           /* bytes that differ */
    }
    for (int pass = 0; pass < 2; pass++) {
        uint32_t vbuf[1] = { 0x00040035 };
        R300DrawPacket q[2];
        if (pass) {
            r300_state_write(&st, 0x2150, 0x01050003);
            r300_state_write(&st, 0x21E0, 0xF688F688 ^ 0x00010000);
        }
        for (int w = 0; w < 2; w++) {
            for (unsigned k = 0; k < 3; k++) {
                arr.a[k].addr = 0x1000 + 16 * k;
                arr.a[k].size_dw = 4;
                arr.a[k].stride_dw = 12;
                arr.a[k].win = w ? mem + 16 * k : NULL;
                arr.a[k].win_lo = 0x1000 + 16 * k;
                arr.a[k].win_len = 3 * 48 - 16 * k;
            }
            arr.count = 3;
            ok = r300_draw_build(&st, &arr, 0x34, vbuf, 1, mem_read, NULL, &q[w], &err);
            if (!ok) { printf("array build failed: %s\n", err); return 1; }
        }
        assert(q[0].num_verts == q[1].num_verts && q[0].num_verts);
        assert(!memcmp(q[0].verts, q[1].verts, q[0].num_verts * sizeof(R300Vertex)));
        r300_draw_free(&q[0]);
        r300_draw_free(&q[1]);
    }
    printf("array fetch through a window: same vertices\n");
    return 0;
}
