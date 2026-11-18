/*
 * Draw assembly for the 3D features games need: depth/stencil state,
 * culling, and DRAW_INDX_2 with indices from memory (INDX_BUFFER).
 * Built on the first Quartz Extreme draw's state (qe_draw1_full.txt).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static R300State st;

static void load_state(void)
{
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
    fclose(f);
}

/* Guest memory for vertex arrays: big-endian dwords, as the PPC wrote them
 * (the captured state has VC_SWAP = 2). */
static uint8_t mem[0x10000];
#define MEM_BASE 0x08000000u

static bool rd(void *o, uint32_t a, void *d, uint32_t l)
{
    if (a < MEM_BASE || a - MEM_BASE + l > sizeof(mem)) return false;
    memcpy(d, mem + (a - MEM_BASE), l);
    return true;
}

static void put_be(uint32_t off, uint32_t v)
{
    mem[off] = v >> 24; mem[off + 1] = v >> 16; mem[off + 2] = v >> 8; mem[off + 3] = v;
}

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static const uint32_t immd[49] = { 0x0004003D };    /* 4 vertices, quads */

/* A DRAW_IMMD_2 packet of the captured QE state: n vertices (pos, colour,
 * texcoord) at window positions xy, colour k/n. */
static uint32_t *immd_pkt(unsigned prim, unsigned n, const float (*xy)[2], uint32_t *ndw)
{
    static uint32_t d[1 + 12 * 16];
    d[0] = (n << 16) | (3u << 4) | prim;
    for (unsigned i = 0; i < n; i++) {
        float v[12] = { xy[i][0], xy[i][1], 0, 1,  (float)i / n, 0.5f, 0.25f, 1,
                        xy[i][0], xy[i][1], 0, 1 };
        memcpy(&d[1 + 12 * i], v, sizeof(v));
    }
    *ndw = 1 + 12 * n;
    return d;
}

/* Window position of a packet vertex (y down). */
static void win(const R300DrawPacket *p, const R300Vertex *v, float *x, float *y)
{
    *x = (v->pos[0] / v->pos[3] + 1) * p->rt_width / 2;
    *y = (1 - v->pos[1] / v->pos[3]) * p->rt_height / 2;
}

#define FEQ(a, b) (fabsf((a) - (b)) < 1e-3f)

static void rendering_completeness(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err;
    uint32_t n;
    uint32_t *d;
    static const float quad4[4][2] = { { 0, 0 }, { 256, 0 }, { 256, 256 }, { 0, 256 } };

    /* ---- texture layout (Mesa r300_texture_desc.c rules) ---- */
    R300TexDesc t;
    memset(&t, 0, sizeof(t));
    t.width = t.height = 64; t.levels = 7; t.dim = R300_TEXDIM_2D; t.depth = 1;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.lvl_off[1] == 16384 && t.lvl_pitch[1] == 128 && t.lvl_pitch[5] == 32 &&
          t.lvl_rows[5] == 2 && t.lvl_off[6] == 16384 + 4096 + 1024 + 256 + 128 + 64 &&
          t.size_bytes == t.lvl_off[6] + 32, "64x64 chain: off1 %u pitch5 %u size %u",
          t.lvl_off[1], t.lvl_pitch[5], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 16; t.levels = 1; t.dim = R300_TEXDIM_CUBE; t.depth = 1;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.size_bytes == 6 * 64 * 16, "cube size %u", t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 8; t.depth = 4; t.levels = 2; t.dim = R300_TEXDIM_3D;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.lvl_off[1] == 32 * 8 * 4 && t.size_bytes == 1024 + 32 * 4 * 2, "3D %u %u",
          t.lvl_off[1], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = 6; t.height = 5; t.levels = 3; t.depth = 1;
    r300_tex_layout(&t, 1, false, false);           /* NPOT mipmapped X8: POT rows */
    CHECK(t.lvl_rows[0] == 8 && t.lvl_pitch[0] == 32 && t.lvl_off[1] == 256 &&
          t.lvl_off[2] == 256 + 64 && t.size_bytes == 352, "npot %u %u %u", t.lvl_rows[0],
          t.lvl_off[2], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 16; t.levels = 3; t.depth = 1;
    r300_tex_layout(&t, 8, true, false);            /* DXT1 */
    CHECK(t.lvl_pitch[0] == 32 && t.lvl_rows[0] == 4 && t.lvl_off[1] == 128 &&
          t.lvl_off[2] == 192 && t.size_bytes == 224, "dxt1 %u %u %u", t.lvl_off[1],
          t.lvl_off[2], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = 100; t.height = 10; t.levels = 1; t.depth = 1; t.pitch_bytes = 512;
    r300_tex_layout(&t, 4, false, true);            /* TX_PITCH_EN keeps its pitch */
    CHECK(t.lvl_pitch[0] == 512 && t.size_bytes == 5120, "pitch_en %u", t.lvl_pitch[0]);

    R300State tex_state = st;
    tex_state.card_bytes = true;
    r300_state_write(&tex_state, 0x4104, 1);
    r300_state_write(&tex_state, 0x4540, 0x00100008);      /* micro tiled */
    r300_state_write(&tex_state, 0x4480, 4 | (1u << 11) | (1u << 31));
    r300_state_write(&tex_state, 0x44C0, 0x0C);          /* RGBA8, 5x2 */
    r300_state_write(&tex_state, 0x4500, 4);             /* explicit 20-byte pitch */
    d = immd_pkt(13, 4, quad4, &n);
    CHECK(r300_draw_build(&tex_state, &none, 0x35, d, n, rd, NULL, &p, &err),
          "bad-pitch draw: %s", err ? err : "");
    CHECK(!p.tex[0].bound && !p.uniforms.tex_info[0][0] && (p.warn & R300_WARN_TEXFMT),
          "invalid tiled pitch reached renderer");
    r300_draw_free(&p);
    r300_state_write(&tex_state, 0x4500, 7);             /* full 32-byte tile row */
    CHECK(r300_draw_build(&tex_state, &none, 0x35, d, n, rd, NULL, &p, &err),
          "aligned-pitch draw: %s", err ? err : "");
    CHECK(p.tex[0].bound && p.tex[0].lvl_pitch[0] == 32 && p.tex[0].size_bytes == 64,
          "valid tiled pitch rejected or changed");
    r300_draw_free(&p);

    /* ---- border colours: the texel's own packing ---- */
    float bc[4];
    r300_border_color(0x0C, 0x80FF0040, bc);
    CHECK(FEQ(bc[0], 0x40 / 255.0f) && bc[1] == 0 && bc[2] == 1 && FEQ(bc[3], 0x80 / 255.0f),
          "8888 border %g %g %g %g", bc[0], bc[1], bc[2], bc[3]);
    r300_border_color(0x06, 0xF800, bc);
    CHECK(bc[0] == 0 && bc[1] == 0 && bc[2] == 1, "565 border %g %g %g", bc[0], bc[1], bc[2]);
    r300_border_color(0x0F, 0x00FF0000, bc);        /* DXT: B8G8R8A8 */
    CHECK(bc[0] == 1 && bc[2] == 0, "dxt border %g %g", bc[0], bc[2]);

    /* ---- texture unit state: mip chain, LOD, 3D, cube, sign, gamma ---- */
    r300_state_write(&st, 0x4104, 1);
    r300_state_write(&st, 0x4540, 0x00100000);
    r300_state_write(&st, 0x4480, 63 | (63u << 11) | (6u << 26));          /* 64x64, 7 levels */
    r300_state_write(&st, 0x44C0, 0x0C | (1u << 8) | (1u << 21));          /* 8888, X signed, gamma */
    r300_state_write(&st, 0x4400, (2u << 9) | (2u << 11) | (2u << 13) | (1u << 17));
    r300_state_write(&st, 0x4440, (uint32_t)(-16 & 0x3FF) << 3);           /* LOD bias -0.5 */
    d = immd_pkt(13, 4, quad4, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].levels == 7 && p.tex[0].size_bytes == 16384 + 4096 + 1024 + 256 + 128 + 64 + 32,
          "levels %u size %u", p.tex[0].levels, p.tex[0].size_bytes);
    CHECK(FEQ(p.uniforms.tex_lod[0][0], -0.5f) && p.uniforms.tex_lod[0][1] == 1 &&
          p.uniforms.tex_lod[0][2] == 6 && p.uniforms.tex_lod[0][3] == 2,
          "lod %g %g %g %g", p.uniforms.tex_lod[0][0], p.uniforms.tex_lod[0][1],
          p.uniforms.tex_lod[0][2], p.uniforms.tex_lod[0][3]);
    CHECK(p.uniforms.tex_dim[0][3] == (R300_TEXF_SIGNED_X | R300_TEXF_GAMMA | R300_TEXF_POT_ROWS),
          "tex flags %x", p.uniforms.tex_dim[0][3]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4480, 7 | (7u << 11) | (2u << 22));            /* 8x8x4 */
    r300_state_write(&st, 0x44C0, 0x0C | (1u << 25));                      /* 3D */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].dim == R300_TEXDIM_3D && p.tex[0].depth == 4 && p.tex[0].size_bytes == 1024,
          "3D unit dim %u depth %u size %u", p.tex[0].dim, p.tex[0].depth, p.tex[0].size_bytes);
    r300_draw_free(&p);
    r300_state_write(&st, 0x44C0, 0x0C | (2u << 25));                      /* cube */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].dim == R300_TEXDIM_CUBE && p.tex[0].size_bytes == 6 * 32 * 8,
          "cube unit %u", p.tex[0].size_bytes);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4104, 0);
    r300_state_write(&st, 0x4440, 0);

    /* ---- polygon mode: quad edges only (no diagonal), points ---- */
    r300_state_write(&st, 0x4288, 1 | (1u << 4) | (1u << 7));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 1 && p.num_verts == 8, "poly lines cls %u verts %u",
          p.prim_class, p.num_verts);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4288, 1 | (0u << 4) | (0u << 7));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 0 && p.num_verts == 6 * 6, "poly points %u", p.num_verts);
    r300_draw_free(&p);
    /* back faces culled on the CPU when polygon mode is on (the quad is
     * clockwise on screen: a back face with FACE_CW clear) */
    r300_state_write(&st, 0x4288, 1 | (1u << 4) | (1u << 7));
    r300_state_write(&st, 0x42B8, R300_CULL_BACK);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_verts == 0 && p.num_line_verts == 0, "poly cull %u", p.num_verts);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);
    r300_state_write(&st, 0x4288, 0);

    /* ---- flat shading, provoking vertex LAST: the quad's vertex 3 ---- */
    r300_state_write(&st, 0x4278, 0x5555 | (3u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    bool flat = p.num_verts == 6;
    for (unsigned i = 0; i < p.num_verts; i++) {
        flat &= FEQ(p.verts[i].v[1][0], 0.75f);
    }
    CHECK(flat, "flat shading");
    r300_draw_free(&p);
    /* FIRST: vertex 0 of the quad */
    r300_state_write(&st, 0x4278, 0x5555);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_verts == 6 && FEQ(p.verts[5].v[1][0], 0.0f), "flat first %g", p.verts[5].v[1][0]);
    r300_draw_free(&p);
    /* solid: GA_SOLID_RG/BA in S3.12 */
    r300_state_write(&st, 0x4278, 0);
    r300_state_write(&st, 0x427C, (0x0800u << 16) | 0x1000);
    r300_state_write(&st, 0x4280, (0x0000u << 16) | 0x1000);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].v[1][0], 0.5f) && FEQ(p.verts[0].v[1][1], 1.0f) &&
          FEQ(p.verts[0].v[1][2], 0.0f), "solid %g %g %g", p.verts[0].v[1][0],
          p.verts[0].v[1][1], p.verts[0].v[1][2]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4278, 0x3AAAA);

    /* ---- two-sided colours: colour 2 present, the quad is a back face:
     * colour 0 comes from output slot 2 (the texcoord output here) ---- */
    r300_state_write(&st, 0x2090, 3 | (1u << 3));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x, y;     /* the texcoord output is window position / 256 */
        win(&p, &p.verts[1], &x, &y);
        CHECK(FEQ(p.verts[1].v[1][0], x / 256) && FEQ(p.verts[1].v[1][1], y / 256),
              "back colour %g %g at %g %g", p.verts[1].v[1][0], p.verts[1].v[1][1], x, y);
    }
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, R300_FACE_CW);     /* now it is front-facing */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[1].v[1][1], 0.5f), "front colour %g", p.verts[1].v[1][1]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);
    r300_state_write(&st, 0x2090, 3);

    /* ---- wide lines: 3 px (half width 18/12), x-major: widened in y ---- */
    static const float seg[2][2] = { { 10, 20 }, { 110, 20 } };
    r300_state_write(&st, 0x4234, 18 | (2u << 16));
    d = immd_pkt(2, 2, seg, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x0, y0, x1, y1;
        CHECK(p.prim_class == 0 && p.num_verts == 6, "wide line %u %u", p.prim_class, p.num_verts);
        win(&p, &p.verts[0], &x0, &y0);
        win(&p, &p.verts[1], &x1, &y1);
        CHECK(FEQ(x0, 10) && FEQ(y0, 18.5f) && FEQ(y1, 21.5f), "wide corners %g,%g %g,%g",
              x0, y0, x1, y1);
    }
    r300_draw_free(&p);
    /* stipple coordinates: s = accumulated subpixels * STIPPLE_SCALE past
     * the vertex's own texture components (RS reads one past the texcoord) */
    r300_state_write(&st, 0x4234, 6 | (2u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 1 && p.num_verts == 2, "thin line %u", p.num_verts);
    r300_draw_free(&p);

    /* ---- rectangle list: the card completes the parallelogram ---- */
    static const float rect3[3][2] = { { 0, 0 }, { 100, 0 }, { 100, 50 } };
    d = immd_pkt(8, 3, rect3, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x, y;
        CHECK(p.num_verts == 6, "rect verts %u", p.num_verts);
        win(&p, &p.verts[5], &x, &y);
        CHECK(FEQ(x, 0) && FEQ(y, 50), "rect 4th corner %g,%g", x, y);
    }
    r300_draw_free(&p);

    /* ---- points: per-vertex size (output slot 1 .x = colour r) clamped
     * to GA_POINT_MINMAX (4..8 px) ---- */
    static const float pt[1][2] = { { 50, 60 } };
    d = immd_pkt(1, 1, pt, &n);
    r300_state_write(&st, 0x2090, 3 | (1u << 16));
    r300_state_write(&st, 0x4230, 24 | (48u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x0, y0, x1, y1;
        win(&p, &p.verts[0], &x0, &y0);
        win(&p, &p.verts[2], &x1, &y1);
        CHECK(p.num_verts == 6 && FEQ(x1 - x0, 4) && FEQ(y1 - y0, 4), "point size %g x %g",
              x1 - x0, y1 - y0);
    }
    r300_draw_free(&p);
    r300_state_write(&st, 0x2090, 3);

    /* ---- user clip planes: distance to plane 0 = clip-space x ---- */
    d = immd_pkt(13, 4, quad4, &n);
    float plane[4] = { 1, 0, 0, 0 };
    memcpy(&st.pvs_mem[1024 * 4], plane, sizeof(plane));
    r300_state_write(&st, 0x221C, 1);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].ucp[0], -1.0f) && FEQ(p.verts[1].ucp[0], -0.36f) &&
          p.verts[0].ucp[1] == 1.0f, "ucp %g %g %g", p.verts[0].ucp[0], p.verts[1].ucp[0],
          p.verts[0].ucp[1]);
    CHECK(p.glsl && strstr(p.glsl, "gl_ClipDistance[6]"), "vertex shader lacks clip distances");
    r300_draw_free(&p);
    r300_state_write(&st, 0x221C, 0);

    /* ---- fog value: C0A (1.0) * GA_FOG_SCALE + GA_FOG_OFFSET ---- */
    r300_state_write(&st, 0x4BC0, 1);
    r300_state_write(&st, 0x401C, 0);
    r300_state_write(&st, 0x4294, fbits(2.0f));
    r300_state_write(&st, 0x4298, fbits(0.5f));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].aux[0], 2.5f) && p.uniforms.fog_blend == 1, "fog %g",
          p.verts[0].aux[0]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4BC0, 0);

    /* ---- render targets: RB3D_CCTL multiwrites ---- */
    CHECK(r300_us_num_targets(&st) == 1, "targets %u", r300_us_num_targets(&st));
    r300_state_write(&st, 0x4E00, 2u << 5);
    r300_state_write(&st, 0x4E2C, 0x00200000);
    r300_state_write(&st, 0x4E30, 0x00300000);
    r300_state_write(&st, 0x4E3C, r300_reg(&st, 0x4E38));
    r300_state_write(&st, 0x4E40, r300_reg(&st, 0x4E38));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_cb == 3 && p.cb[2].gpu_addr == 0x00300000 && p.cb[1].view == p.rt_view,
          "multiwrite cb %u %08x", p.num_cb, p.cb[2].gpu_addr);
    CHECK(p.glsl && strstr(p.glsl, "location = 2) out r300_cb2_o o_c2") &&
          strstr(p.glsl, "location = 3) out uvec4 o_z"),
          "MRT outputs missing");
    r300_draw_free(&p);
    r300_state_write(&st, 0x4E00, 0);
}

static void upload_ports(void)
{
    R300State state;
    const uint32_t words[4] = { 0x00100201, 0x00D10002, 0x00D10001, 0x00D10005 };
    r300_state_reset(&state);
    r300_state_write(&state, 0x2200, 3);
    for (int i = 0; i < 4; i++) {
        r300_state_write(&state, i < 2 ? 0x2204 : 0x2208, words[i]);
    }
    CHECK(!memcmp(&state.pvs_mem[12], words, sizeof(words)) && state.pvs_upload_dw == 16,
          "both PVS upload ports must share the vector index");
    CHECK(state.pvs_gen == 4, "PVS upload must invalidate cached shaders");
    r300_state_write(&state, 0x2200, 2047);
    r300_state_write(&state, 0x2204, 0xdeadbeef);
    CHECK(!memcmp(&state.pvs_mem[12], words, sizeof(words)), "PVS upload bounds");
}

static void draw_bounds(void)
{
    R300Vertex v[3] = { { .pos = { -0.5f, 0.5f, 0, 1 } },
                        { .pos = { 0.5f, 0.5f, 0, 1 } },
                        { .pos = { 0, -0.5f, 0, 1 } } };
    R300DrawPacket p = { .rt_width = 100, .rt_height = 100,
                        .scissor = { 0, 0, 100, 100 }, .verts = v, .num_verts = 3 };
    uint32_t b[4];
    r300_draw_bounds(&p, b);
    CHECK(b[0] == 24 && b[1] == 24 && b[2] == 76 && b[3] == 76,
          "triangle bounds %u,%u-%u,%u", b[0], b[1], b[2], b[3]);
    p.scissor[0] = 30; p.scissor[3] = 50;
    r300_draw_bounds(&p, b);
    CHECK(b[0] == 30 && b[3] == 50, "bounds escaped scissor");
    v[0].pos[3] = -1;
    r300_draw_bounds(&p, b);
    CHECK(b[0] == 30 && b[1] == 0 && b[2] == 100 && b[3] == 50,
          "eye-plane crossing must retain hardware scissor");
    v[0].pos[3] = 1; p.vs_glsl = "gpu shader";
    r300_draw_bounds(&p, b);
    CHECK(b[1] == 0 && b[2] == 100, "GPU vertex positions cannot be bounded on CPU");
}

static void bypass_semantics(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err = NULL;
    uint32_t n;
    const float xy[4][2] = { { 0, 0 }, { 100, 0 }, { 100, 100 }, { 0, 100 } };
    uint32_t *d = immd_pkt(13, 4, xy, &n);

    load_state();
    r300_state_write(&st, 0x2140, 0x102);  /* PVS bypass */
    r300_state_write(&st, 0x20D0, 0);      /* window coordinates */
    r300_state_write(&st, 0x2150, 0x02030003); /* position 0, colour 2 */
    r300_state_write(&st, 0x2154, 0x00002603); /* texture 6, last */
    CHECK(r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err),
          "bypass: %s", err ? err : "");
    if (p.verts && p.num_verts >= 3) {
        CHECK(FEQ(p.verts[0].v[1][1], 0.5f) && FEQ(p.verts[0].v[1][2], 0.25f),
              "bypass colour must come from input 2");
        CHECK(FEQ(p.verts[1].v[0][0], 100) && FEQ(p.verts[2].v[0][1], 100),
              "bypass texture must come from input 6");
    }
    r300_draw_free(&p);
}

static void viewport_ports(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err = NULL;
    const float xy[4][2] = { { 0, 0 }, { 100, 0 }, { 100, 100 }, { 0, 100 } };
    const float vp[6] = { 64, 64, -32, 32, 0.5f, 0.25f };
    uint32_t n, *d = immd_pkt(13, 4, xy, &n);

    load_state();
    r300_state_write(&st, 0x2140, 0x102);
    r300_state_write(&st, 0x20B0, 0x13F);
    for (unsigned i = 0; i < 6; i++) {
        uint32_t bits;
        memcpy(&bits, &vp[i], sizeof(bits));
        r300_state_write(&st, 0x2098 + i * 4, bits);
        CHECK(r300_reg(&st, 0x1D98 + i * 4) == bits, "native viewport port %u", i);
    }
    CHECK(r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err),
          "viewport: %s", err ? err : "");
    if (p.verts && p.num_verts) {
        CHECK(FEQ(p.verts[0].pos[0], 128.0f / p.rt_width - 1.0f) &&
              FEQ(p.verts[0].pos[1], 1.0f - 64.0f / p.rt_height),
              "native viewport transform ignored");
    }
    r300_draw_free(&p);
    r300_state_write(&st, 0x1D98, 0x3F800000);
    CHECK(r300_reg(&st, 0x2098) == 0x3F800000, "legacy viewport port");
}

static void native_texture_bytes(void)
{
    const uint32_t swapped[4] = { 0x01234567, 0x23016745, 0x67452301, 0x45670123 };
    for (unsigned mode = 0; mode < 8; mode++) {
        CHECK(r300_swap_mode(0x01234567, mode) == swapped[mode & 3],
              "dword swap mode %u", mode);
        CHECK(r300_swap_mode(r300_swap_mode(0x89abcdef, mode), mode) == 0x89abcdef,
              "dword swap involution %u", mode);
    }
    R300TexDesc t = { .card_bytes = true, .kind = R300_TEXK_RGBA8,
        .width = 1, .height = 1, .levels = 1, .lvl_pitch = { 4 },
        .lvl_rows = { 1 }, .size_bytes = 4 };
    /* Captured Gears red (.875878,.109485,0,1), US_OUT_FMT=1b00. */
    const uint8_t bgra[4] = { 0, 28, 223, 255 };
    uint32_t bpr;
    for (unsigned mode = 0; mode < 4; mode++) {
        const unsigned lanes[4] = { 0, 1, 3, 2 };
        uint8_t card[4];
        for (unsigned i = 0; i < 4; i++) card[i] = bgra[i ^ lanes[mode]];
        t.endian = mode;
        uint8_t *tex = r300_tex_level_bytes(&t, card, 0, 1, 1, &bpr);
        CHECK(tex && bpr == 4 && !memcmp(tex, bgra, 4), "card texture endian %u", mode);
        if (tex) {
            CHECK(tex[2] == 223 && tex[1] == 28 && tex[0] == 0 && tex[3] == 255,
                  "Gears red texture/scanout roundtrip %u", mode);
        }
        free(tex);
    }
    t.kind = R300_TEXK_CONVERT16; t.format = 6; t.endian = 0;
    const uint8_t red565[4] = { 0, 0xf8, 0, 0 };
    uint8_t *tex = r300_tex_level_bytes(&t, red565, 0, 1, 1, &bpr);
    CHECK(tex && tex[0] == 0 && tex[1] == 0 && tex[2] == 255, "native 565 texel");
    free(tex);
    t.kind = R300_TEXK_RG8; t.endian = 2; t.lvl_pitch[0] = 2;
    CHECK(!r300_tex_card_swap_valid(&t), "partial swap DWORD accepted");
    CHECK(!r300_tex_level_bytes(&t, red565, 0, 1, 1, &bpr), "partial swap texture read accepted");

    /* DXT's implicit CPU-view swap precedes the explicit TX_OFFSET swap. */
    uint8_t blocks[32];
    for (unsigned i = 0; i < sizeof(blocks); i++) blocks[i] = i * 17 + 3;
    t = (R300TexDesc){ .kind = R300_TEXK_DXT1, .width = 4, .height = 4,
        .levels = 1, .lvl_pitch = { 32 }, .lvl_rows = { 1 }, .size_bytes = 32 };
    for (unsigned card = 0; card < 2; card++) {
        for (unsigned host = 0; host < 2; host++) {
            for (unsigned mode = 0; mode < 4; mode++) {
                static const unsigned lanes[4] = { 0, 1, 3, 2 };
                t.card_bytes = card;
                t.host_data = host ? blocks : NULL;
                t.endian = mode;
                tex = r300_tex_level_bytes(&t, blocks, 0, 4, 4, &bpr);
                CHECK(tex && bpr == 8, "DXT source view %u/%u/%u", card, host, mode);
                for (unsigned i = 0; tex && i < 8; i++) {
                    unsigned lane = lanes[mode] ^ (!card && !host ? 3 : 0);
                    CHECK(tex[i] == blocks[i ^ lane], "DXT swap %u/%u/%u byte %u",
                          card, host, mode, i);
                }
                free(tex);
            }
        }
    }
}

static void texture_layout_bounds(void)
{
    static const struct {
        uint32_t kind, bpp, w, h, bad_pitch, tw, th;
        bool dxt;
    } cases[] = {
        { R300_TEXK_R8, 1, 9, 5, 9, 8, 4, false },
        { R300_TEXK_RG8, 2, 9, 3, 18, 8, 2, false },
        { R300_TEXK_RGBA8, 4, 5, 3, 20, 4, 2, false },
        { R300_TEXK_CONVERT16, 2, 9, 3, 18, 8, 2, false },
        { R300_TEXK_DXT1, 8, 12, 12, 24, 2, 2, true },
        { R300_TEXK_DXT3, 16, 12, 12, 16, 2, 1, true },
        { R300_TEXK_DXT5, 16, 12, 12, 16, 2, 1, true },
    };
    const uint8_t dummy = 0;
    uint32_t bpr;
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        R300TexDesc t = { .kind = cases[i].kind, .format = 6, .tiling = 2,
            .width = cases[i].w, .height = cases[i].h, .depth = 1,
            .dim = R300_TEXDIM_2D, .levels = 3, .card_bytes = true };
        r300_tex_layout(&t, cases[i].bpp, cases[i].dxt, false);
        CHECK(r300_tex_layout_valid(&t), "valid tiled layout kind %u", t.kind);
        uint8_t *src = calloc(t.size_bytes, 1);
        for (unsigned j = 0; j < t.size_bytes; j++) src[j] = j * 37 + j / 17 + 5;
        for (uint32_t l = 0; l < t.levels; l++) {
            uint32_t w, h, d;
            r300_tex_level_dims(&t, l, &w, &h, &d);
            uint32_t pitch = t.lvl_pitch[l], rows = t.lvl_rows[l];
            uint32_t tw = cases[i].tw, th = cases[i].th, bpp = cases[i].bpp;
            uint8_t *linear = calloc(pitch * rows, 1);
            /* Independent tile-order traversal, including whole padded bands. */
            for (unsigned ty = 0; ty < rows / th; ty++) {
                for (unsigned tx = 0; tx < pitch / (tw * bpp); tx++) {
                    unsigned tile = ty * (pitch / (tw * bpp)) + tx;
                    for (unsigned y = 0; y < th; y++) {
                        memcpy(linear + (ty * th + y) * pitch + tx * tw * bpp,
                               src + t.lvl_off[l] + tile * 32 + y * tw * bpp,
                               tw * bpp);
                    }
                }
            }
            R300TexDesc linear_t = t;
            linear_t.tiling = 0;
            uint32_t ref_bpr;
            uint8_t *ref = r300_tex_level_bytes(&linear_t, linear, l, w, h, &ref_bpr);
            for (unsigned mode = 2; mode <= 3; mode++) {
                R300TexDesc tiled_t = t;
                tiled_t.tiling = mode;
                uint8_t *tex = r300_tex_level_bytes(&tiled_t, src + t.lvl_off[l], l, w, h, &bpr);
                CHECK(tex && ref && bpr == ref_bpr &&
                      !memcmp(tex, ref, bpr * (cases[i].dxt ? (h + 3) / 4 : h)),
                      "tiled bytes kind %u level %u mode %u", t.kind, l, mode);
                free(tex);
            }
            free(ref);
            free(linear);
        }
        free(src);
        t.size_bytes--;
        bpr = 1;
        CHECK(!r300_tex_layout_valid(&t) &&
              !r300_tex_level_bytes(&t, &dummy, 0, t.width, t.height, &bpr) && !bpr,
              "truncated chain accepted for kind %u", t.kind);

        t.levels = 1;
        t.pitch_bytes = cases[i].bad_pitch;
        r300_tex_layout(&t, cases[i].bpp, cases[i].dxt, true);
        bpr = 1;
        CHECK(t.lvl_pitch[0] == cases[i].bad_pitch && !r300_tex_layout_valid(&t) &&
              !r300_tex_level_bytes(&t, &dummy, 0, t.width, t.height, &bpr) && !bpr,
              "partial-tile pitch accepted or changed for kind %u", t.kind);
    }

    /* RAW keeps its existing upload path; layout still uses 8/16-byte texels. */
    for (unsigned bpp = 8; bpp <= 16; bpp *= 2) {
        R300TexDesc raw = { .kind = R300_TEXK_RAW, .format = bpp == 8 ? 0x1c : 0x1d,
            .view_bpp = bpp, .tiling = 2, .width = 5, .height = 3, .levels = 1 };
        r300_tex_layout(&raw, bpp, false, false);
        CHECK(raw.lvl_pitch[0] == 6 * bpp && raw.lvl_rows[0] == (bpp == 8 ? 4 : 3) &&
              r300_tex_layout_valid(&raw), "RAW microtile layout bpp %u", bpp);
    }

    R300TexDesc t = { .kind = R300_TEXK_DXT1, .tiling = 2, .width = 16,
        .height = 4, .depth = 1, .levels = 1, .dim = R300_TEXDIM_2D };
    r300_tex_layout(&t, 8, true, false);
    CHECK(t.size_bytes == 64 && r300_tex_layout_valid(&t), "DXT1 tile-band padding");
    uint8_t *src = calloc(t.size_bytes, 1);
    uint8_t *tex = r300_tex_level_bytes(&t, src, 0, 16, 4, &bpr);
    CHECK(tex && bpr == 32, "DXT1 16x4 source span");
    free(tex);
    free(src);

    t = (R300TexDesc){ .kind = R300_TEXK_RGBA8, .width = 5, .height = 2,
        .depth = 1, .levels = 1, .pitch_bytes = 16, .dim = R300_TEXDIM_2D };
    r300_tex_layout(&t, 4, false, true);
    CHECK(!r300_tex_layout_valid(&t) &&
          !r300_tex_level_bytes(&t, &dummy, 0, 5, 2, &bpr), "short linear pitch accepted");
    t.pitch_bytes = 32;
    r300_tex_layout(&t, 4, false, true);
    CHECK(r300_tex_layout_valid(&t), "valid explicit linear pitch");
    CHECK(!r300_tex_level_bytes(&t, &dummy, 1, 5, 2, &bpr) && !bpr,
          "invalid mip index accepted");
    CHECK(!r300_tex_level_bytes(&t, &dummy, 0, UINT32_MAX, 2, &bpr) && !bpr,
          "requested width outside its rows accepted");
    t.lvl_rows[0] = 1;
    CHECK(!r300_tex_layout_valid(&t), "missing source rows accepted");
    t.levels = 2; t.width = t.height = 8; t.pitch_bytes = 0;
    r300_tex_layout(&t, 4, false, false);
    t.lvl_off[1] = 0;
    CHECK(!r300_tex_layout_valid(&t), "overlapping mip levels accepted");
    t.dim = R300_TEXDIM_CUBE; t.levels = 1;
    r300_tex_layout(&t, 4, false, false);
    CHECK(r300_tex_layout_valid(&t), "valid cube layout");
    t.size_bytes -= t.lvl_pitch[0] * t.lvl_rows[0];
    CHECK(!r300_tex_layout_valid(&t), "truncated cube face accepted");
    t = (R300TexDesc){ .kind = R300_TEXK_RGBA8, .tiling = 1, .width = 5,
        .height = 2, .depth = 1, .levels = 1, .pitch_bytes = 256 };
    r300_tex_layout(&t, 4, false, true);
    bool saved_macro = r300_macro_layout;
    r300_macro_layout = true;
    CHECK(!r300_tex_level_bytes(&t, &dummy, 0, 5, 2, &bpr) && !bpr,
          "partial macro tile band accepted");
    t.height = 8;
    r300_tex_layout(&t, 4, false, true);
    src = calloc(t.size_bytes, 1);
    tex = r300_tex_level_bytes(&t, src, 0, 5, 8, &bpr);
    CHECK(tex && bpr == 20, "valid macro tile band rejected");
    free(tex);
    free(src);
    r300_macro_layout = saved_macro;
}

static void morphos_color_profile(void)
{
    load_state();
    for (unsigned k = 0; k < 4; k++) {
        for (unsigned e = 0; e < 4; e++) {
            r300_state_write(&st, 0x4E38 + 4 * k, (6u << 21) | (e << 19) | 64);
            r300_state_write(&st, 0x46A4 + 4 * k, 0x1B00);
            st.card_bytes = true;
            st.morphos_color_compat = false;
            CHECK(r300_color_endian(&st, k) == e, "default RT%u endian%u", k, e);
            st.morphos_color_compat = true;
            st.card_bytes = false;
            CHECK(r300_color_endian(&st, k) == e, "legacy RT%u endian%u", k, e);
            st.card_bytes = true;
            CHECK(r300_color_endian(&st, k) == (e == 2 ? 0 : e),
                  "MorphOS RGBA8 RT%u endian%u", k, e);
            r300_state_write(&st, 0x46A4 + 4 * k, 0x1B13); /* R32U, not RGBA8. */
            CHECK(r300_color_endian(&st, k) == e, "typed RT%u endian%u", k, e);
            r300_state_write(&st, 0x4E38 + 4 * k, (4u << 21) | (e << 19) | 64);
            CHECK(r300_color_endian(&st, k) == e, "RGB565 RT%u endian%u", k, e);
        }
    }

    load_state();
    r300_state_write(&st, 0x4E00, r300_reg(&st, 0x4E00) | (3u << 5));
    for (unsigned k = 0; k < 4; k++) {
        r300_state_write(&st, 0x4E38 + 4 * k, (6u << 21) | (2u << 19) | 64);
        r300_state_write(&st, 0x46A4 + 4 * k, 0x1B00);
    }
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *build_err = NULL;
    st.card_bytes = true;
    for (unsigned enabled = 0; enabled < 2; enabled++) {
        st.morphos_color_compat = enabled;
        CHECK(r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &build_err),
              "profile packet: %s", build_err ? build_err : "");
        CHECK(p.uniforms.rt_endian == (enabled ? 0 : 2) &&
              p.uniforms.rt_swap32 == !enabled, "RT0 uniforms did not use the profile");
        r300_draw_free(&p);
    }
    R300FSDesc desc = { 0 };
    const char *err = NULL;
    for (unsigned flags = 0; flags < 2; flags++) {
        uint32_t original_id, legacy_id, compat_id, restored_id;
        st.card_bytes = true;
        st.morphos_color_compat = false;
        char *original = r300_us_glsl_cached(&st, &desc, flags, &original_id, NULL, &err);
        CHECK(original && original_id, "default MRT shader: %s", err ? err : "");
        st.morphos_color_compat = true;
        st.card_bytes = false;
        char *legacy = r300_us_glsl_cached(&st, &desc, flags, &legacy_id, NULL, &err);
        CHECK(legacy && legacy_id == original_id, "legacy handoff changed default cache");
        st.card_bytes = true;
        char *compat = r300_us_glsl_cached(&st, &desc, flags, &compat_id, NULL, &err);
        CHECK(compat && compat_id && compat_id != original_id,
              "native compatibility reused the default MRT shader id");
        if (original && legacy && compat) {
            CHECK(!strcmp(original, legacy) && strcmp(original, compat),
                  "MRT shader text did not separate native compatibility");
            CHECK(strstr(original, "r300_unpack(fb, uvec4(3u, 2u, 1u, 0u), 1u)") &&
                  strstr(compat, "r300_unpack(fb, uvec4(3u, 2u, 1u, 0u), 0u)"),
                  "MRT framebuffer load did not use the effective endian");
            CHECK(strstr(original, "r300_pack(c, uvec4(3u, 2u, 1u, 0u), 1u)") &&
                  strstr(compat, "r300_pack(c, uvec4(3u, 2u, 1u, 0u), 0u)"),
                  "MRT framebuffer store did not use the effective endian");
        }
        st.morphos_color_compat = false;
        char *restored = r300_us_glsl_cached(&st, &desc, flags, &restored_id, NULL, &err);
        CHECK(restored && restored_id == original_id && original &&
              !strcmp(restored, original), "return to default missed its original shader cache");
        free(original); free(legacy); free(compat); free(restored);
    }
}

/* Sampler metadata follows declarations, including cached text-by-id hits. */
static void sampler_masks(void)
{
    R300FSDesc desc = { 0 };
    const char *err;
    uint32_t id, mask, first_id;
    char *text;

    for (unsigned ssbo = 0; ssbo < 2; ssbo++) {
        uint32_t flags = R300_GLSL_TEXT_BY_ID | (ssbo ? R300_GLSL_VRAM_SSBO : 0);
        load_state();
        for (unsigned unit = 0; unit < 16; unit++) {
            r300_state_write(&st, 0x4104, 1u << unit);
            r300_state_write(&st, 0x44C0 + 4 * unit, 0x0000a60c);
            r300_state_write(&st, 0x4620, 0x8040 | unit << 11);
            text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
            CHECK(text && mask == (1u << unit), "unit %u mask %x", unit, mask);
            free(text);
            first_id = id;
            text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
            CHECK(text && !*text && id == first_id && mask == (1u << unit),
                  "cached unit %u lost sampler metadata", unit);
            free(text);
            r300_state_write(&st, 0x4104, 0); /* Used but unbound still needs a dummy. */
            text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
            CHECK(text && mask == (1u << unit), "unbound unit %u mask %x", unit, mask);
            free(text);
            r300_state_write(&st, 0x4104, 1u << unit);
            r300_state_write(&st, 0x44C0 + 4 * unit, 0x0000a61b); /* RAW 32F */
            text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
            CHECK(text && mask == (ssbo ? 0 : 1u << unit),
                  "RAW unit %u SSBO=%u mask %x", unit, ssbo, mask);
            free(text);
        }
    }

    load_state();
    uint32_t flags = R300_GLSL_VRAM_SSBO | R300_GLSL_TEXT_BY_ID;
    r300_state_write(&st, 0x4104, 0x8008);
    r300_state_write(&st, 0x461C, 0x00420000); /* Two texture instructions. */
    r300_state_write(&st, 0x4620, 0x8040 | 3u << 11);
    r300_state_write(&st, 0x4624, 0x8041 | 15u << 11);
    r300_state_write(&st, 0x44C0 + 4 * 3, 0x0000a60c);
    r300_state_write(&st, 0x44C0 + 4 * 15, 0x0000a60c);
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(text && mask == 0x8008, "sparse sampler mask %x", mask);
    free(text);
    first_id = id;
    r300_state_write(&st, 0x44C0 + 4 * 15, 0x0000a61b);
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(text && mask == 8 && id != first_id, "RAW format reused sampled metadata");
    free(text);
    r300_state_write(&st, 0x44C0 + 4 * 15, 0x0000a60c);
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(text && id == first_id && mask == 0x8008, "shader switch missed prior mask");
    free(text);
    for (unsigned i = 0; i < 70; i++) {
        desc.route[0] = i % 10;
        desc.route[1] = i / 10;
        text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
        CHECK(text && mask == 0x8008, "memo eviction changed sampler mask");
        free(text);
    }
    text = r300_us_glsl_by_id(first_id);
    CHECK(!text, "old shader survived more than 64 distinct memo entries");
    free(text);
    memset(&desc, 0, sizeof(desc));
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(text && id != first_id && mask == 0x8008, "evicted mask did not regenerate");
    free(text);
    r300_state_write(&st, 0x4620, 0); /* NOP */
    r300_state_write(&st, 0x4624, 2u << 15); /* KIL */
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(text && mask == 0, "NOP/KIL incorrectly declare samplers");
    free(text);
    r300_state_write(&st, 0x4620, 7u << 15); /* Unsupported program. */
    text = r300_us_glsl_cached(&st, &desc, flags, &id, &mask, &err);
    CHECK(!text && err && mask == UINT32_MAX, "unknown shader did not keep conservative mask");
    free(text);

    for (unsigned gpu_vs = 0; gpu_vs < 2; gpu_vs++) {
        load_state();
        st.glsl_flags = flags | (gpu_vs ? R300_GLSL_GPU_VS : 0);
        r300_state_write(&st, 0x4104, 8); /* Disable the plain-copy shortcut. */
        r300_state_write(&st, 0x4620, 0x8040 | 3u << 11);
        r300_state_write(&st, 0x4278, 0xaaaa); /* Interpolated colours. */
        r300_state_write(&st, 0x4288, 0);
        r300_state_write(&st, 0x2090, r300_reg(&st, 0x2090) & ~(3u << 3));
        R300Arrays none = { 0 };
        R300DrawPacket pkt;
        CHECK(r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &pkt, &err),
              "sampler packet build: %s", err ? err : "");
        CHECK(pkt.sampler_mask_valid && pkt.sampler_mask == 8,
              "packet lost sampler mask (GPU VS flag=%u)", gpu_vs);
        CHECK((pkt.vs_glsl != NULL) == (gpu_vs != 0), "wrong vertex stage path");
        r300_draw_free(&pkt);
    }
}

int main(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err;

    load_state();

    /* The compositor's own state: no depth, no culling. */
    CHECK(r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err), "build: %s", err);
    CHECK(!p.depth.attach && p.cull == 0, "QE draw has depth %d cull %x", p.depth.attach, p.cull);
    r300_draw_free(&p);

    /* ZB_CNTL.STENCIL_FRONT_BACK alone (what Apple leaves set) is not a test. */
    r300_state_write(&st, 0x4F00, 0x10);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(!p.depth.attach, "FRONT_BACK alone attached depth");
    r300_draw_free(&p);

    /* Z24S8, Z test + write, dword-swapped, macro-tiled pitch 704. */
    r300_state_write(&st, 0x4F00, 0x06);
    r300_state_write(&st, 0x4F04, 0x00000001);
    r300_state_write(&st, 0x4F08, 0x00FFFF00);
    r300_state_write(&st, 0x4F10, 2);
    r300_state_write(&st, 0x4F20, 0x01000000 | 0x3);
    r300_state_write(&st, 0x4F24, 704 | (1u << 16) | (2u << 19));
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.depth.attach && p.depth.gpu_addr == 0x01000000 && p.depth.pitch == 704 &&
          p.depth.bpp == 4, "depth %d %08x %u %u", p.depth.attach, p.depth.gpu_addr,
          p.depth.pitch, p.depth.bpp);
    CHECK(p.uniforms.zinfo[0] == 6 && p.uniforms.zinfo[1] == 1 &&
          p.uniforms.zinfo[2] == 0x00FFFF00 && p.uniforms.zinfo[3] == 2,
          "zinfo %x %x %x %x", p.uniforms.zinfo[0], p.uniforms.zinfo[1],
          p.uniforms.zinfo[2], p.uniforms.zinfo[3]);
    CHECK(!(p.warn & R300_WARN_DEPTH), "Z24S8 warned");
    r300_draw_free(&p);

    /* Z16 */
    r300_state_write(&st, 0x4F10, 0);
    r300_state_write(&st, 0x4F24, 1024);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.depth.bpp == 2 && p.uniforms.zinfo[3] == R300_ZFMT_Z16, "z16 %u %x",
          p.depth.bpp, p.uniforms.zinfo[3]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4F00, 0);

    /* Culling: quads cull, points and lines never do. */
    r300_state_write(&st, 0x42B8, R300_CULL_BACK | R300_FACE_CW);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.cull == (R300_CULL_BACK | R300_FACE_CW), "quad cull %x", p.cull);
    r300_draw_free(&p);
    uint32_t lines[49];
    memcpy(lines, immd, sizeof(lines));
    lines[0] = (lines[0] & ~0xFu) | 2;
    r300_draw_build(&st, &none, 0x35, lines, 49, rd, NULL, &p, &err);
    CHECK(p.cull == 0 && p.prim_class == 1, "lines cull %x cls %u", p.cull, p.prim_class);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);

    /*
     * DRAW_INDX_2 + INDX_BUFFER: 4 vertices (pos, colour, texcoord; 12
     * dwords each) in memory, drawn as a triangle list 2,1,0, 0,3,2 of
     * 16-bit indices.  The vertex cache applies VC_SWAP to index dwords
     * like any fetched data; the first index is the low half.
     */
    static const float vtx[4][12] = {
        { 0, 0, 0, 1,  1, 0, 0, 1,  0, 0, 0, 1 },
        { 100, 0, 0, 1,  0, 1, 0, 1,  100, 0, 0, 1 },
        { 100, 100, 0, 1,  0, 0, 1, 1,  100, 100, 0, 1 },
        { 0, 100, 0, 1,  1, 1, 1, 1,  0, 100, 0, 1 },
    };
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 12; k++)
            put_be(0x100 + (i * 12 + k) * 4, fbits(vtx[i][k]));
    uint16_t ix[6] = { 2, 1, 0, 0, 3, 2 };
    uint32_t idw[3];
    for (int k = 0; k < 3; k++) {
        uint32_t w = ix[2 * k] | (uint32_t)ix[2 * k + 1] << 16;
        put_be(0x1000 + k * 4, w);
        memcpy(&idw[k], mem + 0x1000 + k * 4, 4);    /* as fetched */
    }
    R300Arrays arr = { 1 };
    arr.a[0].addr = MEM_BASE + 0x100;
    arr.a[0].size_dw = 12;
    arr.a[0].stride_dw = 12;
    R300Indices idx = { idw, 3 };
    uint32_t vf = (6u << 16) | (1u << 4) | 4;         /* 6 indices, walk indices, tris */
    CHECK(r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err),
          "indexed: %s", err ? err : "");
    if (p.verts) {
        CHECK(p.num_verts == 6 && p.prim_class == 0, "indexed verts %u", p.num_verts);
        /* colour varying (v[1] in this state) identifies the source vertex */
        static const int want[6] = { 2, 1, 0, 0, 3, 2 };
        for (int i = 0; i < 6 && i < (int)p.num_verts; i++) {
            const float *c = vtx[want[i]] + 4;
            CHECK(!memcmp(p.verts[i].v[1], c, 16), "index %d: colour %.0f %.0f %.0f", i,
                  p.verts[i].v[1][0], p.verts[i].v[1][1], p.verts[i].v[1][2]);
        }
    }
    r300_draw_free(&p);

    /* 32-bit indices */
    uint32_t i32[3] = { 3, 2, 1 };
    for (int k = 0; k < 3; k++) {
        put_be(0x2000 + k * 4, i32[k]);
        memcpy(&idw[k], mem + 0x2000 + k * 4, 4);
    }
    vf = (3u << 16) | (1u << 11) | (1u << 4) | 4;
    CHECK(r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err), "i32: %s", err);
    CHECK(p.num_verts == 3 && !memcmp(p.verts[0].v[1], vtx[3] + 4, 16) &&
          !memcmp(p.verts[2].v[1], vtx[1] + 4, 16), "i32 order");
    r300_draw_free(&p);

    /* An index buffer shorter than the draw is refused. */
    idx.ndw = 1;
    CHECK(!r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err), "short ib");

    /* Multisample layout: every sample of a 64x16 buffer lands on a
     * distinct slot inside the buffer, for 2 and 4 samples. */
    for (unsigned ns = 2; ns <= 4; ns += 2) {
        static uint8_t used[64 * 16 * 4 * 4];
        memset(used, 0, sizeof(used));
        bool ok = true;
        for (unsigned y = 0; y < 16; y++)
            for (unsigned x = 0; x < 64; x++)
                for (unsigned k = 0; k < ns; k++) {
                    uint32_t o = r300_msaa_offset(x, y, ns, 64, 4, k);
                    if (o % 4 || o / 4 >= 64 * 16 * ns || used[o / 4]++) ok = false;
                }
        CHECK(ok, "msaa layout ns=%u not a permutation", ns);
    }
    /* The driver's own formula at a picked pixel (2 samples, 704 wide). */
    CHECK(r300_msaa_offset(592, 496, 2, 704, 4, 0) == 0x2b3400, "msaa sample 0 %x",
          r300_msaa_offset(592, 496, 2, 704, 4, 0));
    rendering_completeness();
    sampler_masks();
    bypass_semantics();
    viewport_ports();
    draw_bounds();
    upload_ports();
    native_texture_bytes();
    texture_layout_bounds();
    morphos_color_profile();
    printf(fails ? "test_features: %d FAILED\n" : "test_features: PASS\n", fails);
    return fails != 0;
}
