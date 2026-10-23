/*
 * QEMU graphical console -- opengl helper bits
 *
 * Copyright (c) 2014 Red Hat
 *
 * Authors:
 *    Gerd Hoffmann <kraxel@redhat.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "qemu/osdep.h"
#include "ui/console.h"
#include "ui/shader.h"

/* ---------------------------------------------------------------------- */

bool console_gl_check_format(DisplayChangeListener *dcl,
                             pixman_format_code_t format)
{
    switch (format) {
    case PIXMAN_BE_b8g8r8x8:
    case PIXMAN_BE_b8g8r8a8:
    case PIXMAN_r5g6b5:
        return true;
    default:
        return false;
    }
}

void surface_gl_create_texture(QemuGLShader *gls,
                               DisplaySurface *surface)
{
    assert(gls);
    assert(QEMU_IS_ALIGNED(surface_stride(surface), surface_bytes_per_pixel(surface)));

    if (surface->texture) {
        return;
    }

    switch (surface_format(surface)) {
    case PIXMAN_BE_b8g8r8x8:
    case PIXMAN_BE_b8g8r8a8:
        surface->glformat = GL_RGBA;
        surface->gltype = GL_UNSIGNED_BYTE;
        surface->glswapped = true;
        break;
    case PIXMAN_BE_x8r8g8b8:
    case PIXMAN_BE_a8r8g8b8:
        surface->glformat = GL_RGBA;
        surface->gltype = GL_UNSIGNED_BYTE;
        break;
    case PIXMAN_r5g6b5:
        surface->glformat = GL_RGB;
        surface->gltype = GL_UNSIGNED_SHORT_5_6_5;
        break;
    default:
        g_assert_not_reached();
    }

    glGenTextures(1, &surface->texture);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, surface->texture);
    glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT,
                  surface_stride(surface) / surface_bytes_per_pixel(surface));
    if (epoxy_is_desktop_gl()) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     surface_width(surface),
                     surface_height(surface),
                     0, surface->glformat, surface->gltype,
                     surface_data(surface));
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, surface->glformat,
                     surface_width(surface),
                     surface_height(surface),
                     0, surface->glformat, surface->gltype,
                     surface_data(surface));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
    }

    /* $QEMU_GL_FILTER=nearest: unfiltered scaling (crisp at integer
     * ratios, e.g. a 1920x1080 guest on a 3840x2160 monitor); "auto"
     * switches per frame in surface_gl_setup_viewport */
    {
        const char *f = getenv("QEMU_GL_FILTER");
        GLint filter = f && !strcmp(f, "nearest") ? GL_NEAREST : GL_LINEAR;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    }
}

/* The guest picture fitted into a ww x wh window, aspect kept, centred. */
void surface_gl_fit(int gw, int gh, int ww, int wh,
                    int *x, int *y, int *w, int *h)
{
    if ((int64_t)ww * gh < (int64_t)wh * gw) {
        *w = ww;
        *h = (int64_t)ww * gh / gw;
    } else {
        *w = (int64_t)wh * gw / gh;
        *h = wh;
    }
    *x = (ww - *w) / 2;
    *y = (wh - *h) / 2;
}

void surface_gl_update_texture(QemuGLShader *gls,
                               DisplaySurface *surface,
                               int x, int y, int w, int h)
{
    uint8_t *data = (void *)surface_data(surface);

    assert(gls);

    if (surface->texture) {
        glBindTexture(GL_TEXTURE_2D, surface->texture);
        glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT,
                      surface_stride(surface)
                      / surface_bytes_per_pixel(surface));
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        x, y, w, h,
                        surface->glformat, surface->gltype,
                        data + surface_stride(surface) * y
                        + surface_bytes_per_pixel(surface) * x);
    }
}

void surface_gl_render_texture(QemuGLShader *gls,
                               DisplaySurface *surface)
{
    assert(gls);

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    qemu_gl_run_texture_blit(gls, false, surface->glswapped);
}

void surface_gl_destroy_texture(QemuGLShader *gls,
                                DisplaySurface *surface)
{
    if (!surface || !surface->texture) {
        return;
    }
    glDeleteTextures(1, &surface->texture);
    surface->texture = 0;
}

void surface_gl_setup_viewport(QemuGLShader *gls,
                               DisplaySurface *surface,
                               int ww, int wh)
{
    int gw, gh, x, y, w, h;
    const char *f;

    assert(gls);

    gw = surface_width(surface);
    gh = surface_height(surface);

    /* integer fit: a 1440x900 guest in 2880x1920 gets exactly 2880x1800 */
    surface_gl_fit(gw, gh, ww, wh, &x, &y, &w, &h);
    glViewport(x, y, w, h);

    /* $QEMU_GL_FILTER=auto: nearest at whole-number ratios (1:1, 2x, ...),
     * linear in between */
    f = getenv("QEMU_GL_FILTER");
    if (f && !strcmp(f, "auto") && surface->texture) {
        GLint filter = w % gw == 0 && h % gh == 0 ? GL_NEAREST : GL_LINEAR;
        glBindTexture(GL_TEXTURE_2D, surface->texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    }
}
