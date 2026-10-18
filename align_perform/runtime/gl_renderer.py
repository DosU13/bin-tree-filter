"""GPU renderer: draws OrnamentTree state with an OpenGL 3.3 fragment shader (moderngl).

Same picture as core.engine.NumpyRenderer, but every pixel walks the triangle
tree itself, so there are no lookup tables: changing the output size or the
mapping costs nothing, and 1080p renders in well under a millisecond.

One GLRenderer per OpenGL context.  It has no Qt dependency: projector windows
pass the context Qt made current, tests and tools use OffscreenGL.

Pipeline per frame:
    upload(snapshot)            node colours / split amounts -> textures
    render(fbo, view)           key pass (only for cord / wireframe), then main pass
"""

from __future__ import annotations

from dataclasses import dataclass, field

import moderngl
import numpy as np

from core.engine import TreeSnapshot
from core.view import BlendSpec, RenderView  # noqa: F401  (re-exported)

NODE_TEX_W = 2048
MAX_QUADS = 8
POLARITY_CODES = {"alternate": 0, "none": 1, "invert": 2}


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

VERTEX = """
#version 330 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

COMMON = """
#version 330 core
uniform mat3 u_m;          // normalised output coords -> stage cells
uniform vec2 u_size;       // output size in pixels
uniform ivec2 u_grid;      // cols, rows
uniform int u_lm;          // max level
uniform int u_tex_w;       // node texture width
uniform sampler2D u_nodes; // b, g, r, amount
uniform isampler2D u_dom;  // dominant palette index per node
uniform bool u_has_state;
uniform int u_edge_mode;   // 0 none, 1 cord, 2 wireframe

struct Tri { vec2 ra; vec2 ha; vec2 hb; int idx; };

vec2 to_cell(vec2 pix) {
    vec3 h = u_m * vec3(pix / u_size, 1.0);
    return h.xy / h.z;
}

bool inside(vec2 p) {
    return p.x >= 0.0 && p.y >= 0.0 && p.x < float(u_grid.x) && p.y < float(u_grid.y);
}

int level_offset(int level) {
    return u_grid.x * u_grid.y * ((2 << level) - 2);
}

vec4 node_at(int i) {
    return texelFetch(u_nodes, ivec2(i % u_tex_w, i / u_tex_w), 0);
}

// Level-0 triangle containing p; local = p relative to its cell
Tri root_of(vec2 p, out vec2 local) {
    vec2 c = clamp(floor(p), vec2(0.0), vec2(u_grid) - 1.0);
    local = p - c;
    bool lower = local.y > local.x;
    Tri t;
    t.idx = (int(c.y) * u_grid.x + int(c.x)) * 2 + (lower ? 1 : 0);
    t.ra = lower ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    t.ha = lower ? vec2(1.0, 1.0) : vec2(0.0, 0.0);
    t.hb = lower ? vec2(0.0, 0.0) : vec2(1.0, 1.0);
    return t;
}

// Split from the right angle to the hypotenuse midpoint (child 2i = ra/ha side)
void down(inout Tri t, vec2 q) {
    vec2 mid = (t.ha + t.hb) * 0.5;
    vec2 d = mid - t.ra;
    float side_p = d.x * (q.y - t.ra.y) - d.y * (q.x - t.ra.x);
    float side_a = d.x * (t.ha.y - t.ra.y) - d.y * (t.ha.x - t.ra.x);
    bool first = side_p * side_a >= 0.0;
    t.idx = t.idx * 2 + (first ? 0 : 1);
    t.hb = first ? t.ha : t.hb;
    t.ha = t.ra;
    t.ra = mid;
}

// Deepest fully-open level along p's path; t ends as that level's triangle
int open_level(inout Tri t, vec2 local) {
    int level = 0;
    while (level < u_lm && node_at(level_offset(level) + t.idx).a >= 1.0) {
        down(t, local);
        level++;
    }
    return level;
}

// Edge key: visible leaf (wireframe) or its dominant colour (cord); -1 outside
int key_at(vec2 pix) {
    vec2 p = to_cell(pix);
    if (!inside(p) || !u_has_state) return -1;
    vec2 local;
    Tri t = root_of(p, local);
    int level = open_level(t, local);
    int vis = level_offset(level) + t.idx;
    if (node_at(vis).a >= 0.5 && level < u_lm) {
        down(t, local);
        vis = level_offset(level + 1) + t.idx;
    }
    if (u_edge_mode == 2) return vis;
    return texelFetch(u_dom, ivec2(vis % u_tex_w, vis / u_tex_w), 0).r;
}
"""

KEY_FRAGMENT = COMMON + """
uniform int u_margin;
layout(location = 0) out int o_key;
void main() {
    // Key texture is the output extended by u_margin pixels on every side
    vec2 g = gl_FragCoord.xy - vec2(float(u_margin));
    o_key = key_at(vec2(g.x, u_size.y - g.y));
}
"""

MAIN_FRAGMENT = COMMON + """
uniform sampler2D u_tile;  // ink coverage, mipmapped
uniform vec2 u_tile_size;
uniform int u_polarity;    // 0 alternate, 1 none, 2 invert
uniform vec3 u_bg;         // BGR 0..255
uniform vec3 u_outside;    // BGR 0..255
uniform int u_ss;
uniform bool u_mip;
uniform isampler2D u_keys;
uniform int u_margin;
uniform int u_edge_w;
uniform int u_nq;
uniform vec2 u_quads[32];
uniform float u_feather[8];
uniform int u_edges[8];    // bit k: edge k of quad q fades
uniform int u_self;
uniform float u_gamma;
uniform float u_gain;
uniform bool u_flat;       // test pattern: flat colour instead of the ornament (blend still applies)
uniform vec3 u_flat_color;
out vec4 frag;

float tile_coverage(vec2 uv, int level, float lod) {
    vec2 tc = vec2(((1.0 - uv.x) * (u_tile_size.x - 1.0) + 0.5) / u_tile_size.x,
                   (uv.y * (u_tile_size.y - 1.0) + 0.5) / u_tile_size.y);
    float c = textureLod(u_tile, tc, u_mip ? lod : 0.0).r;
    bool inv = u_polarity == 0 ? (level % 2 == 0) : (u_polarity == 2);
    return inv ? 1.0 - c : c;
}

// Supersampled coverage of pixel `pix` at `level` (each sample in its own level triangle)
float coverage(vec2 pix, int level, float cells_per_px) {
    float leg = exp2(-0.5 * float(level));
    float texels = u_tile_size.x * cells_per_px / float(u_ss) / leg;
    float lod = log2(max(texels, 1e-6));
    float sum = 0.0;
    for (int j = 0; j < u_ss; j++) {
        for (int i = 0; i < u_ss; i++) {
            vec2 s = floor(pix) + (vec2(float(i), float(j)) + 0.5) / float(u_ss);
            vec2 local;
            Tri t = root_of(to_cell(s), local);
            for (int l = 0; l < level; l++) down(t, local);
            vec2 e1 = t.ha - t.ra, e2 = t.hb - t.ra, d = local - t.ra;
            float det = e1.x * e2.y - e1.y * e2.x;
            vec2 uv = vec2((d.x * e2.y - d.y * e2.x) / det, (e1.x * d.y - e1.y * d.x) / det);
            sum += tile_coverage(uv, level, lod);
        }
    }
    float cov = sum / float(u_ss * u_ss);
    return floor(clamp(cov, 0.0, 1.0) * 255.0 + 0.5) / 255.0;   // 8-bit, like the lookup tables
}

bool edge_at(ivec2 c) {
    int k = texelFetch(u_keys, c, 0).r;
    return k != texelFetch(u_keys, c + ivec2(-1, 0), 0).r
        || k != texelFetch(u_keys, c + ivec2(0, 1), 0).r;   // gl +y = the row above
}

float edge_distance(vec2 p, int q, int i) {
    vec2 a = u_quads[q * 4 + i];
    vec2 b = u_quads[q * 4 + (i + 1) % 4];
    vec2 e = b - a;
    return dot(p - a, normalize(vec2(-e.y, e.x)));
}

// Blend weight of quad q at p: 0 outside, else the product of ramps towards its
// fading (interior) edges
float quad_weight(vec2 p, int q) {
    float w = 1.0;
    for (int i = 0; i < 4; i++) {
        float d = edge_distance(p, q, i);
        if (d <= 0.0) return 0.0;
        if ((u_edges[q] & (1 << i)) != 0) w *= smoothstep(0.0, max(u_feather[q], 1e-6), d);
    }
    return w + 1e-4;
}

void main() {
    vec2 g = gl_FragCoord.xy;
    vec2 pix = vec2(g.x, u_size.y - g.y);          // top-left pixel coords, centre at +0.5
    vec2 p = to_cell(pix);
    vec3 col;

    if (u_flat) {
        col = u_flat_color;
    } else if (!inside(p)) {
        col = u_outside;
    } else if (!u_has_state) {
        col = u_bg;
    } else {
        float cells_per_px = max(length(to_cell(pix + vec2(1.0, 0.0)) - p),
                                 length(to_cell(pix + vec2(0.0, 1.0)) - p));
        vec2 local;
        Tri t = root_of(p, local);
        int level = open_level(t, local);
        vec4 na = node_at(level_offset(level) + t.idx);
        float cov_a = coverage(pix, level, cells_per_px);
        col = u_bg + cov_a * (na.rgb - u_bg);
        float amt = na.a;
        if (amt > 0.0 && level < u_lm) {
            Tri tb = t;
            down(tb, local);
            vec4 nb = node_at(level_offset(level + 1) + tb.idx);
            float cov_b = coverage(pix, level + 1, cells_per_px);
            vec3 col_b = u_bg + cov_b * (nb.rgb - u_bg);
            col = col * (1.0 - amt) + col_b * amt;
        }

        if (u_edge_mode != 0) {
            ivec2 kc = ivec2(g) + ivec2(u_margin);
            bool e = false;
            if (u_edge_mode == 2) {
                e = edge_at(kc);
            } else {
                int a = u_edge_w / 2;
                for (int dy = -a; dy <= u_edge_w - 1 - a && !e; dy++)
                    for (int dx = -a; dx <= u_edge_w - 1 - a && !e; dx++)
                        e = edge_at(kc + ivec2(dx, -dy));
            }
            if (e) col = u_edge_mode == 2 ? vec3(0.0) : floor(u_bg);
        }
        col = floor(col + 0.5);
    }

    if (u_nq > 0) {
        float wsum = 0.0, wself = 0.0;
        for (int q = 0; q < u_nq; q++) {
            float w = quad_weight(p, q);
            wsum += w;
            if (q == u_self) wself = w;
        }
        float alpha = wsum > 0.0 ? wself / wsum : 1.0;
        col *= pow(alpha, 1.0 / u_gamma);
    }
    col *= u_gain;
    frag = vec4(col.zyx / 255.0, 1.0);
}
"""

IMAGE_FRAGMENT = """
#version 330 core
uniform sampler2D u_image;
uniform float u_opacity;
out vec4 frag;
in vec2 v_uv;
void main() {
    vec4 c = texture(u_image, v_uv);
    frag = vec4(c.rgb, c.a * u_opacity);
}
"""

IMAGE_VERTEX = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);   // image row 0 at the top
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _mat3_bytes(m: np.ndarray) -> bytes:
    """Row-major numpy 3x3 -> GLSL column-major mat3 bytes."""
    return np.asarray(m, dtype=np.float32).T.tobytes()


def _set(prog: moderngl.Program, name: str, value=None, data: bytes | None = None) -> None:
    """Set a uniform; uniforms the GLSL compiler optimised away are skipped."""
    member = prog.get(name, None)
    if member is None:
        return
    if data is not None:
        member.write(data)
    else:
        member.value = value


def _ccw_quads(quads: np.ndarray) -> np.ndarray:
    """Order every quad so its edges' left-hand normals point inwards (y-down coords)."""
    q = np.asarray(quads, dtype=np.float32).copy()
    for i in range(len(q)):
        x, y = q[i, :, 0], q[i, :, 1]
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        if area < 0:
            q[i] = q[i, ::-1]
    return q


class GLRenderer:
    """Ornament shader + textures for one OpenGL context."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.main = ctx.program(vertex_shader=VERTEX, fragment_shader=MAIN_FRAGMENT)
        self.keys = ctx.program(vertex_shader=VERTEX, fragment_shader=KEY_FRAGMENT)
        self.image = ctx.program(vertex_shader=IMAGE_VERTEX, fragment_shader=IMAGE_FRAGMENT)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao_main = ctx.vertex_array(self.main, [(self._vbo, "2f", "in_pos")])
        self._vao_keys = ctx.vertex_array(self.keys, [(self._vbo, "2f", "in_pos")])
        self._vao_image = ctx.vertex_array(self.image, [(self._vbo, "2f", "in_pos")])
        self._tile: moderngl.Texture | None = None
        self._nodes: moderngl.Texture | None = None
        self._dom: moderngl.Texture | None = None
        self._key_fbo: moderngl.Framebuffer | None = None
        self._key_tex: moderngl.Texture | None = None
        self._images: dict[str, moderngl.Texture] = {}
        self._state: TreeSnapshot | None = None
        self._uploaded_version: int | None = None
        # 1x1 integer texture bound to unused samplers (int samplers must not be empty)
        self._dummy_i = ctx.texture((1, 1), 1, data=np.array([-1], np.int32).tobytes(), dtype="i4")
        self._dummy_i.filter = (moderngl.NEAREST, moderngl.NEAREST)

    # -- resources -----------------------------------------------------------

    def set_tile(self, tile: np.ndarray) -> None:
        """Ornament ink coverage (float32, 1 = ink), as from core.engine.load_tile."""
        if self._tile is not None:
            self._tile.release()
        h, w = tile.shape
        tex = self.ctx.texture((w, h), 1, data=np.ascontiguousarray(tile, dtype=np.float32).tobytes(), dtype="f4")
        tex.repeat_x = tex.repeat_y = False
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._tile = tex

    def upload(self, snap: TreeSnapshot) -> None:
        """Send one frame of tree state to the GPU (skipped if this version is already there)."""
        if self._state is snap and self._uploaded_version == snap.version:
            return
        n = snap.n_nodes
        rows = (n + NODE_TEX_W - 1) // NODE_TEX_W
        nodes = np.zeros((rows * NODE_TEX_W, 4), dtype=np.float32)
        dom = np.full(rows * NODE_TEX_W, -1, dtype=np.int32)
        if snap.color is not None:
            nodes[:n, :3] = snap.color
            nodes[:n, 3] = snap.amount
            dom[:n] = snap.dom
        if self._nodes is None or self._nodes.size != (NODE_TEX_W, rows):
            for t in (self._nodes, self._dom):
                if t is not None:
                    t.release()
            self._nodes = self.ctx.texture((NODE_TEX_W, rows), 4, dtype="f4")
            self._dom = self.ctx.texture((NODE_TEX_W, rows), 1, dtype="i4")
            for t in (self._nodes, self._dom):
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._nodes.write(nodes.tobytes())
        self._dom.write(dom.tobytes())
        self._state = snap
        self._uploaded_version = snap.version

    def _key_target(self, w: int, h: int) -> moderngl.Framebuffer:
        if self._key_tex is None or self._key_tex.size != (w, h):
            if self._key_fbo is not None:
                self._key_fbo.release()
                self._key_tex.release()
            self._key_tex = self.ctx.texture((w, h), 1, dtype="i4")
            self._key_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._key_fbo = self.ctx.framebuffer(color_attachments=[self._key_tex])
        return self._key_fbo

    # -- drawing -------------------------------------------------------------

    def _set_common(self, prog: moderngl.Program, view: RenderView, edge_mode: int) -> None:
        snap = self._state
        _set(prog, "u_m", data=_mat3_bytes(view.norm_to_cell))
        _set(prog, "u_size", (float(view.width), float(view.height)))
        if snap is not None:
            _set(prog, "u_grid", (snap.cols, snap.rows))
            _set(prog, "u_lm", snap.max_level)
        else:
            _set(prog, "u_grid", (1, 1))
            _set(prog, "u_lm", 0)
        _set(prog, "u_tex_w", NODE_TEX_W)
        _set(prog, "u_has_state", bool(snap is not None and snap.color is not None))
        _set(prog, "u_edge_mode", edge_mode)
        _set(prog, "u_nodes", 0)
        _set(prog, "u_dom", 1)

    def render(self, fbo: moderngl.Framebuffer, view: RenderView) -> None:
        """Draw the uploaded state into *fbo* (its viewport must be view.width x view.height)."""
        ctx = self.ctx
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        have_nodes = self._nodes is not None
        if have_nodes:
            self._nodes.use(0)
            self._dom.use(1)
        else:
            self._dummy_i.use(1)
        edge_mode = 2 if view.wireframe else (1 if view.cord else 0)
        edge_w = view.cord_width
        margin = edge_w + 1

        if edge_mode:
            kfbo = self._key_target(view.width + 2 * margin, view.height + 2 * margin)
            kfbo.use()
            self._set_common(self.keys, view, edge_mode)
            _set(self.keys, "u_margin", margin)
            self._vao_keys.render(moderngl.TRIANGLE_STRIP)
            self._key_tex.use(3)
        else:
            self._dummy_i.use(3)

        fbo.use()
        p = self.main
        self._set_common(p, view, edge_mode)
        if self._tile is not None:
            self._tile.use(2)
            _set(p, "u_tile_size", tuple(float(s) for s in self._tile.size))
        _set(p, "u_tile", 2)
        _set(p, "u_keys", 3)
        _set(p, "u_polarity", POLARITY_CODES[view.polarity])
        _set(p, "u_bg", tuple(float(c) for c in view.background))
        _set(p, "u_outside", tuple(float(c) for c in view.outside))
        _set(p, "u_ss", max(1, int(view.supersample)))
        _set(p, "u_mip", bool(view.mip))
        _set(p, "u_margin", margin)
        _set(p, "u_edge_w", edge_w)
        _set(p, "u_gain", float(view.gain))
        _set(p, "u_flat", view.flat is not None)
        _set(p, "u_flat_color", tuple(float(c) for c in (view.flat or (0, 0, 0))))
        b = view.blend
        if b is not None and len(b.quads):
            quads = _ccw_quads(b.quads)[:MAX_QUADS]
            nq = len(quads)
            qdata = np.zeros((MAX_QUADS * 4, 2), dtype=np.float32)
            qdata[: nq * 4] = quads.reshape(-1, 2)
            feather = np.zeros(MAX_QUADS, dtype=np.float32)
            feather[:nq] = np.asarray(b.feather, dtype=np.float32)[:nq]
            _set(p, "u_nq", nq)
            _set(p, "u_quads", data=qdata.tobytes())
            _set(p, "u_feather", data=feather.tobytes())
            edges = np.ones((MAX_QUADS, 4), bool)
            if b.edges is not None:
                edges[:] = False
                edges[:nq] = np.asarray(b.edges, bool)[:nq]
            bits = (edges * (1 << np.arange(4))).sum(axis=1).astype(np.int32)
            _set(p, "u_edges", data=bits.tobytes())
            _set(p, "u_self", int(b.index))
            _set(p, "u_gamma", float(b.gamma))
        else:
            _set(p, "u_nq", 0)
        self._vao_main.render(moderngl.TRIANGLE_STRIP)

    def draw_image(self, fbo: moderngl.Framebuffer, name: str, rgba: np.ndarray | None = None,
                   opacity: float = 1.0) -> None:
        """Draw a full-frame RGBA8 image (test pattern, overlay) over *fbo*, alpha-blended.

        Pass *rgba* to (re)upload the image cached under *name*; omit it to reuse it.
        """
        if rgba is not None:
            h, w = rgba.shape[:2]
            tex = self._images.get(name)
            if tex is None or tex.size != (w, h):
                if tex is not None:
                    tex.release()
                tex = self.ctx.texture((w, h), 4)
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._images[name] = tex
            tex.write(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes())
        tex = self._images.get(name)
        if tex is None:
            return
        fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(4)
        _set(self.image, "u_image", 4)
        _set(self.image, "u_opacity", float(opacity))
        self._vao_image.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def release(self) -> None:
        for obj in (self._tile, self._nodes, self._dom, self._key_fbo, self._key_tex, self._dummy_i,
                    *self._images.values(), self._vao_main, self._vao_keys, self._vao_image,
                    self._vbo, self.main, self.keys, self.image):
            if obj is not None:
                obj.release()


# ---------------------------------------------------------------------------
# Offscreen rendering (tests, tools, previews)
# ---------------------------------------------------------------------------

@dataclass
class OffscreenGL:
    """Standalone OpenGL context that renders to numpy arrays.

    Safe to use alongside Qt's contexts: every call makes this context current.
    """

    ctx: moderngl.Context = field(default_factory=lambda: moderngl.create_standalone_context(require=330))

    def __post_init__(self) -> None:
        with self.ctx:
            self.renderer = GLRenderer(self.ctx)
        self._fbos: dict[tuple[int, int], moderngl.Framebuffer] = {}

    def _fbo(self, w: int, h: int) -> moderngl.Framebuffer:
        fbo = self._fbos.get((w, h))
        if fbo is None:
            fbo = self.ctx.simple_framebuffer((w, h), components=4)
            self._fbos[(w, h)] = fbo
        return fbo

    def set_tile(self, tile: np.ndarray) -> None:
        with self.ctx:
            self.renderer.set_tile(tile)

    def render(self, snap: TreeSnapshot | None, view: RenderView) -> np.ndarray:
        """Render and read back as a (height, width, 3) BGR uint8 array."""
        with self.ctx:
            if snap is not None:
                self.renderer.upload(snap)
            fbo = self._fbo(view.width, view.height)
            fbo.use()
            fbo.viewport = (0, 0, view.width, view.height)
            self.renderer.render(fbo, view)
            data = fbo.read(components=3, alignment=1)
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(view.height, view.width, 3)[::-1]
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def release(self) -> None:
        with self.ctx:
            for fbo in self._fbos.values():
                fbo.release()
            self.renderer.release()
        self.ctx.release()
