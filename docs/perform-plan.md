# Plan: Align Perform + perform.py

Status: **in progress** on branch `feature/align-perform` (plan written 2026-10-17). Ticked boxes are done.

Two new pieces on top of the current `live.py` / `ornament_engine.py` / `stable_posterize.py`:

1. **`perform.py`**: the show runner used on stage / at the installation. It loads a saved scene, opens every projector full-screen on its own display, starts every camera and runs the ornament filter across all of them. It has no editor UI and must survive for hours unattended.
2. **`align_perform/`**: the setup app, split across many files. It's a window with buttons and sliders exposing every `live.py` option, plus a canvas where you add cameras and projectors as rectangles, size and align them, and save the result as a scene file for `perform.py`. For testing, any camera's source can be a video file.

The aim is one continuous ornament across several cameras and several projectors, with no visible seams where cameras or projectors overlap.

---

## 1. Decisions to confirm

The plan below assumes the **default** in each row. Answers change specific phases, noted on the right.

| # | Question | Default assumed | Affects |
|---|---|---|---|
| 1 | Is the projection roughly square-on to a flat wall/floor, or at an angle (needs keystone / 4-corner warp)? | Rectangles first; 4-corner warp as optional phase 6b | Phase 6 |
| 2 | Is `perform.py` an unattended installation, a live show someone operates (cues/looks), or both? | Both: unattended by default, plus hotkeys for looks, crossfades and blackout | Phase 7 |
| 3 | Move the venv from Python 3.10 to 3.12? Python 3.10 reached end of life in Oct 2026, and PySide6 6.12 already requires ≥ 3.10, so the next PySide6 may drop it. 3.12 is installed on this PC and also allows numpy 2.4. | Yes, in phase 0 (unless something like a depth-camera SDK needs 3.10) | Phase 0 |
| 4 | Names: `perform.py` (not `preform.py`) and folder `align_perform/`? | Yes | All |
| 5 | Show rig: how many projectors and cameras, at what resolutions, and does the show PC have an NVIDIA/AMD GPU? | Up to 4 projectors × 1080p, up to 4 webcams × 1080p, a discrete GPU | Phases 2, 7 |

---

## 2. What the measurements say

Measured on this PC (i7-12700, RTX 2070, one 2560×1440 monitor, no projectors attached), current engine, `max_level=12`:

| Step | Cost |
|---|---|
| `PaletteTracker.update`, proc 128 | ~10 ms |
| `OrnamentEngine.update` (tree decisions) | ~2 ms |
| `OrnamentEngine.render` (NumPy) | **~52 ms per megapixel**: 1080p ≈ 108 ms ≈ 9 fps per projector |
| `set_output` lookup-table build | **~7.5 s at 1024², ~17 s at 1536²**, so ~15 s per 1080p output |

Consequences:

- **Projector output must render on the GPU.** The NumPy renderer can't drive even one 1080p projector smoothly, and rebuilding its lookup tables after every rectangle drag would freeze alignment for ~15 s. A fragment shader can walk the triangle tree per pixel with no lookup tables, so geometry changes are instant and 4 × 1080p is easy for the RTX 2070.
- **Tree decisions and colour tracking stay on the CPU** (NumPy, ~12 ms per frame for a 1-cell stage, roughly linear in cells). Only a small node-state snapshot (a few hundred KB) goes to the GPU each frame.
- The NumPy renderer stays as the **reference** that GPU output is tested against, and as the fallback for `live.py` and machines without a GPU.

---

## 3. Concepts and coordinate spaces

- **World**: one 2D canvas in "world pixels". Everything is placed in it.
- **Stage**: a rectangle in the world where the ornament exists. It's divided into a grid of square **cells** of `cell_size` world px; each cell is one bintree root (2 root triangles). `cell_size` is the **ornament scale** slider.
- **Camera**: a source (webcam index, or video/image/GIF file for testing) plus a **transform** mapping camera pixels → world. Its rectangle's aspect ratio is locked to the source's real resolution, which is probed when the source is chosen. It also has mirror, 90° rotation and its own brightness/contrast/saturation for colour-matching cameras.
- **Projector**: an output (a display, or a plain window for testing) plus a **transform** mapping projector pixels → world. Its rectangle's aspect ratio is locked to the display resolution, so the projector shows exactly the part of the stage under its rectangle.
- Transforms are stored as **3×3 matrices** from day one. A rectangle is just a special case, so adding 4-corner warp later (phase 6b) changes no maths.

Seamless behaviour comes from three things:
1. **Cameras** are stitched into one world image; overlaps are **feathered** (weights ramp to 0 at each camera's edge and are normalised).
2. **One** `PaletteTracker` and **one** bintree cover the whole stage, so colours and triangles continue across camera and projector boundaries. Conforming splits also work across cell borders.
3. **Projectors** that overlap get **edge blending**: in the overlap each projector's light ramps down (gamma-corrected) so the sum looks uniform.

### Data flow (one frame)

```
 webcams / video files            (one capture thread each, latest frame wins)
        │
        ▼
 compositor ── remap each camera into the stage's proc grid, per-camera colour fix,
        │      feather-blend overlaps            → world input (rows·P × cols·P)
        ▼
 PaletteTracker (one palette for the stage)      → labels + palette
   or depth band (depth mode)
        ▼
 OrnamentTree.update (multi-cell bintree, CPU)   → node snapshot: amount, colour, dominant
        ▼                                          (versioned, handed to the UI thread)
 GPU renderer, per projector window:
   pixel → world (transform) → cell → descend tree → tile coverage (mip AA)
   → cross-fade splitting levels → cord / wireframe → edge blend × gain → screen
```

---

## 4. Code layout

```
bin-tree-filter/
  live.py                  # stays: quick single-camera tool (imports core/, NumPy renderer)
  perform.py               # NEW: show runner
  config.py                # stays for live.py
  core/                    # NEW: no Qt imports; used by live.py, align_perform, perform.py
    params.py              #   parameter registry (single source of truth, see §5)
    scene.py               #   Scene / Stage / CameraSpec / ProjectorSpec dataclasses, JSON v1, validation, migration
    geometry.py            #   3×3 transforms, rect ↔ matrix, stage grid
    sources.py             #   Camera/Video/Gif/Image sources (moved from live.py) + probe, MJPG/fps, reconnect
    filters.py             #   blur / sharpen / colour, depth-band labels (moved from live.py)
    compositor.py          #   cameras → world input image (remap + feather)
    posterize.py           #   PaletteTracker (moved from stable_posterize.py)
    engine.py              #   multi-cell OrnamentTree + NumPy reference renderer (from ornament_engine.py)
    blend.py               #   projector overlap / edge-blend weights
    looks.py               #   named parameter presets + crossfade
    pipeline.py            #   Pipeline(scene).step(dt) → Snapshot
  align_perform/           # NEW: the setup app (PySide6)
    __main__.py            #   python -m align_perform [scene.json]
    app.py                 #   QApplication, OpenGL format + context sharing, DPI policy
    main_window.py         #   assembles menus, toolbar, docks, status bar
    document.py            #   SceneDocument: scene + QUndoStack + change signals, dirty flag, save/load/autosave
    commands.py            #   undoable edits: AddItem, RemoveItem, Transform, SetProperty, SetParam, ApplyLook
    actions.py             #   every QAction: text, shortcut, enable rule → handler (keeps main_window small)
    canvas/
      view.py              #     zoom / pan / fit, grid, world-preview background
      scene.py             #     keeps graphics items in sync with SceneDocument
      items.py             #     StageItem, CameraItem, ProjectorItem
      handles.py           #     resize handles (and corner handles for 6b)
      snapping.py          #     snap targets + guide lines
    panels/
      scene_tree.py        #     list of stage / cameras / projectors, enable + visibility
      inspector.py         #     properties of the selected item
      look_panel.py        #     all live options, generated from core.params
      looks_panel.py       #     presets: save / load / crossfade
      status.py            #     fps, per-stage timings, camera / projector health
    dialogs/
      add_camera.py        #     webcam list (name, index, resolution, thumbnail) or file picker
      add_projector.py     #     display list (name, resolution, position) or test window
    runtime/               #   shared with perform.py
      engine_thread.py     #     runs core.pipeline off the UI thread, publishes snapshots
      gl_renderer.py       #     moderngl shader renderer
      projector_window.py  #     borderless full-screen GL window on a chosen display
      test_patterns.py     #     grid, IDs, overlap zones, white field, colour bars
      screens.py           #     enumerate displays, match saved display by name / serial
  scenes/example.json      # NEW: sample scene using video files
  tests/                   # NEW: pytest
  tools/bench.py           # NEW: performance benchmark (the numbers in §2)
```

`ornament_engine.py` and `stable_posterize.py` move into `core/` with `git mv`, so their history follows.

---

## 5. Key designs

### 5.1 Parameter registry (`core/params.py`)

Every option `live.py` has is declared once:

```python
Param("max_depth", "Max depth", group="Tree", kind="int", default=8, min=1, max=12,
      keys=("d", "D"), target="tree.max_depth")
```

The same declaration generates the UI widget (slider + spin box, toggle, dropdown, colour picker, file picker), the keyboard shortcuts (the same keys as `live.py`), the JSON field and the looks crossfade (floats interpolate, switches flip at the midpoint). Adding an option later means one new line.

Options covered: max/min depth, split threshold, hysteresis, colour hysteresis, conforming, animate, split time, colour count, palette style, tracker smoothing, label margin, blur, sharpen, brightness, contrast, saturation, colour/depth mode, depth band min/max, foreground/background presets, polarity, cord, wireframe, background colour, ornament tile (+ crop), proc resolution, ornament scale (`cell_size`), mirror, HUD.

### 5.2 Multi-cell bintree (`core/engine.py`)

The current engine covers one square. The stage is `cols × rows` squares:

- Level L has `n_sq · 2^(L+1)` nodes. Children `(L+1, 2i)` / `(L+1, 2i+1)` and ancestor `f >> (max_level − L)` stay the same; per-level offsets scale by `n_sq`. A level-0 node is `cell · 2 + lower`.
- `_descend` maps a point to its cell plus local coordinates, then runs unchanged.
- `_base_neighbours` matches hypotenuses in **global** coordinates (wider bit packing), so conforming splits cross cell borders.
- Split into `OrnamentTree` (state + `update`) and `NumpyRenderer` (lookup tables for any set of sample points + `render`). `live.py` = 1×1 stage + one NumPy renderer, which must produce exactly the same image as today.

### 5.3 GPU renderer (`align_perform/runtime/gl_renderer.py`)

- One full-screen quad per projector window. Uniforms: pixel→world matrix, stage origin, `cell_size`, cols/rows, max level, polarity, background, blend data.
- Per frame, upload the node snapshot as a float texture (RGB colour + split amount) plus the dominant colour for cord.
- Fragment shader mirrors `NumpyRenderer`: find the cell, walk down while the node is fully open (tracking the triangle's corners as `_descend` does), sample the tile at the two levels being cross-faded, mix towards background, blend by the split amount.
- Anti-aliasing: mipmapped tile texture sampled with explicit gradients.
- Cord / wireframe: pass 1 writes the leaf key, pass 2 marks edges within the cord width.
- Edge blend: for every projector covering the pixel, `w_j = smoothstep(0, feather, distance to that projector's edge)`; this projector outputs `colour · (w_i / Σw)^(1/γ) · gain`. Computed analytically, so it's correct for any layout.
- Qt integration: a `QOpenGLWindow` per projector with `moderngl.create_context()` inside it; Qt's shared contexts let the tile texture be uploaded once.

### 5.4 Camera compositor (`core/compositor.py`)

- For each camera, precompute `cv2.remap` maps from the stage's proc grid to camera pixels (inverse transform) and a feather weight map; rebuild only when that camera's transform or source changes.
- Per frame: area-downscale the frame, apply that camera's colour fix, remap, then accumulate `Σ w·image / Σ w`. Areas no camera covers become background.
- Depth mode does the same on grayscale, then applies the depth band.

### 5.5 Threads

- **Capture**: one thread per camera (today's `CameraSource` pattern) plus a watchdog that reopens a stalled camera with back-off.
- **Pipeline**: one worker thread runs compositor → posterize → tree update and publishes a versioned snapshot. NumPy releases the GIL in the heavy calls; if profiling shows contention, move it to a subprocess with shared memory (phase 7).
- **UI / GL**: the Qt main thread. Projector windows repaint on vsync with the newest snapshot. The UI never waits on the pipeline.

### 5.6 Scene file (JSON, `version: 1`)

```json
{
  "version": 1,
  "stage": {"x": 0, "y": 0, "width": 3840, "height": 1080, "cell_size": 1080},
  "cameras": [{
    "id": "cam1", "name": "Left", "enabled": true,
    "source": {"kind": "webcam", "index": 0, "device_name": "C920", "width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG"},
    "transform": {"x": 0, "y": 0, "width": 1920, "height": 1080, "rotation": 0, "mirror": true, "corners": null},
    "color": {"brightness": 0, "contrast": 1.0, "saturation": 1.0},
    "feather": 120
  }],
  "projectors": [{
    "id": "proj1", "name": "Left", "enabled": true,
    "output": {"kind": "screen", "screen_name": "\\\\.\\DISPLAY2", "serial": "", "width": 1920, "height": 1080},
    "transform": {"x": 0, "y": 0, "width": 1920, "height": 1080, "corners": null},
    "blend": {"feather": "auto", "gamma": 2.2}, "gain": 1.0
  }],
  "params": {"max_depth": 8, "threshold": 0.75, "palette_style": "shyrdak"},
  "looks": {"calm": {"max_depth": 6}, "busy": {"max_depth": 10}}
}
```

Video paths are stored relative to the scene file. Webcams are matched by device name first and index second, because Windows can renumber cameras after a reboot. Unknown fields are kept, and `version` drives migrations.

### 5.7 Align Perform UI

- **Menus / toolbar:** File (New, Open, Save, Save As, Recent), Edit (Undo, Redo, Duplicate, Delete), Add (Camera…, Projector…), View (Input / Labels / Ornament / Test pattern; fit stage; zoom), Output (projectors on/off, test pattern on projectors, identify projectors), Run (play/pause).
- **Left:** scene tree. **Centre:** canvas. **Right:** inspector, look panel and looks panel as tabs. **Bottom:** fps, per-step timings, a green/amber/red dot per camera and projector.
- **Canvas:** drag to move; 8 handles to resize, aspect-locked by default (Shift = free, Alt = from centre); snap to stage edges, the cell grid and other rectangles' edges with guide lines; arrow keys nudge 1 px (Shift = 10); wheel zooms at the cursor; middle-drag or Space-drag pans; F fits. Overlap zones are hatched. Camera rectangles show live thumbnails; the stage shows the live ornament preview.
- **Context menu:** duplicate, delete, rotate 90°, mirror, reset to native size, bring forward / send back (camera stacking order).
- **Add Camera:** list of webcams (name, index, probed resolution, thumbnail), or browse for a video/image/GIF; requested resolution, fps and MJPG.
- **Add Projector:** list of displays (name, resolution, position), or "Window (testing)" with a typed resolution, so the whole flow works on a one-monitor PC.
- Every edit goes through `commands.py`, so everything is undoable. Autosave writes a backup copy.

### 5.8 `perform.py`

```
python perform.py scenes/show.json [--look calm] [--windowed] [--wait-for-screens] [--status]
```

- Opens each projector on its saved display (matched by name/serial, falling back to index), full-screen and borderless with the cursor hidden. If a display is missing, it waits or retries (or uses a window with `--windowed`).
- Starts cameras and the pipeline. No editor.
- **Hotkeys:** `B` blackout (fade), `F` freeze, `1`–`9` looks with crossfade, `R` reload the scene, `H` status overlay, `Ctrl+Shift+Q` quit (Esc alone does nothing during a show).
- **Robustness:** a stalled camera is reopened with back-off while its area shows the last good frame and then background; the frame loop catches and logs every exception and keeps running; daily log files in `logs/`; display sleep is blocked while running; a documented autostart (Task Scheduler, or a `.bat` that restarts on exit).

---

## 6. Phases

Each phase ends in a working, tested state and gets its own commit(s) on `feature/align-perform`. `live.py` must keep working after every phase.

### Phase 0: Groundwork (small)
- [x] Settle decision 3 (Python version); if 3.12, recreate the venv and update `requirements.txt`. *(Done: Python 3.12.6, numpy 2.5.4.)*
- [x] Create `core/`: `git mv` the engine and posterizer in; move sources, filters and depth labels out of `live.py`; update imports.
- [x] `core/params.py` registry covering every `live.py` option (optionally drive `live.py`'s keys from it). *(live.py keeps its own key handling.)*
- [x] `tests/` with pytest: render shape and determinism, still frames → stable labels, conforming invariant (neighbouring leaves differ by ≤ 1 level), headless `live.py` smoke run. *(Plus golden renders from the original engine in `tests/data/engine_golden.npz`.)*
- [x] `tools/bench.py`.
- **Done when:** `live.py` behaves exactly as before and tests pass.

### Phase 1: Multi-cell engine (medium)
- [x] Generalise the bintree to `cols × rows` cells (§5.2); split into `OrnamentTree` + `NumpyRenderer`. *(Renderers take a 3×3 matrix from normalised output coords to stage cells, so homographies already work.)*
- [x] Tests: a 1×1 stage is bit-identical to today's engine; conforming holds across cell borders; three side-by-side renders equal one wide render (no seams). *(Exact except cord lines lying exactly on a tile's first column, which the NumPy renderer can't see.)*
- **Done when:** the tests pass and `live.py` looks unchanged.

### Phase 2: GPU renderer (medium–large, the main technical risk, so done early)
- [x] Start with a short spike: a moderngl context inside a PySide6 `QOpenGLWindow` on Windows; two windows sharing one texture. *(Each window has its own context and uploads the tile itself; `tests/test_qt_gl_windows.py`.)*
- [x] Shader renderer (§5.3) including cord and wireframe. *(Edge blending and gain are in too.)*
- [x] Parity tests against `NumpyRenderer` using a headless context (mean pixel difference under tolerance across states); skipped automatically if there is no GPU. *(Max 2–3 levels difference once views avoid pixel centres lying exactly on triangle edges.)*
- [x] Benchmark: 2 × 1920×1080 at ≥ 60 fps on the RTX 2070. *(1.2 ms per 1080p frame ≈ 850 fps per output; 4K 4.2 ms; with cord 2.3 / 14 ms.)*
- **Done when:** parity and the benchmark pass.

### Phase 3: Scene, compositor, pipeline, headless (medium)
- [x] `scene.py` (JSON v1, validation, relative paths, round-trip test), `geometry.py`.
- [x] `sources.py`: probe the real resolution, MJPG/fps requests, reconnect watchdog, looping video files that stay in step.
- [x] `compositor.py` (§5.4), `blend.py`, `pipeline.py` (rebuild maps only when geometry changes). *(Blend maths and RenderView builders live in `core/view.py`.)*
- [x] `tools/render_scene.py`: run a scene with video-file cameras and write each projector's output as PNG/MP4. *(Also writes `projected.png`, the outputs added as light in world space.)*
- [x] `scenes/example.json` with short test clips. *(`tools/make_test_clips.py` cuts a panorama into 3 overlapping 480×360 clips.)*
- **Done when:** the example (3 video cameras, 2 virtual projectors) renders with no visible seams.

### Phase 4: Align Perform shell + canvas (large)
- [x] App, main window, document + undo, actions, canvas items / handles / snapping / zoom / pan, scene tree, inspector.
- [x] Add Camera (names via `cv2-enumerate-cameras`) and Add Projector (displays via `QGuiApplication.screens()`) dialogs.
- [x] Engine thread; live stage preview and camera thumbnails on the canvas. *(`runtime/session.py`: the GPU renders exactly the visible world area at screen resolution; view modes Ornament / Stitched input / Labels / Camera images.)*
- [x] Open / Save / Save As / Recent, unsaved-changes prompt, autosave.
- **Done when:** you can build, edit (with undo), save and reopen a 3-camera / 2-projector scene from video files and watch it live.

### Phase 5: Look panel, every live option (medium)
- [x] Look panel generated from the registry; same hotkeys as `live.py`; per-camera colour fix in the inspector.
- [x] Looks: save, rename, delete, crossfade time. *(`core/looks.py` LookFader; a crossfade moves only the live output and becomes one undo step at the end.)*
- [x] Parity checklist: `docs/live-parity.md`.
- **Done when:** every control in the `live.py` docstring has a working UI equivalent (checklist).

### Phase 6: Projector outputs + physical alignment (medium)
- [ ] Projector windows (full-screen on a display, or a test window), pixel-exact under Windows display scaling.
- [ ] Test patterns and "identify projectors".
- [ ] Edge blending (§5.3) with gamma and per-projector gain controls.
- [ ] **6b (optional, decision 1):** 4-corner warp for projectors and cameras: corner handles on the canvas, plus fine adjustment with arrow keys while watching the projector.
- **Done when:** with real projectors, the overlap is invisible on a white field and on the ornament.

### Phase 7: `perform.py` (medium)
- [ ] Everything in §5.8.
- [ ] Soak test: 2 h with video files; unplug/replug a camera; switch a display off/on. If the GIL shows up in profiling, move the pipeline to a subprocess.
- **Done when:** the soak test runs clean and recovery works.

### Phase 8: Docs + polish (small)
- [ ] Update `CLAUDE.md`; add a README section on the workflow (align → save → perform) and a key reference.

Order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Phases 2 and 3 only share `core/engine.py`'s interface, so they can run in parallel once phase 1 lands.

---

## 7. New dependencies

| Package | Version (resolves on Py 3.10 today) | Why |
|---|---|---|
| `PySide6-Essentials` | 6.12.0 (needs Python ≥ 3.10) | UI, canvas, multi-display windows, OpenGL windows |
| `moderngl` | 5.13.0 | GPU renderer |
| `cv2-enumerate-cameras` | 1.4.0 | webcam names ↔ OpenCV indices |
| `pytest` (dev only) | 9.1.1 | tests |

`PySide6-Essentials` rather than full `PySide6` keeps the venv lean. Dev-only packages go in `requirements-dev.txt`.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| moderngl inside Qt windows on Windows; context sharing between projector windows | Spike first in phase 2; fall back to one shared offscreen context plus blitting |
| Windows display scaling (125–150 %) makes projector output not pixel-exact | Per-monitor DPI awareness; render at the physical pixel size from the screen's device pixel ratio |
| Several USB webcams exceed USB bandwidth | Request MJPG and a sensible resolution; read back the real delivered size; spread cameras across USB controllers |
| Windows renumbers webcams after a reboot | Match by device name (and path) before index |
| Cameras' auto-exposure / white balance differ, making seams visible in the input | Per-camera colour fix, one shared palette, try disabling auto-exposure (depends on the driver) |
| CPU pipeline + capture threads + UI fight over the GIL | Measure; move the pipeline to a subprocess with shared memory if needed |
| Python 3.10 end of life | Decision 3 |
