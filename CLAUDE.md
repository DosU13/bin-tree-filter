# bin-tree-filter

Real-time "Kyrgyz ornament" filter. Cameras (webcams, or video / GIF / image files for testing) are stitched into one stage, posterized, and turned into a mosaic of ornament tiles laid out on an adaptive binary triangle tree (bintree), rendered on the GPU across any number of blended projectors. Big flat areas get large triangles; detail gets small ones.

Programs: `live.py` (original single-camera tool), `python -m align_perform` (setup editor), `perform.py` (show runner). User-facing docs: `README.md`. Design and history of the build: `docs/perform-plan.md`. `docs/live-parity.md` maps every live.py control to the new apps.

## Setup

- Windows, Python **3.12** venv in `.venv/` (not committed). Python 3.10 is end of life and PySide6 ≥ 6.12 needs ≥ 3.10.
- Runtime: numpy, opencv-python (not `-headless`: live.py uses `cv2.imshow`), pillow, PySide6-Essentials, moderngl, cv2-enumerate-cameras. Dev: pytest.
- Needs an OpenGL 3.3 GPU for projector output and previews (developed on an RTX 2070).

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest                # ~130 tests, ~75 s; gui tests open real windows briefly
.venv\Scripts\python -m pytest -m "not gui"   # headless subset
```

## Layout

| Path | Role |
|---|---|
| `core/` | Qt-free runtime shared by all three programs |
| `core/engine.py` | `OrnamentTree` (multi-cell bintree decisions + animation), `NumpyRenderer` (CPU reference renderer, lookup tables), `OrnamentEngine` (1×1 + NumPy renderer, live.py API), `TreeSnapshot`, `load_tile` |
| `core/posterize.py` | `PaletteTracker`: temporally stable k-means, optional fit mask, palette styles |
| `core/params.py` | Registry of every tunable option: UI widgets, hotkeys, JSON, crossfades are generated from it |
| `core/scene.py` | Scene model (stage, cameras, projectors, params, looks) ↔ JSON v1 |
| `core/geometry.py` | 3×3 transforms: rects, 4-corner warps, camera rotation / mirror, quad distance |
| `core/sources.py` | live.py sources + scene sources (`LiveSource` thread with pacing and reconnect, `SyncSource`, probe, webcam list) |
| `core/compositor.py` | Stitch cameras into the stage at proc resolution, feathered overlaps, per-camera colour fix, depth band |
| `core/pipeline.py` | `Pipeline` (one frame per `step`) and `PipelineRunner` (background thread, never stops on errors) |
| `core/view.py` | `RenderView` / `BlendSpec` and builders: projector, stage and world views, edge-blend maths |
| `core/looks.py` | `LookFader` crossfades |
| `align_perform/` | PySide6 editor: `main_window.py`, `document.py` (scene + undo), `commands.py`, `actions.py` (all actions as a table), `canvas/` (view, scene, items, handles, snapping), `panels/` (scene tree, inspector, look, looks, status), `dialogs/` (add camera / projector) |
| `align_perform/runtime/` | Shared with perform.py: `gl_renderer.py` (moderngl shader renderer, `OffscreenGL`), `qt_gl.py`, `session.py` (pipeline + GPU canvas preview), `outputs.py` + `projector_window.py` (projector windows), `test_patterns.py`, `screens.py` |
| `perform.py` | Show runner: outputs, look keys, blackout, watchdog, logs, keep-awake |
| `live.py`, `config.py` | Original tool (unchanged behaviour; imports `core`) |
| `scenes/example.json`, `scenes/media/` | 3 overlapping test clips cut from one panorama + 2 test-window projectors |
| `tools/` | `render_scene` (headless run + simulated projection), `bench` (`--gpu`), `make_test_clips`, `make_engine_golden`, `perform_loop.bat` |
| `tests/` | pytest; `tests/data/engine_golden.npz` = renders of the original engine |

## Data flow

Capture threads (one per camera, `LiveSource`) → pipeline thread (`PipelineRunner`: compositor → filters → `PaletteTracker` or depth band → `OrnamentTree.update` → `TreeSnapshot`) → Qt main thread: `LiveSession` polls `runner.latest()` for the canvas preview (GPU offscreen, exactly the visible world rect at screen resolution), `ProjectorOutputs` pulls the newest snapshot every 15 ms and each `ProjectorWindow` renders it with `GLRenderer`. Edits go document → `LiveSession.set_scene` / `ProjectorOutputs.set_scene` (debounced 40 ms); the runner applies queued changes between frames.

## Coordinate spaces

- **world**: scene canvas pixels, y down. Cameras / projectors / stage are placed here.
- **cell**: `(world - stage origin) / cell_size`; the stage is `cols × rows` cells, each a bintree root.
- **normalised**: an image's own u, v in 0..1. Renderers take `norm_to_cell` (3×3, affine or homography); `projector_view` = `stage.world_to_cell() @ transform.norm_to_world()`.
- Camera frames: `geometry.source_to_display` (mirror, then rotate clockwise) → normalised display → world via the camera's transform.

## Engine essentials

- Node indexing: with `n_sq = cols*rows`, level L has `n_sq·2^(L+1)` nodes at offset `n_sq·(2^(L+1)-2)`; children of (L,i) are (L+1,2i),(L+1,2i+1); level-0 node `2c`/`2c+1` = upper-right / lower-left half of cell `c = y*cols + x`; ancestor of finest f at level L is `f >> (max_level-L)`. Each triangle splits from its right angle to the hypotenuse midpoint.
- `update`: bincount histograms (EMA `tau_hist`) → purity → split < `threshold`, merge ≥ `threshold + hysteresis`; depth clamps; parent rule; conforming closure via base neighbours matched in global coordinates (works across cells); split amounts animate one level at a time (`split_time`); colours ease (`tau_color`, `color_hysteresis`).
- Render (both renderers): deepest fully-open level per pixel, cross-fade into the next while it opens, supersampled tile coverage per level (8-bit), polarity `alternate` inverts even levels. Cord / wireframe: key edges (the GPU's key pass has a margin, so lines continue across output borders).
- The GPU shader mirrors the NumPy renderer within 2–3 levels; parity tests nudge views off exact triangle edges (pixel centres exactly on an edge legitimately go either way in float32 vs float64).
- Edge blending: each projector fades only towards its *interior* edges (edges inside another projector), `w = Π smoothstep(dist/feather)`, normalised across projectors, then `^(1/gamma)`. `core.view.blend_alpha` is the CPU twin of the shader.

## Conventions

- Google-style docstrings, type hints, `from __future__ import annotations`, `# ---` section banners; images BGR; colours in params are `#rrggbb`.
- Per-frame work vectorised in numpy; Python loops only over tree levels or cameras, never pixels.
- Import `core` before `cv2` in entry points: it sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` (otherwise some webcams take ~13 s to open).
- New option = one `Param(...)` in `core/params.py` + the code that reads it (`core.pipeline.apply_params` or a renderer's view builder).
- All scene edits in the editor go through `SceneDocument` methods (undoable); continuous gestures pass a `merge_key`.
- moderngl uniforms that GLSL optimised away must be set via `_set` in `gl_renderer.py` (it skips missing ones). Offscreen GL work is wrapped in `with ctx:` so it coexists with Qt's contexts.
- Don't regenerate `tests/data/engine_golden.npz` unless the picture is meant to change (`python -m tools.make_engine_golden`).

## History

The original pipeline (`main.py`, `webcam.py`, …) built a Python triangle tree and re-ran k-means every frame; it is archived under `old/` on the `chore/archive-old-pipeline` branch. `.claude/settings.local.json` and `.claude/worktrees/` are machine-local; don't commit them.
