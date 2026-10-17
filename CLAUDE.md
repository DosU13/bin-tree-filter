# bin-tree-filter

Real-time "Kyrgyz ornament" filter. It turns a webcam, video, GIF or still image (or depth-camera footage) into a mosaic of ornament tiles laid out on an adaptive binary triangle tree (bintree). Big flat areas of one colour get large triangles; detailed areas get split into smaller ones. The OpenCV window is titled "Oimo-chiyme v2".

## Setup

- Windows, Python **3.10** venv in `.venv/` (not committed). numpy 2.3+ needs Python 3.11+, which is why `requirements.txt` pins `numpy~=2.2.6`.
- Dependencies are only numpy, opencv-python and pillow. Keep `opencv-python` (not `-headless`): `live.py` needs `cv2.imshow`.

```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Running

```
.venv\Scripts\python live.py                     # webcam 0 (mirrored)
.venv\Scripts\python live.py me.jpg              # still image + simulated sensor noise
.venv\Scripts\python live.py depth.gif --depth   # depth footage, band mode
```

Flags: `--out` render size (512), `--proc` statistics size (128), `--levels` deepest precomputed level (12), `--display` window size (900), `--crop` trim the tile's white margin, `--fullscreen`, `--mirror/--no-mirror`. Keyboard controls are listed in the `live.py` docstring; `s` saves `ornament_v2_<timestamp>.png` to the working directory.

There are no tests. To check a change without a display, import `live` and drive `PaletteTracker.update` → `OrnamentEngine.update` → `OrnamentEngine.render` on `me.jpg` / `depth.gif` frames (colour and `--depth` paths), writing the result with `cv2.imwrite`.

## Files

| File | Role |
|---|---|
| `live.py` | CLI and main loop: frame sources, pre-filters, depth-band labels, HUD, keyboard, fullscreen |
| `ornament_engine.py` | `OrnamentEngine`: bintree decisions + rendering, `load_tile` |
| `stable_posterize.py` | `PaletteTracker`: temporally stable k-means posterizer, palette styles |
| `config.py` | `ORNAMENT_FILENAME`, `BACKGROUND_COLOR` (BGR) |
| `ornam.jpg` | The ornament tile (black ink on white) |
| `ornam_or.jpg` | Alternate tile with a white margin (needs `--crop`) |
| `me.jpg`, `aziz.jpg`, `test_girl_sq.jpg`, `depth.gif` | Sample inputs (`depth.gif` is ~98 MB) |

## Per-frame pipeline (`live.py`)

1. Read frame (`CameraSource` reads on a background thread; `VideoSource`/`GifSource` loop; `ImageSource` adds noise and 1-px jitter so a still behaves like a webcam), optional mirror, centre-crop to a square.
2. **Colour mode:** resize to `proc_size` (INTER_AREA) → blur → sharpen → brightness/contrast/saturation → `PaletteTracker.update` → `labels` (proc×proc ints) + `palette` (k×3 BGR).
   **Depth mode:** grayscale → band mask `[depth_min, depth_max]` area-downsampled and majority-thresholded → 2 labels; palette = `[BG_PRESETS[i], FG_PRESETS[j]]`.
3. `engine.update(labels, palette, dt)` then `engine.render(cord, wireframe)` → `out_size`² BGR image, letterboxed into the window.

## OrnamentEngine (`ornament_engine.py`)

The triangle hierarchy is fixed geometry, so everything geometric is precomputed and a frame costs the same regardless of leaf count:

- `_finest_proc` / `_finest_out`: finest-level node of every proc / output pixel (`_descend`).
- `_coverage[level, pixel]`: uint8 anti-aliased ink coverage if that pixel's level-L triangle were a leaf, with polarity baked in. Rebuilt by `set_output` (~0.5–2 s); `set_proc_size` rebuilds the proc lookup.
- Node indexing: level L has `2**(L+1)` nodes; children of (L, i) are (L+1, 2i) and (L+1, 2i+1); the level-L ancestor of finest node f is `f >> (max_level - L)`. Per-node arrays are flat, level L starting at offset `2**(L+1) - 2`. Level 0: node 0 = upper-right half, node 1 = lower-left half.
- Each triangle splits from its right angle to its hypotenuse midpoint.

`update`: one `bincount` gives per-finest-cell colour histograms (EMA, `tau_hist`), summed up the levels → purity of the dominant colour → split when purity < `threshold`, merge only at ≥ `threshold + hysteresis`; clamp to `min_depth`/`max_depth`; a child can only split if its parent does. With `conforming` on, splitting a triangle forces its base (hypotenuse) neighbour to split, so neighbouring leaves differ by at most one level. Split amounts animate one level at a time (`split_time`); node colours ease towards the dominant palette colour (`tau_color`, with `color_hysteresis`).

`render`: per finest cell, find the deepest fully-open level and cross-fade into the next level while it opens; gather coverage per output pixel and mix `background` → node colour. `cord` draws background-coloured seams between leaves of different colour; `wireframe` outlines every leaf. Polarity: `alternate` (inverts even levels), `none`, `invert`.

All time constants are in seconds and go through `_blend(dt, tau)`, so behaviour is frame-rate independent.

## PaletteTracker (`stable_posterize.py`)

Runs k-means once (`cv2.kmeans`, Lab space, centres sorted by lightness), then per frame does one Lloyd step from the previous centres blended in with an EMA (`tau`), so cluster identities never swap. A pixel keeps its previous label unless another centre is closer by `label_margin` Lab units (kills boundary flicker). Dead clusters are re-seeded at the worst-fit pixel. `style` maps clusters onto a dark→light hex ramp by lightness rank (`PALETTE_STYLES`). Call `reset()` after changing `k` or the proc size.

## Conventions

- Google-style docstrings, type hints, `from __future__ import annotations`, `# ---` section banners.
- Images are BGR (OpenCV order); hex colours in `PALETTE_STYLES` are RGB and converted.
- Keep per-frame work vectorised in numpy; Python loops only over tree levels, never over leaves or pixels.
- `live.py` sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before importing cv2: with hardware transforms on, MSMF takes ~13 s to open some webcams. Camera open falls back to `CAP_DSHOW`.

## History

The original pipeline (`main.py`, `image_editor.py`, `webcam.py`, `depth_webcam.py`, `posterize.py`, `renderer.py`, `triangle_tree.py`) built a Python triangle tree and re-ran k-means on every frame, which was slow and flickered. It is archived under `old/` on the `chore/archive-old-pipeline` branch for comparison.

`.claude/settings.local.json` and `.claude/worktrees/` are machine-local; don't commit them.
