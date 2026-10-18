"""Real-time ornament filter (v2) — fast, temporally stable, seam-aware.

Uses ornament_engine.OrnamentEngine + stable_posterize.PaletteTracker.
The original pipeline (webcam.py, depth_webcam.py, ...) lives in old/ on the
chore/archive-old-pipeline branch, for comparison.

Usage:
    python live.py                     # webcam 0, colour mode
    python live.py 1                   # webcam 1
    python live.py clip.mp4            # video file (loops)
    python live.py me.jpg              # still image + simulated sensor noise
    python live.py depth.gif --depth   # depth footage, band mode
    python live.py --crop              # trim the ornament tile's white margin
    python live.py --fullscreen        # start in fullscreen
    python live.py --no-mirror         # don't mirror the webcam (mirrored by default)

Controls:
    q         quit                         s      save frame
    w / F11   toggle fullscreen            ESC    leave fullscreen, else quit
    u         toggle HUD + thumbnails      r      toggle mirror (horizontal flip)
    d / D     max depth -/+                m / M  min depth -/+
    h / H     split threshold -/+          c / C  colours -/+
    p / P     proc resolution -/+          o / O  output resolution -/+
    [ / ]     blur -/+                     ; / '  sharpen -/+
    n / N     contrast -/+                 t / T  saturation -/+
    b / B     brightness -/+    (depth mode: b cycles background colour)
    k         cycle palette style          v      toggle conforming mesh
    i         toggle animation             x      cycle ornament polarity
    g         toggle cord outline          e      toggle triangle wireframe
  depth mode:
    a / A     band min -/+                 z / Z  band max -/+
    f         cycle foreground colour
"""

from __future__ import annotations

import argparse
import os
import time

# Must be set before cv2 loads: with hardware transforms on, MSMF takes ~13 s
# to open some webcams (measured on a Logitech C920) and runs them at ~17 fps.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

from config import BACKGROUND_COLOR, ORNAMENT_FILENAME
from core.engine import POLARITIES, OrnamentEngine, load_tile
from core.filters import (BG_PRESETS, FG_PRESETS, apply_blur, apply_filters, apply_sharpen,
                          centre_square, depth_labels)
from core.posterize import PALETTE_STYLES, PaletteTracker
from core.sources import CameraSource, open_source


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

WINDOW = "Oimo-chiyme v2"
F11_KEYS = (0x7A0000, 65480)   # cv2.waitKeyEx codes: Windows, GTK


def set_fullscreen(on: bool, windowed_size: int) -> None:
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
    if not on:
        cv2.resizeWindow(WINDOW, windowed_size, windowed_size)


def compose(output: np.ndarray, fullscreen: bool, windowed_size: int) -> tuple[np.ndarray, int]:
    """Fit the square output into the window, letterboxed in fullscreen.

    Returns the canvas and the side length of the square image on it.
    """
    w = h = windowed_size
    if fullscreen:
        _, _, rw, rh = cv2.getWindowImageRect(WINDOW)
        if rw > 0 and rh > 0:
            w, h = rw, rh
    side = min(w, h)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - side) // 2, (h - side) // 2
    canvas[y0 : y0 + side, x0 : x0 + side] = cv2.resize(output, (side, side), interpolation=cv2.INTER_LINEAR)
    return canvas, side


def put_text(img: np.ndarray, txt: str, x: int, y: int, scale: float = 0.65) -> None:
    """White text with a black outline, readable on any background."""
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def paste_thumb(display: np.ndarray, img: np.ndarray, x: int, y: int, size: int, label: str) -> None:
    display[y - 1 : y + size + 1, x - 1 : x + size + 1] = (255, 255, 255)
    display[y : y + size, x : x + size] = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(display, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="webcam index, video, gif or image (default: webcam 0)")
    ap.add_argument("--depth", action="store_true", help="depth-band mode instead of colour posterize")
    ap.add_argument("--out", type=int, default=512, help="render resolution")
    ap.add_argument("--proc", type=int, default=128, help="statistics resolution")
    ap.add_argument("--levels", type=int, default=12, help="deepest precomputed level")
    ap.add_argument("--display", type=int, default=900, help="window size")
    ap.add_argument("--crop", action="store_true", help="trim the ornament tile's white margin")
    ap.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                    help="flip the input horizontally (default: on for webcams, off for files)")
    args = ap.parse_args()

    source = open_source(args.source)
    if not source.is_opened:
        print(f"Error: cannot open source {args.source!r}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tile = load_tile(os.path.join(script_dir, ORNAMENT_FILENAME), crop=args.crop)

    t0 = time.perf_counter()
    engine = OrnamentEngine(tile, out_size=args.out, proc_size=args.proc, max_level=args.levels,
                            background=BACKGROUND_COLOR)
    print(f"Precomputed lookup tables in {time.perf_counter() - t0:.2f} s")

    tracker = PaletteTracker(k=4)
    styles = list(PALETTE_STYLES)

    # Tuneable parameters
    if args.depth:
        engine.threshold = 0.95
    blur_size, sharpen_amount = 0, 0.0
    brightness, contrast, saturation = 0, 1.0, 1.0
    depth_min, depth_max = 60, 140
    fg_idx, bg_idx = 0, 0
    cord, wireframe = False, False
    fullscreen, show_hud = args.fullscreen, True
    mirror = isinstance(source, CameraSource) if args.mirror is None else args.mirror

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.display, args.display)
    if fullscreen:
        set_fullscreen(True, args.display)

    frame_count, fps = 0, 0.0
    t_fps = t_prev = time.perf_counter()
    output = None

    while True:
        ok, frame = source.read()
        if not ok:
            break
        now = time.perf_counter()
        dt = min(max(now - t_prev, 1 / 240), 0.25)
        t_prev = now
        timings = {}

        if mirror:
            frame = cv2.flip(frame, 1)
        square = centre_square(frame)

        t = time.perf_counter()
        if args.depth:
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            labels = depth_labels(gray, depth_min, depth_max, engine.proc_size)
            fg = FG_PRESETS[fg_idx % len(FG_PRESETS)]
            bg = BG_PRESETS[bg_idx % len(BG_PRESETS)]
            engine.background = np.array(bg, np.float32)
            palette = np.array([bg, fg], np.float32)
            preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            post_img = palette[labels].astype(np.uint8)
            timings["band"] = (time.perf_counter() - t) * 1000
        else:
            small = cv2.resize(square, (engine.proc_size, engine.proc_size), interpolation=cv2.INTER_AREA)
            small = apply_blur(small, blur_size)
            small = apply_sharpen(small, sharpen_amount)
            preview = apply_filters(small, brightness, contrast, saturation)
            timings["filt"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            labels, palette = tracker.update(preview, dt)
            post_img = palette[labels].astype(np.uint8)
            timings["post"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        engine.update(labels, palette, dt)
        timings["tree"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        output = engine.render(cord=cord, wireframe=wireframe)
        timings["rend"] = (time.perf_counter() - t) * 1000
        total_ms = (time.perf_counter() - now) * 1000

        # --- Display ---
        display, side = compose(output, fullscreen, args.display)

        frame_count += 1
        if now - t_fps >= 0.5:
            fps = frame_count / (now - t_fps)
            frame_count, t_fps = 0, now

        if show_hud:
            h, w = display.shape[:2]
            thumb = side // 5
            by = h - thumb - 8
            paste_thumb(display, post_img, w - thumb - 8, by, thumb, "Labels")
            paste_thumb(display, preview, w - 2 * thumb - 14, by, thumb, "Input")

        lines = [
            f"FPS: {fps:.1f}   total: {total_ms:.0f} ms   leaves: {engine.leaf_count}",
            f"Depth: {engine.min_depth}-{engine.max_depth}  Thresh: {engine.threshold:.2f}  "
            + (f"Band: {depth_min}-{depth_max}" if args.depth else f"Colors: {tracker.k}  Style: {tracker.style}"),
            f"proc={engine.proc_size}  out={engine.out_size}  polarity={engine.polarity}",
            f"conform={'on' if engine.conforming else 'off'}  anim={'on' if engine.animate else 'off'}  "
            f"cord={'on' if cord else 'off'}  mirror={'on' if mirror else 'off'}",
            "  ".join(f"{k}:{v:.1f}" for k, v in timings.items()) + " ms",
        ]
        if show_hud:
            for i, txt in enumerate(lines):
                put_text(display, txt, 10, 25 + i * 24, scale=0.55)

        cv2.imshow(WINDOW, display)

        # --- Key handling ---
        key_ex = cv2.waitKeyEx(1)
        key = key_ex & 0xFF
        if key == ord("q") or (key == 27 and not fullscreen):
            break
        elif key_ex in F11_KEYS or key == ord("w") or (key == 27 and fullscreen):
            fullscreen = not fullscreen
            set_fullscreen(fullscreen, args.display)
        elif key == ord("u"):
            show_hud = not show_hud
        elif key == ord("r"):
            mirror = not mirror
        elif key == ord("d"):
            engine.max_depth = max(engine.min_depth + 1, engine.max_depth - 1)
        elif key == ord("D"):
            engine.max_depth = min(engine.max_level, engine.max_depth + 1)
        elif key == ord("m"):
            engine.min_depth = max(0, engine.min_depth - 1)
        elif key == ord("M"):
            engine.min_depth = min(engine.max_depth - 1, engine.min_depth + 1)
        elif key == ord("h"):
            engine.threshold = max(0.01, round(engine.threshold - 0.01, 2))
        elif key == ord("H"):
            engine.threshold = min(1.0, round(engine.threshold + 0.01, 2))
        elif key == ord("c") and not args.depth:
            tracker.k = max(2, tracker.k - 1); tracker.reset()
        elif key == ord("C") and not args.depth:
            tracker.k = min(8, tracker.k + 1); tracker.reset()
        elif key == ord("p"):
            engine.set_proc_size(max(32, engine.proc_size - 32)); tracker.reset()
        elif key == ord("P"):
            engine.set_proc_size(min(512, engine.proc_size + 32)); tracker.reset()
        elif key == ord("o"):
            engine.set_output(max(128, engine.out_size - 128))
        elif key == ord("O"):
            engine.set_output(min(1024, engine.out_size + 128))
        elif key == ord("x"):
            engine.set_output(engine.out_size, polarity=POLARITIES[(POLARITIES.index(engine.polarity) + 1) % len(POLARITIES)])
        elif key == ord("k"):
            tracker.style = styles[(styles.index(tracker.style) + 1) % len(styles)]
        elif key == ord("v"):
            engine.conforming = not engine.conforming
        elif key == ord("i"):
            engine.animate = not engine.animate
        elif key == ord("g"):
            cord = not cord
        elif key == ord("e"):
            wireframe = not wireframe
        elif key == ord("["):
            blur_size = max(0, blur_size - 1)
        elif key == ord("]"):
            blur_size += 1
        elif key == ord(";"):
            sharpen_amount = max(0.0, round(sharpen_amount - 0.2, 1))
        elif key == ord("'"):
            sharpen_amount = round(sharpen_amount + 0.2, 1)
        elif key == ord("n"):
            contrast = round(contrast - 0.1, 2)
        elif key == ord("N"):
            contrast = round(contrast + 0.1, 2)
        elif key == ord("t"):
            saturation = max(0.0, round(saturation - 0.1, 2))
        elif key == ord("T"):
            saturation = round(saturation + 0.1, 2)
        elif key == ord("b"):
            if args.depth:
                bg_idx += 1
            else:
                brightness -= 10
        elif key == ord("B") and not args.depth:
            brightness += 10
        elif key == ord("f") and args.depth:
            fg_idx += 1
        elif key == ord("a") and args.depth:
            depth_min = max(0, depth_min - 5)
        elif key == ord("A") and args.depth:
            depth_min = min(depth_max - 5, depth_min + 5)
        elif key == ord("z") and args.depth:
            depth_max = max(depth_min + 5, depth_max - 5)
        elif key == ord("Z") and args.depth:
            depth_max = min(255, depth_max + 5)
        elif key == ord("s") and output is not None:
            fname = f"ornament_v2_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
