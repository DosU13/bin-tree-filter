"""Kyrgyz ornamental pattern filter for depth camera footage.

A depth band [depth_min, depth_max] (0–255) selects a slice of the scene.
Pixels inside the band → FOREGROUND colour.
Pixels outside the band → BACKGROUND colour.
This binary two-colour image drives the ornament triangle tree — no posterize needed.

Usage:
    python depth_webcam.py                  # real depth camera (device 0)
    python depth_webcam.py depth.gif        # test with GIF file
    python depth_webcam.py 1                # real depth camera (device 1)

Controls:
    q / ESC  — quit
    a / A    — depth_min   -5 / +5
    z / Z    — depth_max   -5 / +5
    d / D    — tree max_depth  -1 / +1
    m / M    — tree min_depth  -1 / +1
    h / H    — homogeneity threshold  -0.01 / +0.01
    p / P    — proc resolution  -32 / +32
    o / O    — output resolution  -64 / +64
    f        — cycle foreground colour
    b        — cycle background colour
    s        — save current output frame
"""

import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

from config import ORNAMENT_FILENAME
from renderer import load_ornament_mask, render
from triangle_tree import build_tree, collect_leaves, make_initial_triangles


# ---------------------------------------------------------------------------
# Colour presets
# ---------------------------------------------------------------------------

FG_PRESETS = [
    (0,   0,   160),   # dark red
    (120, 30,  0),     # dark navy
    (0,   100, 10),    # dark green
    (30,  0,   130),   # dark maroon
    (20,  20,  20),    # near-black
]

BG_PRESETS = [
    (245, 235, 210),   # warm cream
    (255, 245, 220),   # light gold
    (255, 255, 255),   # white
    (220, 235, 245),   # cool white
]


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class GifSource:
    """Iterates frames from a GIF file, looping indefinitely."""

    def __init__(self, path: str) -> None:
        self._gif = Image.open(path)
        self._n_frames = self._gif.n_frames
        self._idx = 0

    def read(self) -> tuple[bool, np.ndarray]:
        self._gif.seek(self._idx % self._n_frames)
        self._idx += 1
        gray = np.array(self._gif.convert("L"))
        return True, gray

    def release(self) -> None:
        self._gif.close()


class CameraSource:
    """Reads frames from a cv2.VideoCapture and converts to grayscale."""

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index)

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            return False, np.zeros((1, 1), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return True, gray

    def release(self) -> None:
        self._cap.release()


def open_source(arg: str | None):
    """Return the appropriate frame source for the given argument."""
    if arg is None:
        src = CameraSource(0)
        if not src.is_opened:
            print("Error: cannot open camera device 0")
            sys.exit(1)
        return src

    if arg.lower().endswith(".gif"):
        return GifSource(arg)

    try:
        index = int(arg)
        src = CameraSource(index)
        if not src.is_opened:
            print(f"Error: cannot open camera device {index}")
            sys.exit(1)
        return src
    except ValueError:
        # Treat as a video file path
        src = CameraSource.__new__(CameraSource)
        src._cap = cv2.VideoCapture(arg)
        if not src._cap.isOpened():
            print(f"Error: cannot open file {arg}")
            sys.exit(1)
        return src


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------

def depth_to_two_color(
    gray: np.ndarray,
    depth_min: int,
    depth_max: int,
    fg_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    """Map a grayscale depth image to a binary two-colour BGR image.

    Pixels in [depth_min, depth_max] → fg_color.
    All other pixels                 → bg_color.
    """
    mask = (gray >= depth_min) & (gray <= depth_max)
    result = np.full((*gray.shape, 3), bg_color, dtype=np.uint8)
    result[mask] = fg_color
    return result


def make_depth_thumbnail(
    gray: np.ndarray,
    depth_min: int,
    depth_max: int,
    size: int,
) -> np.ndarray:
    """Build a thumbnail showing the depth map with the active band highlighted.

    In-band pixels are tinted yellow; out-of-band pixels stay grayscale.
    """
    thumb_gray = cv2.resize(gray, (size, size))
    bgr = cv2.cvtColor(thumb_gray, cv2.COLOR_GRAY2BGR)

    mask = (thumb_gray >= depth_min) & (thumb_gray <= depth_max)
    # Yellow tint on foreground band
    bgr[mask] = np.clip(
        bgr[mask].astype(np.int32) + np.array([0, 60, 80], dtype=np.int32),
        0, 255,
    ).astype(np.uint8)
    return bgr


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _scale_leaves(leaves, scale: float) -> None:
    """Scale all leaf triangle vertices in-place."""
    if scale == 1.0:
        return
    for leaf in leaves:
        leaf.right_angle = (leaf.right_angle[0] * scale, leaf.right_angle[1] * scale)
        leaf.hyp_a       = (leaf.hyp_a[0]       * scale, leaf.hyp_a[1]       * scale)
        leaf.hyp_b       = (leaf.hyp_b[0]       * scale, leaf.hyp_b[1]       * scale)


def process_frame(
    gray: np.ndarray,
    proc_size: int,
    out_size: int,
    depth_min: int,
    depth_max: int,
    max_depth: int,
    min_depth: int,
    threshold: float,
    fg_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    ornament_mask: np.ndarray,
    ornament_mask_inv: np.ndarray,
    src_tri: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Run the depth ornament pipeline on one grayscale frame.

    Pipeline:
        gray → centre-crop square → resize(proc_size)
             → depth_to_two_color
             → build_tree
             → scale vertices to out_size
             → render

    Returns:
        output:  Rendered ornament image (out_size × out_size BGR).
        timings: Dict of per-step milliseconds.
    """
    timings = {}

    # Centre-crop to square
    h, w = gray.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    square = gray[y0 : y0 + s, x0 : x0 + s]
    square = cv2.resize(square, (proc_size, proc_size), interpolation=cv2.INTER_NEAREST)

    # Two-colour image
    t0 = time.perf_counter()
    two_color = depth_to_two_color(square, depth_min, depth_max, fg_color, bg_color)
    timings["band"] = (time.perf_counter() - t0) * 1000

    # Build tree
    t0 = time.perf_counter()
    roots = make_initial_triangles(proc_size)
    all_leaves = []
    for root in roots:
        build_tree(two_color, root, max_depth, threshold, min_depth)
        all_leaves.extend(collect_leaves(root))
    timings["tree"] = (time.perf_counter() - t0) * 1000

    # Render at out_size
    t0 = time.perf_counter()
    _scale_leaves(all_leaves, out_size / proc_size)
    output = np.full((out_size, out_size, 3), bg_color, dtype=np.uint8)
    render(output, all_leaves, ornament_mask, ornament_mask_inv, src_tri)
    timings["render"] = (time.perf_counter() - t0) * 1000

    return output, timings


# ---------------------------------------------------------------------------
# HUD helper
# ---------------------------------------------------------------------------

def _put_text(img: np.ndarray, txt: str, x: int, y: int, scale: float = 0.65) -> None:
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    source_arg = sys.argv[1] if len(sys.argv) > 1 else None
    source = open_source(source_arg)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ornament_path = os.path.join(script_dir, ORNAMENT_FILENAME)
    ornament_mask, ornament_mask_inv, src_tri = load_ornament_mask(ornament_path)

    # Tuneable parameters
    depth_min    = 60
    depth_max    = 140
    proc_size    = 128
    out_size     = 512
    max_depth    = 8
    min_depth    = 0
    threshold    = 0.95
    display_size = 800
    fg_idx       = 0
    bg_idx       = 0

    print("Depth ornament filter running.")
    print("  a/A   depth_min  -5/+5")
    print("  z/Z   depth_max  -5/+5")
    print("  d/D   tree max_depth  -1/+1")
    print("  m/M   tree min_depth  -1/+1")
    print("  h/H   threshold  -0.01/+0.01")
    print("  p/P   proc resolution  -32/+32")
    print("  o/O   output resolution  -64/+64")
    print("  f     cycle foreground colour")
    print("  b     cycle background colour")
    print("  s     save frame  |  q/ESC  quit")

    frame_count = 0
    fps = 0.0
    t_fps = time.perf_counter()
    last_timings: dict = {}
    last_gray: np.ndarray = np.zeros((10, 10), dtype=np.uint8)

    while True:
        ret, gray = source.read()
        if not ret:
            break
        last_gray = gray

        fg_color = FG_PRESETS[fg_idx % len(FG_PRESETS)]
        bg_color = BG_PRESETS[bg_idx % len(BG_PRESETS)]

        t0 = time.perf_counter()
        output, last_timings = process_frame(
            gray, proc_size, out_size,
            depth_min, depth_max,
            max_depth, min_depth, threshold,
            fg_color, bg_color,
            ornament_mask, ornament_mask_inv, src_tri,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        # Upscale for display
        display = cv2.resize(output, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        # Depth thumbnail — bottom-right
        thumb_size = display_size // 4
        bx = display_size - thumb_size - 8
        by = display_size - thumb_size - 8
        thumb = make_depth_thumbnail(last_gray, depth_min, depth_max, thumb_size)
        display[by - 1 : by + thumb_size + 1, bx - 1 : bx + thumb_size + 1] = (255, 255, 255)
        display[by : by + thumb_size, bx : bx + thumb_size] = thumb
        cv2.putText(display, "Depth", (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(display, "Depth", (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # FPS
        frame_count += 1
        elapsed = time.perf_counter() - t_fps
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_fps = time.perf_counter()

        # HUD
        fg_hex = "#{:02x}{:02x}{:02x}".format(fg_color[2], fg_color[1], fg_color[0])
        bg_hex = "#{:02x}{:02x}{:02x}".format(bg_color[2], bg_color[1], bg_color[0])
        lines = [
            f"FPS: {fps:.1f}   total: {total_ms:.0f} ms",
            f"Band: {depth_min}–{depth_max}   tree: {min_depth}–{max_depth}   thresh: {threshold:.2f}",
            f"Proc: {proc_size}   Out: {out_size}",
            f"FG: {fg_hex}  BG: {bg_hex}",
        ]
        for i, txt in enumerate(lines):
            _put_text(display, txt, 10, 25 + i * 26)

        if last_timings:
            parts = "  ".join(f"{k}:{v:.0f}" for k, v in last_timings.items())
            _put_text(display, parts + " ms", 10, 25 + len(lines) * 26, scale=0.5)

        cv2.imshow("Depth Ornament", display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            depth_min = max(0, depth_min - 5)
        elif key == ord("A"):
            depth_min = min(depth_max - 5, depth_min + 5)
        elif key == ord("z"):
            depth_max = max(depth_min + 5, depth_max - 5)
        elif key == ord("Z"):
            depth_max = min(255, depth_max + 5)
        elif key == ord("d"):
            max_depth = max(min_depth + 1, max_depth - 1)
        elif key == ord("D"):
            max_depth = min(15, max_depth + 1)
        elif key == ord("m"):
            min_depth = max(0, min_depth - 1)
        elif key == ord("M"):
            min_depth = min(max_depth - 1, min_depth + 1)
        elif key == ord("h"):
            threshold = max(0.01, round(threshold - 0.01, 2))
        elif key == ord("H"):
            threshold = min(1.0,  round(threshold + 0.01, 2))
        elif key == ord("p"):
            proc_size = max(32, proc_size - 32)
        elif key == ord("P"):
            proc_size = min(512, proc_size + 32)
        elif key == ord("o"):
            out_size = max(64, out_size - 64)
        elif key == ord("O"):
            out_size = min(1024, out_size + 64)
        elif key == ord("f"):
            fg_idx += 1
        elif key == ord("b"):
            bg_idx += 1
        elif key == ord("s"):
            fname = f"depth_capture_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
