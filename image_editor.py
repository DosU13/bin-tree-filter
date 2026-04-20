"""Interactive image editor — Kyrgyz ornamental pattern.

Loads a static image, lets you tune all parameters live, then save the result.

Usage:
    python image_editor.py [input_image]

If no input is given, defaults to me.jpg.

Controls:
    q / ESC   — quit
    d / D     — decrease / increase max depth
    m / M     — decrease / increase min depth
    c / C     — decrease / increase number of colours
    p / P     — decrease / increase proc resolution (tree building)
    o / O     — decrease / increase output resolution (render quality)
    b / B     — decrease / increase brightness
    n / N     — decrease / increase contrast
    t / T     — decrease / increase saturation
    s         — save output image
"""

import os
import sys
import time

import cv2
import numpy as np

from config import BACKGROUND_COLOR
from posterize import posterize
from triangle_tree import make_initial_triangles, build_tree, collect_leaves
from renderer import load_ornament_mask, render
from webcam import apply_filters, _scale_leaves, _put_text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_image(
    image: np.ndarray,
    proc_size: int,
    out_size: int,
    num_colors: int,
    max_depth: int,
    min_depth: int,
    threshold: float,
    brightness: int,
    contrast: float,
    saturation: float,
    ornament_mask: np.ndarray,
    ornament_mask_inv: np.ndarray,
    src_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the full ornament pipeline on a pre-loaded square BGR image.

    Returns:
        output:     Rendered ornament image (out_size x out_size).
        filtered:   Image after calibration filters.
        posterized: Image after posterize.
        timings:    Per-step milliseconds.
    """
    timings = {}

    # Step 1 — calibration filters
    t0 = time.perf_counter()
    filtered = apply_filters(image, brightness, contrast, saturation)
    timings["filter"] = (time.perf_counter() - t0) * 1000

    # Step 2 — posterize
    t0 = time.perf_counter()
    post, _ = posterize(filtered, num_colors)
    timings["post"] = (time.perf_counter() - t0) * 1000

    # Step 3 — downscale to proc_size for tree building
    t0 = time.perf_counter()
    src_h = image.shape[0]
    if proc_size != src_h:
        post_proc = cv2.resize(post, (proc_size, proc_size), interpolation=cv2.INTER_NEAREST)
    else:
        post_proc = post
    timings["resize"] = (time.perf_counter() - t0) * 1000

    # Step 4 — build tree
    t0 = time.perf_counter()
    roots = make_initial_triangles(proc_size)
    all_leaves = []
    for root in roots:
        build_tree(post_proc, root, max_depth, threshold, min_depth)
        all_leaves.extend(collect_leaves(root))
    timings["tree"] = (time.perf_counter() - t0) * 1000

    # Step 5 — scale vertices + render at out_size
    t0 = time.perf_counter()
    _scale_leaves(all_leaves, out_size / proc_size)
    output = np.full((out_size, out_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    render(output, all_leaves, ornament_mask, ornament_mask_inv, src_tri)
    timings["render"] = (time.perf_counter() - t0) * 1000

    return output, filtered, post, timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_and_square(path: str) -> np.ndarray:
    """Load image and centre-pad to square with white fill."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    h, w = img.shape[:2]
    size = max(h, w)
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img
    return out


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "me.jpg")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    ornament_path = os.path.join(script_dir, "ornam.jpg")
    ornament_mask, ornament_mask_inv, src_tri = load_ornament_mask(ornament_path)

    source = _load_and_square(input_path)
    print(f"Loaded: {input_path}  ({source.shape[1]}x{source.shape[0]})")

    # Parameters
    proc_size  = 128
    out_size   = 512
    num_colors = 4
    max_depth  = 8
    min_depth  = 0
    threshold  = 0.75
    display_size = 800
    brightness = 0
    contrast   = 1.0
    saturation = 1.0

    print("Image editor running.  Controls:")
    print("  q/ESC  quit")
    print("  d/D    max depth -/+")
    print("  m/M    min depth -/+")
    print("  h/H    homogeneity threshold -/+")
    print("  c/C    colors -/+")
    print("  p/P    proc resolution -/+")
    print("  o/O    output resolution -/+")
    print("  b/B    brightness -/+")
    print("  n/N    contrast -/+")
    print("  t/T    saturation -/+")
    print("  s      save output")

    dirty = True
    output = filtered_img = post_img = None
    last_timings: dict = {}
    total_ms = 0.0

    while True:
        if dirty:
            t0 = time.perf_counter()
            output, filtered_img, post_img, last_timings = process_image(
                source, proc_size, out_size,
                num_colors, max_depth, min_depth, threshold,
                brightness, contrast, saturation,
                ornament_mask, ornament_mask_inv, src_tri,
            )
            total_ms = (time.perf_counter() - t0) * 1000
            dirty = False

        # Upscale for display
        display = cv2.resize(output, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        # --- Two thumbnails bottom-right ---
        thumb_size = display_size // 5
        gap = 4
        by = display_size - thumb_size - 8

        bx_post = display_size - thumb_size - 8
        post_thumb = cv2.resize(post_img, (thumb_size, thumb_size))
        display[by - 1 : by + thumb_size + 1, bx_post - 1 : bx_post + thumb_size + 1] = (255, 255, 255)
        display[by : by + thumb_size, bx_post : bx_post + thumb_size] = post_thumb

        bx_filt = bx_post - thumb_size - gap - 2
        filt_thumb = cv2.resize(filtered_img, (thumb_size, thumb_size))
        display[by - 1 : by + thumb_size + 1, bx_filt - 1 : bx_filt + thumb_size + 1] = (255, 255, 255)
        display[by : by + thumb_size, bx_filt : bx_filt + thumb_size] = filt_thumb

        label_y = by - 5
        for txt, bx in [("Filtered", bx_filt), ("Posterized", bx_post)]:
            cv2.putText(display, txt, (bx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(display, txt, (bx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- HUD ---
        lines = [
            f"total: {total_ms:.0f} ms",
            f"Depth: {min_depth}-{max_depth}  Colors: {num_colors}  Thresh: {threshold:.2f}",
            f"proc={proc_size}  out={out_size}",
            f"Bright: {brightness:+d}  Contr: {contrast:.2f}  Sat: {saturation:.2f}",
        ]
        for i, txt in enumerate(lines):
            _put_text(display, txt, 10, 25 + i * 26)

        if last_timings:
            steps  = ["filter", "post", "resize", "tree", "render"]
            labels = ["filt",   "post", "dwn",    "tree", "rend"]
            parts = "  ".join(
                f"{lbl}:{last_timings.get(s, 0):.0f}"
                for lbl, s in zip(labels, steps)
            )
            _put_text(display, parts + " ms", 10, 25 + len(lines) * 26, scale=0.5)

        # --- Source thumbnail bottom-left ---
        src_thumb = cv2.resize(source, (thumb_size, thumb_size))
        bx_src = 8
        display[by - 1 : by + thumb_size + 1, bx_src - 1 : bx_src + thumb_size + 1] = (255, 255, 255)
        display[by : by + thumb_size, bx_src : bx_src + thumb_size] = src_thumb
        cv2.putText(display, "Source", (bx_src, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(display, "Source", (bx_src, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Ornament Editor", display)

        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            max_depth = max(min_depth + 1, max_depth - 1);  dirty = True
        elif key == ord("D"):
            max_depth = min(15, max_depth + 1);             dirty = True
        elif key == ord("m"):
            min_depth = max(0, min_depth - 1);              dirty = True
        elif key == ord("M"):
            min_depth = min(max_depth - 1, min_depth + 1);  dirty = True
        elif key == ord("c"):
            num_colors = max(2, num_colors - 1);            dirty = True
        elif key == ord("C"):
            num_colors = min(8, num_colors + 1);            dirty = True
        elif key == ord("p"):
            proc_size = max(32, proc_size - 32);            dirty = True
        elif key == ord("P"):
            proc_size = min(out_size, proc_size + 32);      dirty = True
        elif key == ord("o"):
            out_size = max(64, out_size - 64);              dirty = True
        elif key == ord("O"):
            out_size = min(2048, out_size + 64);            dirty = True
        elif key == ord("h"):
            threshold = max(0.01, round(threshold - 0.01, 2)); dirty = True
        elif key == ord("H"):
            threshold = min(1.0, round(threshold + 0.01, 2));  dirty = True
        elif key == ord("b"):
            brightness -= 10;                               dirty = True
        elif key == ord("B"):
            brightness += 10;                               dirty = True
        elif key == ord("n"):
            contrast = round(contrast - 0.1, 2);            dirty = True
        elif key == ord("N"):
            contrast = round(contrast + 0.1, 2);            dirty = True
        elif key == ord("t"):
            saturation = max(0.0, round(saturation - 0.1, 2)); dirty = True
        elif key == ord("T"):
            saturation = round(saturation + 0.1, 2);        dirty = True
        elif key == ord("s"):
            base = os.path.splitext(os.path.basename(input_path))[0]
            fname = f"{base}_ornament_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}  ({out_size}x{out_size})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
