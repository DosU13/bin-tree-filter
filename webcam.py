"""Real-time webcam filter — Kyrgyz ornamental pattern.

Controls:
    q / ESC   — quit
    d / D     — decrease / increase max depth
    m / M     — decrease / increase min depth
    c / C     — decrease / increase number of colours
    r / R     — decrease / increase input resolution (filters + posterize)
    p / P     — decrease / increase proc resolution (tree building)
    o / O     — decrease / increase output resolution (render quality)
    [ / ]     — decrease / increase blur radius
    ; / '     — decrease / increase sharpen amount
    b / B     — decrease / increase brightness
    n / N     — decrease / increase contrast
    t / T     — decrease / increase saturation
    s         — save current frame to file
"""

import os
import time

import cv2
import numpy as np

from config import BACKGROUND_COLOR
from posterize import posterize
from triangle_tree import make_initial_triangles, build_tree, collect_leaves
from renderer import load_ornament_mask, render


# ---------------------------------------------------------------------------
# Image calibration filters
# ---------------------------------------------------------------------------

def apply_blur(image: np.ndarray, blur_size: int) -> np.ndarray:
    """Gaussian blur. blur_size=0 means no blur; each step adds 2px to the kernel."""
    if blur_size <= 0:
        return image
    k = blur_size * 2 + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp-mask sharpening. amount=0 means no sharpening."""
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def apply_filters(
    image: np.ndarray,
    brightness: int,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """Apply brightness, contrast, and saturation adjustments."""
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


# ---------------------------------------------------------------------------
# Pipeline (single frame)
# ---------------------------------------------------------------------------

def _scale_leaves(leaves, scale: float) -> None:
    """Scale all leaf triangle vertices in-place by scale factor."""
    if scale == 1.0:
        return
    for leaf in leaves:
        leaf.right_angle = (leaf.right_angle[0] * scale, leaf.right_angle[1] * scale)
        leaf.hyp_a      = (leaf.hyp_a[0]      * scale, leaf.hyp_a[1]      * scale)
        leaf.hyp_b      = (leaf.hyp_b[0]      * scale, leaf.hyp_b[1]      * scale)


def process_frame(
    frame: np.ndarray,
    size: int,
    proc_size: int,
    out_size: int,
    num_colors: int,
    max_depth: int,
    min_depth: int,
    threshold: float,
    blur_size: int,
    sharpen_amount: float,
    brightness: int,
    contrast: float,
    saturation: float,
    ornament_mask: np.ndarray,
    ornament_mask_inv: np.ndarray,
    src_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the full ornament pipeline on one square BGR frame.

    Pipeline:
        frame → resize(size) → blur → sharpen → filters → posterize →
        resize(proc_size) → build tree → scale vertices → render(out_size)

    Returns:
        output:     Rendered ornament image (out_size x out_size).
        filtered:   Frame after blur + sharpen + calibration filters.
        posterized: Frame after posterize (at size x size).
        timings:    Dict of per-step milliseconds.
    """
    timings = {}

    # Centre-crop to square, resize to input resolution
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    square = frame[y0 : y0 + s, x0 : x0 + s]
    square = cv2.resize(square, (size, size))

    # Step 1 — blur → sharpen → calibration filters
    t0 = time.perf_counter()
    filtered = apply_blur(square, blur_size)
    filtered = apply_sharpen(filtered, sharpen_amount)
    filtered = apply_filters(filtered, brightness, contrast, saturation)
    timings["filter"] = (time.perf_counter() - t0) * 1000

    # Step 2 — posterize
    t0 = time.perf_counter()
    post, _ = posterize(filtered, num_colors)
    timings["post"] = (time.perf_counter() - t0) * 1000

    # Step 3 — downscale to proc_size
    t0 = time.perf_counter()
    if proc_size != size:
        post_proc = cv2.resize(post, (proc_size, proc_size), interpolation=cv2.INTER_NEAREST)
    else:
        post_proc = post
    timings["resize"] = (time.perf_counter() - t0) * 1000

    # Step 4 — build tree at proc_size
    t0 = time.perf_counter()
    roots = make_initial_triangles(proc_size)
    all_leaves = []
    for root in roots:
        build_tree(post_proc, root, max_depth, threshold, min_depth)
        all_leaves.extend(collect_leaves(root))
    timings["tree"] = (time.perf_counter() - t0) * 1000

    # Step 5 — scale vertices to out_size, then render
    t0 = time.perf_counter()
    _scale_leaves(all_leaves, out_size / proc_size)
    output = np.full((out_size, out_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    render(output, all_leaves, ornament_mask, ornament_mask_inv, src_tri)
    timings["render"] = (time.perf_counter() - t0) * 1000

    return output, filtered, post, timings


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def _put_text(img, txt, x, y, scale=0.65):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ornament_path = os.path.join(script_dir, "ornam.jpg")
    ornament_mask, ornament_mask_inv, src_tri = load_ornament_mask(ornament_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    # Tuneable parameters
    size      = 256    # input resolution: filters + posterize
    proc_size = 128    # tree-building resolution (fast)
    out_size  = 256    # render resolution (quality)
    num_colors = 4
    max_depth  = 8
    min_depth  = 0
    threshold  = 0.75
    display_size = 1280
    blur_size      = 0
    sharpen_amount = 0.0
    brightness = 0
    contrast   = 1.0
    saturation = 1.0

    print("Webcam ornament filter running.  Controls:")
    print("  q/ESC  quit")
    print("  d/D    max depth -/+")
    print("  m/M    min depth -/+")
    print("  h/H    homogeneity threshold -/+")
    print("  c/C    colors -/+")
    print("  r/R    input resolution -/+   (filters+posterize)")
    print("  p/P    proc resolution -/+    (tree building)")
    print("  o/O    output resolution -/+  (render quality)")
    print("  [/]    blur radius -/+")
    print("  ;/'    sharpen amount -/+")
    print("  b/B    brightness -/+")
    print("  n/N    contrast -/+")
    print("  t/T    saturation -/+")
    print("  s      save frame")

    frame_count = 0
    fps = 0.0
    t_start = time.perf_counter()
    last_timings: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_total0 = time.perf_counter()

        output, filtered_img, post_img, last_timings = process_frame(
            frame, size, proc_size, out_size,
            num_colors, max_depth, min_depth, threshold,
            blur_size, sharpen_amount,
            brightness, contrast, saturation,
            ornament_mask, ornament_mask_inv, src_tri,
        )

        total_ms = (time.perf_counter() - t_total0) * 1000

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

        # --- FPS counter ---
        frame_count += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_start = time.perf_counter()

        # --- HUD lines ---
        lines = [
            f"FPS: {fps:.1f}   total: {total_ms:.0f} ms",
            f"Depth: {min_depth}-{max_depth}  Colors: {num_colors}  Thresh: {threshold:.2f}",
            f"Res: in={size}  proc={proc_size}  out={out_size}",
            f"Blur: {blur_size}  Sharp: {sharpen_amount:.1f}  Bright: {brightness:+d}  Contr: {contrast:.2f}  Sat: {saturation:.2f}",
        ]
        for i, txt in enumerate(lines):
            _put_text(display, txt, 10, 25 + i * 26)

        # --- Per-step timing bar ---
        if last_timings:
            steps = ["filter", "post", "resize", "tree", "render"]
            labels = ["filt", "post", "dwn", "tree", "rend"]
            parts = "  ".join(
                f"{lbl}:{last_timings.get(s, 0):.0f}"
                for lbl, s in zip(labels, steps)
            )
            _put_text(display, parts + " ms", 10, 25 + len(lines) * 26, scale=0.5)

        cv2.imshow("Kyrgyz Ornament", display)

        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            max_depth = max(min_depth + 1, max_depth - 1)
        elif key == ord("D"):
            max_depth = min(15, max_depth + 1)
        elif key == ord("m"):
            min_depth = max(0, min_depth - 1)
        elif key == ord("M"):
            min_depth = min(max_depth - 1, min_depth + 1)
        elif key == ord("c"):
            num_colors = max(2, num_colors - 1)
        elif key == ord("C"):
            num_colors = min(8, num_colors + 1)
        elif key == ord("r"):
            size = max(64, size - 32)
            proc_size = min(proc_size, size)
            out_size  = min(out_size, size * 4)
        elif key == ord("R"):
            size = min(512, size + 32)
        elif key == ord("p"):
            proc_size = max(32, proc_size - 32)
        elif key == ord("P"):
            proc_size = min(size, proc_size + 32)
        elif key == ord("o"):
            out_size = max(64, out_size - 64)
        elif key == ord("O"):
            out_size = min(1024, out_size + 64)
        elif key == ord("h"):
            threshold = max(0.01, round(threshold - 0.01, 2))
        elif key == ord("H"):
            threshold = min(1.0, round(threshold + 0.01, 2))
        elif key == ord("["):
            blur_size = max(0, blur_size - 1)
        elif key == ord("]"):
            blur_size += 1
        elif key == ord(";"):
            sharpen_amount = max(0.0, round(sharpen_amount - 0.2, 1))
        elif key == ord("'"):
            sharpen_amount = round(sharpen_amount + 0.2, 1)
        elif key == ord("b"):
            brightness -= 10
        elif key == ord("B"):
            brightness += 10
        elif key == ord("n"):
            contrast = round(contrast - 0.1, 2)
        elif key == ord("N"):
            contrast = round(contrast + 0.1, 2)
        elif key == ord("t"):
            saturation = max(0.0, round(saturation - 0.1, 2))
        elif key == ord("T"):
            saturation = round(saturation + 0.1, 2)
        elif key == ord("s"):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
