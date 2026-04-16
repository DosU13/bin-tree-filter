"""Real-time webcam filter — Kyrgyz ornamental pattern.

Controls:
    q / ESC   — quit
    d / D     — decrease / increase max depth
    c / C     — decrease / increase number of colours
    r / R     — decrease / increase render resolution
    s         — save current frame to file
"""

import os
import time

import cv2
import numpy as np

from posterize import posterize
from triangle_tree import make_initial_triangles, build_tree, collect_leaves
from renderer import load_ornament_mask, render


# ---------------------------------------------------------------------------
# Pipeline (single frame)
# ---------------------------------------------------------------------------

def process_frame(
    frame: np.ndarray,
    size: int,
    num_colors: int,
    max_depth: int,
    threshold: float,
    ornament_mask: np.ndarray,
    ornament_mask_inv: np.ndarray,
    src_tri: np.ndarray,
) -> np.ndarray:
    """Run the full ornament pipeline on one square BGR frame."""

    # Centre-crop to square, then resize
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    square = frame[y0 : y0 + s, x0 : x0 + s]
    square = cv2.resize(square, (size, size))

    # Posterize
    post, _ = posterize(square, num_colors)

    # Build tree
    roots = make_initial_triangles(size)
    all_leaves = []
    for root in roots:
        build_tree(post, root, max_depth, threshold)
        all_leaves.extend(collect_leaves(root))

    # Render
    output = np.full((size, size, 3), 255, dtype=np.uint8)
    render(output, all_leaves, ornament_mask, ornament_mask_inv, src_tri)

    return output


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
    size = 256
    num_colors = 4
    max_depth = 8
    threshold = 0.75
    display_size = 640

    print("Webcam ornament filter running.  Controls:")
    print("  q/ESC  quit")
    print("  d/D    depth  -/+")
    print("  c/C    colors -/+")
    print("  r/R    resolution -/+")
    print("  s      save frame")

    frame_count = 0
    fps = 0.0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        output = process_frame(
            frame, size, num_colors, max_depth, threshold,
            ornament_mask, ornament_mask_inv, src_tri,
        )

        proc_ms = (time.perf_counter() - t0) * 1000

        # Upscale for display
        display = cv2.resize(output, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        # Raw webcam thumbnail in bottom-right corner
        thumb_size = display_size // 4
        h_f, w_f = frame.shape[:2]
        s_f = min(h_f, w_f)
        y0_f, x0_f = (h_f - s_f) // 2, (w_f - s_f) // 2
        raw_crop = frame[y0_f : y0_f + s_f, x0_f : x0_f + s_f]
        thumb = cv2.resize(raw_crop, (thumb_size, thumb_size))
        # Place with a 1px white border
        bx = display_size - thumb_size - 8
        by = display_size - thumb_size - 8
        display[by - 1 : by + thumb_size + 1, bx - 1 : bx + thumb_size + 1] = (255, 255, 255)
        display[by : by + thumb_size, bx : bx + thumb_size] = thumb

        # HUD overlay
        frame_count += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_start = time.perf_counter()

        lines = [
            f"FPS: {fps:.1f}  ({proc_ms:.0f} ms)",
            f"Depth: {max_depth}   Colors: {num_colors}   Res: {size}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(
                display, txt, (10, 25 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
            )
            cv2.putText(
                display, txt, (10, 25 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
            )

        cv2.imshow("Kyrgyz Ornament", display)

        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            max_depth = max(1, max_depth - 1)
        elif key == ord("D"):
            max_depth = min(10, max_depth + 1)
        elif key == ord("c"):
            num_colors = max(2, num_colors - 1)
        elif key == ord("C"):
            num_colors = min(8, num_colors + 1)
        elif key == ord("r"):
            size = max(64, size - 64)
        elif key == ord("R"):
            size = min(512, size + 64)
        elif key == ord("s"):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
