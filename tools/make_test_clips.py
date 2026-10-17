"""Generate the test clips used by scenes/example.json.

A 1620 x 540 panorama (the sample photos side by side, with a disc and a bar
moving across) is cut into three overlapping "camera" views, like three
webcams side by side.  The right camera is 12 levels brighter, to exercise the
per-camera colour fix.

    python -m tools.make_test_clips          # writes scenes/media/*.mp4
"""

from __future__ import annotations

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scenes", "media")

PANO_W, PANO_H = 1620, 540
FPS, SECONDS = 15, 4
CAMERAS = {            # name: (x0 in the panorama, brightness offset)
    "cam_left": (0, 0),
    "cam_mid": (450, 0),
    "cam_right": (900, 12),
}
CROP_W = 720
CLIP_SIZE = (480, 360)


def panorama_background() -> np.ndarray:
    tiles = []
    for name in ("aziz.jpg", "me.jpg", "test_girl_sq.jpg"):
        img = cv2.imread(os.path.join(ROOT, name))
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        tiles.append(cv2.resize(img, (PANO_H, PANO_H), interpolation=cv2.INTER_AREA))
    return np.concatenate(tiles, axis=1)


def panorama_frame(bg: np.ndarray, i: int) -> np.ndarray:
    t = i / (FPS * SECONDS)
    img = bg.copy()
    # Disc travelling left -> right across every camera overlap
    cx = int(-80 + t * (PANO_W + 160))
    cv2.circle(img, (cx, PANO_H // 2), 70, (40, 60, 230), -1, cv2.LINE_AA)
    # Bar sweeping down
    y = int(t * PANO_H)
    cv2.rectangle(img, (0, y - 12), (PANO_W, y + 12), (230, 200, 40), -1)
    return img


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    bg = panorama_background()
    writers = {
        name: cv2.VideoWriter(os.path.join(OUT, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, CLIP_SIZE)
        for name in CAMERAS
    }
    for i in range(FPS * SECONDS):
        pano = panorama_frame(bg, i)
        for name, (x0, bright) in CAMERAS.items():
            crop = pano[:, x0 : x0 + CROP_W]
            if bright:
                crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=bright)
            writers[name].write(cv2.resize(crop, CLIP_SIZE, interpolation=cv2.INTER_AREA))
    for w in writers.values():
        w.release()
    cv2.imwrite(os.path.join(OUT, "panorama.jpg"), panorama_frame(bg, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
