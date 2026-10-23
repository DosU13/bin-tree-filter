"""Generate the test clips used by scenes/example.json and scenes/people.json.

A 1620 x 540 panorama (the sample photos side by side, with a disc and a bar
moving across) is cut into three overlapping "camera" views, like three
webcams side by side.  The right camera is 12 levels brighter, to exercise the
per-camera colour fix.

The people clip is one still camera view with two figures walking across it
(2 s empty, two walkers, 6 s empty again), plus sensor noise: presence, waves
and the idle look can be tried without a webcam.

    python -m tools.make_test_clips             # writes scenes/media/cam_*.mp4
    python -m tools.make_test_clips --people    # writes scenes/media/walkers.mp4 and scenes/people.json
"""

from __future__ import annotations

import argparse
import math
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


# ---------------------------------------------------------------------------
# People clip
# ---------------------------------------------------------------------------

PEOPLE_SIZE = (720, 360)
PEOPLE_SECONDS = 16
WALKERS = (   # start s, end s, x from, x to (clip px), feet y, height, shirt, trousers (BGR)
    (2.0, 9.0, -60, 780, 345, 240, (40, 40, 200), (90, 50, 30)),
    (4.0, 10.5, 780, -60, 300, 170, (40, 190, 230), (70, 70, 70)),
)


def draw_person(img: np.ndarray, x: float, feet: float, height: float, phase: float, shirt, trousers) -> None:
    """A simple walking figure: legs and arms swing with *phase*."""
    h = height
    x, feet = int(x), int(feet)
    head_r = int(h * 0.09)
    neck = feet - int(h * 0.80)
    hip = feet - int(h * 0.47)
    swing = int(h * 0.13 * math.sin(phase))
    leg_w, arm_w = max(2, int(h * 0.09)), max(2, int(h * 0.055))
    cv2.line(img, (x, hip), (x - swing, feet), trousers, leg_w, cv2.LINE_AA)
    cv2.line(img, (x, hip), (x + swing, feet), trousers, leg_w, cv2.LINE_AA)
    cv2.ellipse(img, (x, (neck + hip) // 2 + 4), (int(h * 0.11), (hip - neck) // 2 + 6), 0, 0, 360, shirt, -1,
                cv2.LINE_AA)
    cv2.line(img, (x, neck + 8), (x + swing, hip + 6), shirt, arm_w, cv2.LINE_AA)
    cv2.line(img, (x, neck + 8), (x - swing, hip + 6), shirt, arm_w, cv2.LINE_AA)
    cv2.circle(img, (x, neck - head_r + 2), head_r, (120, 160, 210), -1, cv2.LINE_AA)


def people_frame(bg: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    img = bg.copy()
    for t0, t1, x0, x1, feet, height, shirt, trousers in WALKERS:
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            draw_person(img, x0 + (x1 - x0) * u, feet, height, (t - t0) * 7.0, shirt, trousers)
    noise = rng.normal(0.0, 2.5, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_people() -> None:
    os.makedirs(OUT, exist_ok=True)
    pano = panorama_background()
    bg = cv2.resize(pano[:, 270:1350], PEOPLE_SIZE, interpolation=cv2.INTER_AREA)
    path = os.path.join(OUT, "walkers.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, PEOPLE_SIZE)
    rng = np.random.default_rng(7)
    for i in range(FPS * PEOPLE_SECONDS):
        writer.write(people_frame(bg, i / FPS, rng))
    writer.release()
    write_people_scene()
    print("wrote", path)


def write_people_scene() -> None:
    """scenes/people.json: the walkers clip on a 2 x 1 stage, one test-window projector,
    people only + waves, and an idle look after 4 s with nobody."""
    from core import params as P
    from core.scene import Scene

    show = P.normalise(dict(P.defaults(), max_depth=10, threshold=0.8, presence=True, waves=True,
                            idle_look="idle", idle_after=4.0, idle_fade=2.0))
    idle = P.look_values(dict(show, presence=False, waves=False, max_depth=5, colors=3, palette_tau=2.0,
                              tau_color=0.6, split_time=0.4))
    data = {
        "version": 1,
        "stage": {"x": 0, "y": 0, "cols": 2, "rows": 1, "cell_size": 540, "levels": 11},
        "cameras": [{
            "id": "cam1", "name": "Front", "enabled": True,
            "source": {"kind": "video", "path": os.path.join(OUT, "walkers.mp4"),
                       "width": PEOPLE_SIZE[0], "height": PEOPLE_SIZE[1], "fps": FPS},
            "transform": {"x": 0, "y": 0, "width": 1080, "height": 540},
        }],
        "projectors": [{
            "id": "proj1", "name": "Wall", "enabled": True,
            "output": {"kind": "window", "width": 1920, "height": 960},
            "transform": {"x": 0, "y": 0, "width": 1080, "height": 540},
        }],
        "params": show,
        "looks": {"people": P.look_values(show), "idle": idle},
    }
    path = os.path.join(ROOT, "scenes", "people.json")
    Scene.from_dict(data, path=path).save(path)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", action="store_true", help="write the people clip and scene instead")
    if ap.parse_args().people:
        make_people()
        return
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
