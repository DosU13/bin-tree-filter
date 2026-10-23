"""Run a scene without any UI and write what each projector would show.

    python -m tools.render_scene scenes/example.json --frames 45 --out out/render
    python -m tools.render_scene scenes/example.json --video     # also an MP4 per projector

Writes, for the last frame:
    <projector id>.png     each projector's output (native resolution x --scale)
    stage_input.png        the cameras stitched at statistics resolution
    stage_labels.png       the posterized / depth-band labels
    stage_presence.png     people found (when presence, waves or an idle look is on)
    stage.png              the ornament over the whole stage, one render
    projected.png          the projector outputs added together as light in
                           world space (edge blending included) - seams show here
"""

from __future__ import annotations

import argparse
import os
import time

import core  # noqa: F401  (OpenCV env before cv2 loads)
import cv2
import numpy as np

from core import geometry as geo
from core.engine import load_tile
from core.pipeline import Pipeline
from core.presence import draw_presence
from core.scene import Scene
from core.view import projector_view, stage_view


def scene_tile(scene: Scene, params: dict) -> np.ndarray:
    return load_tile(scene.resolve(params["tile"]), crop=params["tile_crop"])


def simulate_projection(scene: Scene, outputs: dict[str, np.ndarray], scale: float, gamma: float = 2.2) -> np.ndarray:
    """Add projector images as light on a world canvas covering the stage."""
    st = scene.stage
    w, h = round(st.width * scale), round(st.height * scale)
    world_to_canvas = geo.translate_scale(-st.x * scale, -st.y * scale, scale, scale)
    light = np.zeros((h, w, 3), np.float32)
    for proj in scene.enabled_projectors():
        img = outputs.get(proj.id)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        pix_to_norm = geo.translate_scale(0, 0, 1 / iw, 1 / ih)
        m = world_to_canvas @ proj.transform.norm_to_world() @ pix_to_norm
        warped = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        light += (warped.astype(np.float32) / 255) ** gamma
    return (np.clip(light, 0, 1) ** (1 / gamma) * 255 + 0.5).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--frames", type=int, default=45, help="frames to run (default 45)")
    ap.add_argument("--fps", type=float, default=15.0, help="simulated frame rate")
    ap.add_argument("--out", default="out/render", help="output folder")
    ap.add_argument("--scale", type=float, default=0.5, help="projector output scale (default 0.5)")
    ap.add_argument("--video", action="store_true", help="also write an MP4 per projector")
    args = ap.parse_args()

    from align_perform.runtime.gl_renderer import OffscreenGL

    scene = Scene.load(args.scene)
    os.makedirs(args.out, exist_ok=True)
    pipe = Pipeline(scene, realtime=False)
    gl = OffscreenGL()
    gl.set_tile(scene_tile(scene, pipe.params))
    projs = scene.enabled_projectors()
    sizes = {p.id: (max(1, round(p.output.width * args.scale)), max(1, round(p.output.height * args.scale)))
             for p in projs}
    writers = {}
    if args.video:
        for p in projs:
            writers[p.id] = cv2.VideoWriter(os.path.join(args.out, f"{p.id}.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"), args.fps, sizes[p.id])

    t0 = time.perf_counter()
    outputs: dict[str, np.ndarray] = {}
    result = None
    for _ in range(args.frames):
        result = pipe.step(1 / args.fps)
        for p in projs:
            w, h = sizes[p.id]
            view = projector_view(scene, p, w, h, params=pipe.effective)
            view.clock = result.snapshot.clock           # simulated time, not the wall clock
            outputs[p.id] = gl.render(result.snapshot, view)
            if p.id in writers:
                writers[p.id].write(outputs[p.id])
    elapsed = time.perf_counter() - t0
    for w in writers.values():
        w.release()

    for pid, img in outputs.items():
        cv2.imwrite(os.path.join(args.out, f"{pid}.png"), img)
    cv2.imwrite(os.path.join(args.out, "stage_input.png"), result.stage_input)
    cv2.imwrite(os.path.join(args.out, "stage_labels.png"), result.labels_image)
    if result.activity is not None:
        cv2.imwrite(os.path.join(args.out, "stage_presence.png"),
                    draw_presence(result.stage_input, result.activity, result.people, pipe.params["proc_size"]))
    stage_w = round(scene.stage.width * args.scale)
    view = stage_view(scene, stage_w, params=pipe.effective)
    view.clock = result.snapshot.clock
    cv2.imwrite(os.path.join(args.out, "stage.png"), gl.render(result.snapshot, view))
    world_scale = min(1.0, stage_w / scene.stage.width)
    cv2.imwrite(os.path.join(args.out, "projected.png"), simulate_projection(scene, outputs, world_scale))
    pipe.close()
    gl.release()
    print(f"{args.frames} frames in {elapsed:.1f} s ({args.frames / elapsed:.1f} fps incl. readback); "
          f"leaves {result.leaves}; people {result.people_count}; idle {result.idle:.2f}; last frame timings: "
          + ", ".join(f"{k} {v:.1f} ms" for k, v in result.timings.items()))
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
