"""Stitch camera frames into one image of the stage at statistics resolution.

The stage's proc grid has proc_size pixels per cell.  For every camera the
compositor precomputes (inside the camera's bounding box on that grid):

    map_x, map_y   where each grid pixel samples the (pre-downscaled) frame
    weight         feather: 0 at the camera's edge, rising to 1 over `feather`
                   world px inside, so overlapping cameras cross-fade

A frame is then: area-downscale, per-camera colour fix, remap, weighted sum.
Grid pixels no camera sees are reported as uncovered.  Depth mode stitches
grey depth pictures instead (compose_depth), skipping pixels without a reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core import geometry as geo
from core.filters import apply_filters


@dataclass
class CameraMap:
    cam_id: str
    frame_size: tuple[int, int]      # (w, h) of the frames the maps were built for
    small_size: tuple[int, int]      # (w, h) frames are area-downscaled to first
    box: tuple[slice, slice]         # (rows, cols) region of the proc grid
    map_x: np.ndarray
    map_y: np.ndarray
    weight: np.ndarray               # float32, same shape as the box


def smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Compositor:
    """Cameras -> stage image (rows*proc x cols*proc)."""

    def __init__(self, stage, proc_size: int) -> None:
        self.stage = stage
        self.proc = proc_size
        self.height = stage.rows * proc_size
        self.width = stage.cols * proc_size
        # proc pixel -> world
        cs = stage.cell_size / proc_size
        self._grid_to_world = np.array([[cs, 0.0, stage.x], [0.0, cs, stage.y], [0.0, 0.0, 1.0]])
        self.maps: dict[str, CameraMap] = {}
        # Last downscaled (colour-fixed) frame per camera, for UI thumbnails
        self.last_small: dict[str, np.ndarray] = {}

    # -- geometry ------------------------------------------------------------

    def build(self, cam, frame_size: tuple[int, int]) -> CameraMap | None:
        """(Re)compute a camera's maps for frames of *frame_size*; None if it misses the stage."""
        fw, fh = frame_size
        t = cam.transform
        quad = t.quad()
        # Bounding box on the proc grid
        world_to_grid = geo.invert(self._grid_to_world)
        g = geo.apply(world_to_grid, quad)
        x0 = int(np.clip(np.floor(g[:, 0].min()), 0, self.width))
        x1 = int(np.clip(np.ceil(g[:, 0].max()), 0, self.width))
        y0 = int(np.clip(np.floor(g[:, 1].min()), 0, self.height))
        y1 = int(np.clip(np.ceil(g[:, 1].max()), 0, self.height))
        if x1 <= x0 or y1 <= y0:
            self.maps.pop(cam.id, None)
            return None

        # Grid pixel centres in the box -> world -> normalised display -> source pixels
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
        world = geo.apply(self._grid_to_world, pts)
        disp = geo.apply(geo.invert(t.norm_to_world()), world)
        src = geo.apply(geo.invert(geo.source_to_display(fw, fh, t.rotation, t.mirror)), disp)

        # Downscale frames first so each frame pixel is about one grid pixel
        px_per_world = self.proc / self.stage.cell_size
        dw, dh = geo.display_size(fw, fh, t.rotation)
        bx, by, bw, bh = t.bounds()
        scale = min(1.0, max(bw * px_per_world / dw, bh * px_per_world / dh) * 1.5)
        sw, sh = max(1, round(fw * scale)), max(1, round(fh * scale))
        sx, sy = sw / fw, sh / fh

        shape = (y1 - y0, x1 - x0)
        map_x = (src[:, 0] * sx - 0.5).astype(np.float32).reshape(shape)
        map_y = (src[:, 1] * sy - 0.5).astype(np.float32).reshape(shape)

        # Feather weight from the distance to the camera's edge (world px)
        dist = geo.quad_distance(world, quad).reshape(shape)
        feather = max(float(cam.feather), 1e-6)
        weight = np.where(dist > 0, smoothstep(dist / feather) + 1e-3, 0.0).astype(np.float32)

        m = CameraMap(cam.id, (fw, fh), (sw, sh), (slice(y0, y1), slice(x0, x1)), map_x, map_y, weight)
        self.maps[cam.id] = m
        return m

    def remove(self, cam_id: str) -> None:
        self.maps.pop(cam_id, None)
        self.last_small.pop(cam_id, None)

    # -- per frame -----------------------------------------------------------

    def _prepare(self, cam, frame: np.ndarray, grayscale: bool) -> tuple[CameraMap, np.ndarray] | None:
        m = self.maps.get(cam.id)
        size = (frame.shape[1], frame.shape[0])
        if m is None or m.frame_size != size:
            m = self.build(cam, size)
            if m is None:
                return None
        small = frame
        if m.small_size != size:
            small = cv2.resize(frame, m.small_size, interpolation=cv2.INTER_AREA)
        c = cam.color
        small = apply_filters(small, c.brightness, c.contrast, c.saturation)
        self.last_small[cam.id] = small
        if grayscale and small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return m, small

    def compose(self, cameras, frames: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Colour mode: (stage image uint8 HxWx3, covered bool HxW)."""
        acc = np.zeros((self.height, self.width, 3), dtype=np.float32)
        wsum = np.zeros((self.height, self.width), dtype=np.float32)
        for cam in cameras:
            frame = frames.get(cam.id)
            if frame is None:
                continue
            prep = self._prepare(cam, frame, grayscale=False)
            if prep is None:
                continue
            m, small = prep
            img = cv2.remap(small, m.map_x, m.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc[m.box] += img.astype(np.float32) * m.weight[:, :, None]
            wsum[m.box] += m.weight
        covered = wsum > 0
        out = np.zeros_like(acc)
        np.divide(acc, wsum[:, :, None], out=out, where=covered[:, :, None])
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), covered

    def compose_depth(self, cameras, frames: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Depth mode: (depth float32 HxW 0..255, valid bool HxW, covered bool HxW).

        Frames are grey depth pictures (near bright, 0 = no reading).  A grid pixel
        averages only the readings it covers, so holes don't drag depths towards
        far; it is valid where most of it had a reading.  The per-camera colour
        fix is not applied: it would shift depths.
        """
        acc = np.zeros((self.height, self.width), dtype=np.float32)
        vsum = np.zeros((self.height, self.width), dtype=np.float32)
        wsum = np.zeros((self.height, self.width), dtype=np.float32)
        for cam in cameras:
            frame = frames.get(cam.id)
            if frame is None:
                continue
            m = self.maps.get(cam.id)
            size = (frame.shape[1], frame.shape[0])
            if m is None or m.frame_size != size:
                m = self.build(cam, size)
                if m is None:
                    continue
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # (depth, has a reading): area-downscaling sums both, so depth / readings = mean reading
            pair = cv2.merge([gray.astype(np.float32), (gray > 0).astype(np.float32)])
            if m.small_size != size:
                pair = cv2.resize(pair, m.small_size, interpolation=cv2.INTER_AREA)
                self.last_small[cam.id] = cv2.resize(frame, m.small_size, interpolation=cv2.INTER_AREA)
            else:
                self.last_small[cam.id] = frame
            img = cv2.remap(pair, m.map_x, m.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc[m.box] += img[:, :, 0] * m.weight
            vsum[m.box] += img[:, :, 1] * m.weight
            wsum[m.box] += m.weight
        covered = wsum > 0
        valid = vsum > 0.5 * wsum
        depth = np.zeros_like(acc)
        np.divide(acc, vsum, out=depth, where=valid)
        return depth, valid, covered
