"""Snapping: pull moving edges onto nearby stage edges, cell lines and other items' edges."""

from __future__ import annotations

from dataclasses import dataclass, field

SNAP_PX = 8        # snap distance on screen


@dataclass
class Snapper:
    """Snap targets in world px; threshold in world px (SNAP_PX / view scale)."""

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    threshold: float = 10.0
    enabled: bool = True

    def _best(self, values: list[float], targets: list[float]) -> tuple[float, float | None]:
        """Smallest shift (and the target hit) that puts one of *values* on a target."""
        best_d, best_t = None, None
        for v in values:
            for t in targets:
                d = t - v
                if abs(d) <= self.threshold and (best_d is None or abs(d) < abs(best_d)):
                    best_d, best_t = d, t
        return (best_d or 0.0), best_t

    def snap_move(self, x: float, y: float, w: float, h: float) -> tuple[float, float, list, list]:
        """Shift a rectangle so its left / centre / right (top / middle / bottom) touches a target.

        Returns (x, y, vertical guide xs, horizontal guide ys).
        """
        if not self.enabled:
            return x, y, [], []
        dx, gx = self._best([x, x + w / 2, x + w], self.xs)
        dy, gy = self._best([y, y + h / 2, y + h], self.ys)
        return x + dx, y + dy, [gx] if gx is not None else [], [gy] if gy is not None else []

    def snap_point(self, px: float, py: float, snap_x: bool = True, snap_y: bool = True) -> tuple[float, float, list, list]:
        """Snap a dragged handle / corner."""
        if not self.enabled:
            return px, py, [], []
        gx = gy = None
        if snap_x:
            d, gx = self._best([px], self.xs)
            px += d
        if snap_y:
            d, gy = self._best([py], self.ys)
            py += d
        return px, py, [gx] if gx is not None else [], [gy] if gy is not None else []


def targets_for(scene, exclude: tuple[str, str] | None = None, cell_lines: bool = True) -> tuple[list, list]:
    """Snap targets: stage edges (+ cell lines) and every camera / projector edge."""
    st = scene.stage
    xs = [st.x, st.x + st.width, st.x + st.width / 2]
    ys = [st.y, st.y + st.height, st.y + st.height / 2]
    if cell_lines:
        xs += [st.x + i * st.cell_size for i in range(1, st.cols)]
        ys += [st.y + j * st.cell_size for j in range(1, st.rows)]
    for kind, items in (("camera", scene.cameras), ("projector", scene.projectors)):
        for item in items:
            if exclude == (kind, item.id):
                continue
            q = item.transform.quad()
            xs += [float(q[:, 0].min()), float(q[:, 0].max())]
            ys += [float(q[:, 1].min()), float(q[:, 1].max())]
    return xs, ys
