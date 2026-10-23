"""Status bar: frame rate, per-step timings and a health dot per camera / projector."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from align_perform.canvas.items import STATUS_COLORS


class StatusPanel(QWidget):
    def __init__(self, doc, parent=None) -> None:
        super().__init__(parent)
        self.doc = doc
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        self.perf = QLabel("starting…")
        self.cams = QLabel()
        self.projs = QLabel()
        lay.addWidget(self.perf)
        lay.addSpacing(16)
        lay.addWidget(self.cams)
        lay.addSpacing(16)
        lay.addWidget(self.projs)

    def update_frame(self, r, paused: bool = False) -> None:
        t = r.timings
        parts = [f"{'PAUSED' if paused else f'{r.fps:4.1f} fps'}", f"{t.get('total', 0):4.1f} ms",
                 f"leaves {r.leaves}"]
        if r.activity is not None:
            parts.append("learning empty stage…" if r.learning else f"people {r.people_count}")
        if r.idle > 0:
            parts.append(f"idle {r.idle:.0%}")
        detail = "  ".join(f"{k} {v:.1f}" for k, v in t.items() if k != "total")
        self.perf.setText("   ".join(parts))
        self.perf.setToolTip(detail + " (ms)")
        dots = []
        for c in self.doc.scene.cameras:
            st = r.camera_status.get(c.id, "disabled")
            dots.append(f'<span style="color:{STATUS_COLORS.get(st, STATUS_COLORS["stopped"]).name()}">●</span> '
                        f'{c.name or c.id}')
        self.cams.setText("&nbsp;&nbsp;".join(dots) if dots else "no cameras")

    def set_projectors(self, states: dict[str, str]) -> None:
        dots = []
        for p in self.doc.scene.projectors:
            st = states.get(p.id, "disabled" if not p.enabled else "stopped")
            color = {"shown": STATUS_COLORS["ok"], "missing": STATUS_COLORS["reconnecting"]}.get(
                st, STATUS_COLORS["stopped"])
            dots.append(f'<span style="color:{color.name()}">■</span> {p.name or p.id}')
        self.projs.setText("&nbsp;&nbsp;".join(dots))
