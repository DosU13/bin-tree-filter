"""Crossfading between looks (named parameter sets).

Numbers and colours glide with an ease-in-out curve; switches and choices
flip half way (core.params.interpolate).
"""

from __future__ import annotations

from core import params as P


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class LookFader:
    """Interpolates from one parameter dict to another over *duration* seconds."""

    def __init__(self, start: dict, end: dict, duration: float, now: float) -> None:
        self.start = P.normalise(start)
        self.end = P.normalise(end)
        self.duration = max(0.0, duration)
        self.t0 = now

    def progress(self, now: float) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (now - self.t0) / self.duration))

    def values(self, now: float) -> dict:
        return P.normalise(P.interpolate(self.start, self.end, ease(self.progress(now))))

    def done(self, now: float) -> bool:
        return self.progress(now) >= 1.0
