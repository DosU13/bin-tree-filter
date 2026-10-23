"""Crossfading between looks (named parameter sets), and the idle look.

Numbers and colours glide with an ease-in-out curve; switches and choices
flip half way (core.params.interpolate).  Show settings never fade.
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
        self.end = P.look_target(self.start, end)
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


# ---------------------------------------------------------------------------
# Idle look
# ---------------------------------------------------------------------------

class IdleFader:
    """How far to fade towards the idle look: up after nobody has been on stage for
    `idle_after` seconds, back down as soon as someone appears (`idle_fade` each way)."""

    def __init__(self) -> None:
        self.empty_for = 0.0
        self.level = 0.0               # linear 0..1; value() eases it

    def update(self, occupied: bool, dt: float, after: float, fade: float) -> float:
        self.empty_for = 0.0 if occupied else self.empty_for + dt
        target = 1.0 if self.empty_for >= after else 0.0
        step = dt / fade if fade > 0 else 1.0
        if target > self.level:
            self.level = min(target, self.level + step)
        else:
            self.level = max(target, self.level - step)
        return self.value()

    def value(self) -> float:
        return ease(self.level)

    def reset(self) -> None:
        self.empty_for = 0.0
        self.level = 0.0


def idle_mix(params: dict, looks: dict, amount: float) -> dict:
    """*params* faded *amount* (0..1) of the way towards the idle look (params["idle_look"])."""
    look = looks.get(params.get("idle_look") or "")
    if amount <= 0.0 or look is None:
        return params
    return P.normalise(P.interpolate(params, P.look_target(params, look), amount))
