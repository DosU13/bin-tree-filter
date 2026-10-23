import numpy as np
import pytest

from core import params as P
from core.looks import IdleFader, LookFader, idle_mix
from core.presence import (ARRIVE_SECONDS, LEARN_SECONDS, MAX_WAVES, RELEARN_SECONDS, TRACK_GRACE, Person,
                           PresenceTracker, WaveEmitter, draw_presence)

CELL = 64                       # proc px per cell
H, W = CELL, 3 * CELL           # a 3 x 1 stage
DT = 1 / 15


def stage_bg(seed=0) -> np.ndarray:
    """A textured still stage."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (H // 8, W // 8, 3)).astype(np.uint8)
    return np.kron(base, np.ones((8, 8, 1), np.uint8))


def with_noise(img, rng, sigma=3.0) -> np.ndarray:
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def with_person(img, x, y=20, w=12, h=36, colour=(20, 30, 230)) -> np.ndarray:
    out = img.copy()
    out[y:y + h, x:x + w] = colour
    return out


@pytest.fixture
def covered():
    return np.ones((H, W), bool)


def run(tracker, frames, covered, **kw):
    out = None
    for f in frames:
        out = tracker.update(f, covered, DT, cell_px=CELL, **kw)
    return out


def learned(covered, seed=0):
    t = PresenceTracker()
    rng = np.random.default_rng(seed)
    bg = stage_bg()
    run(t, [with_noise(bg, rng) for _ in range(int(LEARN_SECONDS / DT) + 2)], covered)
    assert not t.learning
    return t, bg, rng


def test_still_stage_is_empty(covered):
    t, bg, rng = learned(covered)
    pres = run(t, [with_noise(bg, rng) for _ in range(30)], covered)
    assert not pres.mask.any() and pres.count == 0 and pres.activity.max() == 0


def test_moving_person_is_found_and_tracked(covered):
    t, bg, rng = learned(covered)
    ids = set()
    for i in range(20):
        pres = t.update(with_noise(with_person(bg, 20 + 3 * i), rng), covered, DT, cell_px=CELL)
        ids |= {p.id for p in pres.people}
    x = 20 + 3 * 19
    region = np.zeros((H, W), bool)
    region[20:56, x:x + 12] = True
    iou = (pres.mask & region).sum() / (pres.mask | region).sum()
    assert iou > 0.7
    assert pres.count == 1 and len(ids) == 1
    p = pres.people[0]
    assert p.x == pytest.approx((x + 6) / CELL, abs=0.05)
    assert p.foot_y == pytest.approx(56 / CELL, abs=0.05)
    assert p.age > 1.0


def test_trail_fades_after_the_person_leaves(covered):
    t, bg, rng = learned(covered)
    run(t, [with_noise(with_person(bg, 40), rng) for _ in range(5)], covered, trail=1.0)
    pres = run(t, [with_noise(bg, rng) for _ in range(3)], covered, trail=1.0)
    assert not pres.mask.any()
    assert 0.7 < pres.activity[30, 45] < 1.0                      # lingering
    pres = run(t, [with_noise(bg, rng) for _ in range(75)], covered, trail=1.0)
    assert pres.activity.max() < 0.01


def test_exposure_change_is_not_motion(covered):
    t, bg, rng = learned(covered)
    brighter = np.clip(bg.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    pres = run(t, [with_noise(brighter, rng) for _ in range(10)], covered)
    assert pres.mask.mean() < 0.01


def test_still_person_is_absorbed_unless_disabled(covered):
    for absorb, gone in ((2.0, True), (0.0, False)):
        t, bg, rng = learned(covered)
        frame = lambda: with_noise(with_person(bg, 80), rng)
        pres = run(t, [frame() for _ in range(5)], covered, absorb=absorb)
        assert pres.count == 1
        pres = run(t, [frame() for _ in range(150)], covered, absorb=absorb)     # 10 s standing still
        assert (pres.count == 0) == gone


def test_big_change_relearns(covered):
    t, bg, rng = learned(covered)
    other = stage_bg(seed=5)
    run(t, [with_noise(other, rng) for _ in range(int(RELEARN_SECONDS / DT) + 2)], covered)
    pres = run(t, [with_noise(other, rng) for _ in range(int(LEARN_SECONDS / DT) + 5)], covered)
    assert not t.learning and pres.mask.mean() < 0.01


def test_relearn_clears_a_ghost(covered):
    t = PresenceTracker()
    rng = np.random.default_rng(1)
    bg = stage_bg()
    run(t, [with_noise(with_person(bg, 100), rng) for _ in range(10)], covered)   # learned with someone there
    pres = run(t, [with_noise(bg, rng) for _ in range(5)], covered, absorb=0)
    assert pres.count == 1                                                        # the ghost
    t.relearn()
    pres = run(t, [with_noise(bg, rng) for _ in range(12)], covered, absorb=0)
    assert pres.count == 0 and not pres.mask.any()


def test_uncovered_pixels_never_count():
    cov = np.zeros((H, W), bool)
    cov[:, :CELL] = True
    t, bg, rng = learned(cov)
    pres = run(t, [with_noise(with_person(bg, 120), rng) for _ in range(5)], cov)
    assert not pres.mask.any()


def test_small_blobs_are_noise(covered):
    t, bg, rng = learned(covered)
    pres = run(t, [with_noise(with_person(bg, 50, w=4, h=4), rng) for _ in range(5)], covered, min_size=1.0)
    assert pres.count == 0


def test_two_people_keep_their_ids(covered):
    t, bg, rng = learned(covered)
    first = None
    for i in range(15):
        f = with_person(with_person(bg, 10 + 2 * i), 150 - 2 * i, colour=(230, 200, 20))
        pres = t.update(with_noise(f, rng), covered, DT, cell_px=CELL)
        if first is None and pres.count == 2:
            first = {p.id: p.x for p in pres.people}
    ids = {p.id: p.x for p in pres.people}
    assert set(ids) == set(first)
    left = min(first, key=first.get)
    assert ids[left] > first[left]          # the left one moved right, same id


def test_draw_presence_shape(covered):
    t, bg, rng = learned(covered)
    pres = run(t, [with_noise(with_person(bg, 60), rng) for _ in range(3)], covered)
    img = draw_presence(bg, pres.activity, pres.people, CELL)
    assert img.shape == bg.shape and img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------

def person(pid, age=1.0, missing=0.0):
    return Person(pid, 1.0, 0.5, 1.0, 0.9, 0.1, age=age, missing=missing)


def test_waves_on_arrival_then_every_interval():
    e = WaveEmitter()
    assert len(e.update([person(1, age=ARRIVE_SECONDS / 2)], 0.0, 1.0, "feet", 5.0)) == 0
    w = e.update([person(1)], 0.1, 1.0, "feet", 5.0)
    assert len(w) == 1 and tuple(w[0]) == pytest.approx((1.0, 0.9, 0.1, 1.0))
    assert len(e.update([person(1)], 0.6, 1.0, "feet", 5.0)) == 1
    w = e.update([person(1)], 1.15, 1.0, "body", 5.0)
    assert len(w) == 2 and tuple(w[1][:2]) == pytest.approx((1.0, 0.5)) and w[1][3] < 1.0
    # Expire, cap, and no rings without people
    assert len(e.update([], 10.0, 1.0, "feet", 5.0)) == 0
    for i in range(40):
        e.update([person(100 + i)], 20.0, 0.0, "feet", 5.0)
    assert len(e.array()) == MAX_WAVES


def test_lost_people_send_no_rings_and_interval_zero_is_arrival_only():
    e = WaveEmitter()
    assert len(e.update([person(1, missing=TRACK_GRACE / 2)], 0.0, 1.0, "feet", 5.0)) == 0
    e.update([person(2)], 0.0, 0.0, "feet", 5.0)
    assert len(e.update([person(2)], 3.0, 0.0, "feet", 5.0)) == 1


# ---------------------------------------------------------------------------
# Idle look and show settings
# ---------------------------------------------------------------------------

def test_idle_fader_timeline():
    f = IdleFader()
    for _ in range(int(4.9 / 0.1)):
        assert f.update(False, 0.1, after=5.0, fade=1.0) == 0.0
    for _ in range(12):
        v = f.update(False, 0.1, after=5.0, fade=1.0)
    assert v == 1.0
    v = f.update(True, 0.1, after=5.0, fade=1.0)
    assert 0.0 < v < 1.0
    for _ in range(10):
        v = f.update(True, 0.1, after=5.0, fade=1.0)
    assert v == 0.0 and f.empty_for == 0.0


def test_idle_mix_fades_the_look_but_not_show_settings():
    base = P.normalise({"max_depth": 10, "idle_look": "calm", "presence_threshold": 30, "proc_size": 64})
    calm = P.normalise({"max_depth": 4, "presence_threshold": 90, "proc_size": 256, "idle_look": ""})
    looks = {"calm": calm}
    assert idle_mix(base, looks, 0.0) is base
    half = idle_mix(base, looks, 0.5)
    assert half["max_depth"] == 7
    full = idle_mix(base, looks, 1.0)
    assert full["max_depth"] == 4
    for k in ("presence_threshold", "proc_size", "idle_look"):
        assert full[k] == base[k]
    assert idle_mix(dict(base, idle_look="missing"), looks, 1.0)["max_depth"] == 10


def test_look_helpers_keep_show_settings():
    cur = P.normalise({"max_depth": 9, "idle_after": 45.0, "presence_threshold": 40})
    look = P.normalise({"max_depth": 3})                      # saved with default show settings
    assert set(P.look_values(look)).isdisjoint({p.key for p in P.PARAMS if not p.look})
    target = P.look_target(cur, look)
    assert target["max_depth"] == 3 and target["idle_after"] == 45.0 and target["presence_threshold"] == 40
    fader = LookFader(cur, look, 1.0, now=0.0)
    assert fader.values(1.0)["idle_after"] == 45.0 and fader.values(1.0)["max_depth"] == 3


# ---------------------------------------------------------------------------
# Tree: detail and fading follow activity
# ---------------------------------------------------------------------------

def _tree_with_noise(seed=0):
    from core.engine import OrnamentTree
    tree = OrnamentTree(2, 1, 8, 32)
    tree.max_depth, tree.still_depth, tree.animate = 8, 2, False
    h, w = tree.proc_shape
    labels = np.random.default_rng(seed).integers(0, 4, (h, w))      # impure everywhere
    palette = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0]], np.float32)
    return tree, labels, palette


def _split_levels_by_cell(tree):
    """{cell: deepest level with a split node}."""
    deepest = {}
    for level in range(tree.max_level + 1):
        s = tree.split[tree.level_slice(level)]
        cells = np.arange(len(s)) >> (level + 1)
        for c in np.unique(cells[s]):
            deepest[int(c)] = level
    return deepest


def test_tree_detail_follows_activity():
    tree, labels, palette = _tree_with_noise()
    act = np.zeros(tree.proc_shape, np.float32)
    act[:, :32] = 1.0                                   # someone in the left cell only
    tree.still_dim = 1.0
    tree.dim_color = (10, 20, 30)
    tree.conforming = False                             # (closure would carry detail over the border)
    tree.update(labels, palette, 0.1, activity=act)
    deepest = _split_levels_by_cell(tree)
    assert deepest[0] == 7                              # full detail (max_depth 8)
    assert deepest[1] == 1                              # still: nothing below still_depth 2
    n_sq = 2
    cells = np.concatenate([np.arange(n_sq << (L + 1)) >> (L + 1) for L in range(tree.max_level + 1)])
    assert np.allclose(tree.color[cells == 1], (10, 20, 30))
    assert not np.allclose(tree.color[cells == 0], (10, 20, 30))


def test_tree_without_activity_is_unchanged():
    a, labels, palette = _tree_with_noise()
    b, _, _ = _tree_with_noise()
    a.update(labels, palette, 0.1)
    b.update(labels, palette, 0.1, activity=None)
    assert np.array_equal(a.split, b.split) and np.array_equal(a.color, b.color)


def test_outline_splits_triangles_its_edge_cuts():
    from core.engine import OrnamentTree
    tree = OrnamentTree(1, 1, 8, 64)
    tree.max_depth, tree.animate = 8, False
    labels = np.zeros(tree.proc_shape, np.int64)        # one colour: nothing to split for
    palette = np.array([[0, 0, 255]], np.float32)
    outline = np.zeros(tree.proc_shape, bool)
    outline[20:50, 25:40] = True
    tree.update(labels, palette, 0.1, activity=np.ones(tree.proc_shape, np.float32))
    assert tree.leaf_count == 2
    tree.update(labels, palette, 0.1, activity=outline.astype(np.float32), outline=outline)
    assert tree.leaf_count > 40


def test_torso_and_legs_are_one_person(covered):
    t, bg, rng = learned(covered)
    f = with_person(bg, 60, y=10, h=20)                           # torso
    f = with_person(f, 61, y=36, h=22, colour=(200, 40, 40))      # legs, 6 px lower
    pres = run(t, [with_noise(f, rng) for _ in range(4)], covered)
    assert pres.count == 1
    p = pres.people[0]
    assert p.foot_y == pytest.approx(58 / CELL, abs=0.03)
