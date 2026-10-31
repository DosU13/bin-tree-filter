import pytest

from core import params as P


def test_keys_unique_and_defaults_valid():
    keys = [p.key for p in P.PARAMS]
    assert len(keys) == len(set(keys))
    for p in P.PARAMS:
        assert p.kind in P.KINDS
        assert P.coerce(p.key, p.default) == p.default, p.key
        if p.kind in ("int", "float"):
            assert p.min <= p.default <= p.max, p.key
            assert p.step and p.step > 0, p.key


@pytest.mark.parametrize("mode", P.MODES)
def test_hotkeys_do_not_collide_within_a_mode(mode):
    seen = {}
    for p in P.PARAMS:
        if p.mode not in (None, mode):
            continue
        for k in p.keys:
            assert k not in seen, f"{k!r} used by {seen[k]} and {p.key}"
            seen[k] = p.key


def test_live_py_hotkeys_are_covered():
    # Every parameter key live.py handles (see its docstring) exists in the registry
    colour = P.hotkeys("colour")
    depth = P.hotkeys("depth")
    for k in "dDmMhHcCpP[];'nNtTbBkvixge":
        assert k in colour, k
    for k in "aAzZfb":
        assert k in depth, k
    assert depth["c"] == ("depth_layers", -1) and depth["k"] == ("depth_palette", +1)


def test_coerce_clamps_and_converts():
    assert P.coerce("max_depth", 99) == 12
    assert P.coerce("max_depth", "3") == 3
    assert P.coerce("threshold", -1) == 0.01
    assert P.coerce("cord", "true") is True
    assert P.coerce("background", (181, 217, 253)) == "#fdd9b5"
    with pytest.raises(ValueError):
        P.coerce("polarity", "sideways")


def test_normalise_fills_defaults_and_fixes_ranges():
    v = P.normalise({"min_depth": 9, "max_depth": 5, "unknown": 1, "polarity": "bad"})
    assert v["max_depth"] == 5 and v["min_depth"] == 4
    assert "unknown" not in v
    assert v["polarity"] == "alternate"
    assert set(v) == set(P.BY_KEY)


def test_nudge():
    v = P.defaults()
    assert P.nudge(v, "max_depth", +1) == 9
    assert P.nudge(v, "threshold", -1) == 0.74
    assert P.nudge(v, "cord", +1) is True
    assert P.nudge(v, "polarity", +1) == "none"
    assert P.nudge(v, "depth_fg", +1) == P.BY_KEY["depth_fg"].choices[1]


def test_interpolate():
    a = P.defaults()
    b = dict(a, max_depth=12, threshold=0.95, cord=True, background="#000000")
    mid = P.interpolate(a, b, 0.5)
    assert mid["max_depth"] == 10
    assert abs(mid["threshold"] - 0.85) < 1e-9
    assert mid["cord"] is True
    assert P.interpolate(a, b, 0.0) == a
    assert P.interpolate(a, b, 1.0)["background"] == "#000000"
