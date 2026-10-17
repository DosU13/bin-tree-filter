"""Structural rules of the bintree that must hold after every update."""

import numpy as np
import pytest

from core.engine import OrnamentEngine, load_tile
from tests.engine_cases import ROOT, frame_labels


def check_tree(tree) -> None:
    split = tree.split
    lm = tree.max_level
    offsets = tree.offsets
    sizes = tree.sizes
    for level in range(1, lm + 1):
        s = split[offsets[level] : offsets[level] + sizes[level]]
        parent = split[offsets[level - 1] : offsets[level - 1] + sizes[level - 1]]
        # A node is only split if its parent is
        assert not np.any(s & ~np.repeat(parent, 2)), f"orphan split at level {level}"
    if tree.conforming:
        for level in range(lm + 1):
            s = split[offsets[level] : offsets[level] + sizes[level]]
            nb = tree.base_nb[level]
            has = nb >= 0
            # Splitting a triangle splits its base neighbour too
            assert np.all(s[has] == s[nb[has]]), f"non-conforming split at level {level}"


@pytest.mark.parametrize("conforming", [True, False])
def test_tree_rules_hold(conforming):
    tile = load_tile(f"{ROOT}/ornam.jpg")
    eng = OrnamentEngine(tile, out_size=32, proc_size=48, max_level=9)
    eng.conforming = conforming
    eng.max_depth = 9
    for f in range(8):
        labels, pal = frame_labels("me.jpg", 48, 4, f)
        eng.update(labels, pal, 1 / 30)
        check_tree(eng.tree)
    assert eng.leaf_count > 20


def test_deterministic():
    tile = load_tile(f"{ROOT}/ornam.jpg")
    outs = []
    for _ in range(2):
        eng = OrnamentEngine(tile, out_size=48, proc_size=32, max_level=8)
        for f in range(4):
            labels, pal = frame_labels("aziz.jpg", 32, 4, f)
            eng.update(labels, pal, 1 / 30)
        outs.append(eng.render(cord=True))
    assert np.array_equal(outs[0], outs[1])
