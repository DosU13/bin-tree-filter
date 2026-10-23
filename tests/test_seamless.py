"""Seamless colours: different colours only meet along ornament edges without ink."""

import os

import numpy as np
import pytest

from core.engine import OrnamentTree, _finest_edge_pairs, load_tile, tile_edge_ink
from tests.engine_cases import ROOT

PALETTE = np.array([[40, 40, 200], [200, 120, 30], [30, 160, 60], [181, 217, 253]], np.float32)


def leaf_levels(tree):
    lm = tree.max_level
    cells = tree._cells
    lvl = np.zeros(tree.n_cells, np.int64)
    full = np.ones(tree.n_cells, bool)
    for level in range(lm):
        full &= tree.split[tree.offsets[level] + (cells >> (lm - level))]
        lvl += full
    return lvl


def harsh(tree, ink=(True, False)):
    """Length (in finest edges) of leaf borders where both sides draw ink in different colours."""
    f, g, diag, _ = _finest_edge_pairs(tree.max_level, tree.cols, tree.rows)
    lm = tree.max_level
    lvl = leaf_levels(tree)
    lf, lg = lvl[f], lvl[g]
    cf = tree.dom[tree.offsets[lf] + (f >> (lm - lf))]
    cg = tree.dom[tree.offsets[lg] + (g >> (lm - lg))]
    saved, tree.edge_ink = tree.edge_ink, ink
    bad = (cf != cg) & tree._edge_inked(lf, diag) & tree._edge_inked(lg, diag)
    tree.edge_ink = saved
    flat = (cf == 3) | (cg == 3)                     # label 3 is drawn in the background colour
    return int((bad & ~flat).sum()) // 2


def scene_labels(tree, kind):
    h, w = tree.proc_shape
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "vertical":
        return (xx > 77).astype(np.int64)
    labels = ((xx - 100) ** 2 + (yy - 60) ** 2 < 45 ** 2).astype(np.int64)
    labels[(xx > 170) & (yy > 30)] = 2
    labels[(yy < 12) & (xx < 60)] = 3                # some "uncovered" background
    return labels


def make_tree(seamless, max_depth=9, polarity="alternate"):
    tree = OrnamentTree(2, 1, 12, 128)
    tree.max_depth, tree.animate, tree.threshold = max_depth, False, 0.8
    tree.polarity = polarity
    tree.edge_ink = (True, False) if seamless else None
    return tree


def test_tile_edge_ink():
    tile = load_tile(os.path.join(ROOT, "ornam.jpg"))
    assert tile_edge_ink(tile) == (True, False)          # ink along the hypotenuse, clear legs
    assert tile_edge_ink(1.0 - tile) == (False, True)
    assert tile_edge_ink(np.zeros((64, 64), np.float32)) is None


def test_finest_edge_pairs_cover_every_inner_edge():
    f, g, diag, sep = _finest_edge_pairs(4, 2, 1)
    n = 2 * 2 ** 5
    deg = np.bincount(f, minlength=n)
    assert set(np.unique(deg)) <= {2, 3}                # border triangles have one edge outside
    assert np.all(sep[:-1] <= sep[1:])
    assert (np.sort(np.stack([f, g], 1), axis=1) == np.sort(np.stack([g, f], 1), axis=1)).all()


@pytest.mark.parametrize("kind", ["vertical", "circle"])
@pytest.mark.parametrize("max_depth", [9, 10])
@pytest.mark.parametrize("polarity", ["alternate", "none"])
def test_no_inked_seams_between_colours(kind, max_depth, polarity):
    plain, seam = make_tree(False, max_depth, polarity), make_tree(True, max_depth, polarity)
    labels = scene_labels(plain, kind)
    for _ in range(14):
        plain.update(labels, PALETTE, 1 / 30)
        seam.update(labels, PALETTE, 1 / 30)
    assert harsh(plain) > 0 or polarity != "alternate"
    assert harsh(seam) == 0


def test_seamless_is_stable_once_settled():
    tree = make_tree(True)
    labels = scene_labels(tree, "circle")
    for _ in range(20):
        tree.update(labels, PALETTE, 1 / 30)
    split, dom = tree.split.copy(), tree.dom.copy()
    for _ in range(5):
        tree.update(labels, PALETTE, 1 / 30)
    assert np.array_equal(split, tree.split) and np.array_equal(dom, tree.dom)


def test_seamless_with_presence_depth_caps():
    tree = make_tree(True, max_depth=10)
    tree.still_depth, tree.still_dim = 4, 0.0
    labels = scene_labels(tree, "circle")
    act = np.zeros(tree.proc_shape, np.float32)
    act[:, :60] = 1.0                                  # someone on the left; the rest is capped at 4
    for _ in range(14):
        tree.update(labels, PALETTE, 1 / 30, activity=act)
    assert harsh(tree) == 0
    assert leaf_levels(tree)[(tree._cells >> 12) == 3].max() <= 4      # right cell's lower half: still


def test_pipeline_finds_the_tile_ink():
    from core import params as P
    from core.pipeline import Pipeline
    from core.scene import Scene
    scene = Scene.load(os.path.join(ROOT, "scenes", "example.json"))
    scene.params = P.normalise(dict(scene.params, seamless=True))
    pipe = Pipeline(scene, realtime=False)
    try:
        pipe.step(1 / 15)
        assert pipe.tree.edge_ink == (True, False)
        pipe.set_params(dict(pipe.params, seamless=False))
        pipe.step(1 / 15)
        assert pipe.tree.edge_ink is None
    finally:
        pipe.close()
