"""Multi-cell stages: seams, conforming across cell borders, renderer mappings."""

import os

import cv2
import numpy as np
import pytest

from core.engine import NumpyRenderer, OrnamentEngine, OrnamentTree, load_tile
from tests.engine_cases import PALETTE, ROOT
from tests.test_engine_invariants import check_tree

TILE = load_tile(os.path.join(ROOT, "ornam.jpg"))


def wide_labels(cols: int, rows: int, proc: int, frame: int = 0) -> np.ndarray:
    """Nearest-palette labels of a photo stretched over the whole stage."""
    img = cv2.imread(os.path.join(ROOT, "test_girl_sq.jpg"))
    img = np.roll(img, frame * 7, axis=1)
    small = cv2.resize(img, (cols * proc, rows * proc), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = ((small[:, :, None, :] - PALETTE[None, None]) ** 2).sum(axis=3)
    return d.argmin(axis=2).astype(np.int64)


def run_tree(cols, rows, proc=32, levels=9, frames=5, **settings):
    tree = OrnamentTree(cols, rows, levels, proc)
    tree.max_depth = levels
    for k, v in settings.items():
        setattr(tree, k, v)
    for f in range(frames):
        tree.update(wide_labels(cols, rows, proc, f), PALETTE, 1 / 30)
    return tree


@pytest.mark.parametrize("cols,rows", [(3, 1), (2, 2), (1, 3)])
def test_rules_hold_across_cells(cols, rows):
    tree = run_tree(cols, rows)
    check_tree(tree)
    assert tree.leaf_count > 10 * cols * rows


def test_base_neighbours_cross_cell_borders():
    tree = OrnamentTree(3, 2, 6, 16)
    lvl1 = tree.base_nb[1]
    # Level-1 node index -> cell = index >> 2 (4 nodes per cell at level 1)
    cells = np.arange(len(lvl1)) >> 2
    has = lvl1 >= 0
    crossing = cells[has] != cells[lvl1[has]]
    # 3x2 grid has 7 interior cell edges, each shared by one level-1 pair
    assert crossing.sum() == 2 * 7
    # Border edges have no neighbour: 10 outer cell edges
    assert (~has).sum() == 10


def test_flat_input_gives_root_leaves_only():
    tree = OrnamentTree(4, 2, 8, 16)
    labels = np.zeros(tree.proc_shape, dtype=np.int64)
    tree.update(labels, PALETTE, 1 / 30)
    assert tree.leaf_count == 2 * 4 * 2


@pytest.mark.parametrize("cord", [False, True])
def test_split_tiles_render_like_the_whole_stage(cord):
    tree = run_tree(3, 1, proc=32, levels=8, frames=4)
    s = 64
    whole = NumpyRenderer(tree, (3 * s, s), TILE).render((181, 217, 253), cord=cord)
    tiles = []
    for cx in range(3):
        m = np.array([[1.0, 0, cx], [0, 1.0, 0], [0, 0, 1]])
        tiles.append(NumpyRenderer(tree, (s, s), TILE, norm_to_cell=m).render((181, 217, 253), cord=cord))
    joined = np.concatenate(tiles, axis=1)
    diff = np.abs(joined.astype(int) - whole.astype(int)).max(axis=2)
    if not cord:
        assert diff.max() == 0
    else:
        # A tile can't see the pixel left of its first column, so a cord line that
        # falls exactly on the tile border is missing there; nowhere else may differ.
        assert set(np.nonzero(diff)[1].tolist()) <= {s, 2 * s}


def test_outside_stage_is_black():
    tree = run_tree(2, 1, frames=2)
    # Output covers x from -0.5 to 2.5 cells
    m = np.array([[3.0, 0, -0.5], [0, 1.0, 0], [0, 0, 1]])
    img = NumpyRenderer(tree, (96, 32), TILE, norm_to_cell=m).render((181, 217, 253))
    assert img[:, :14].max() == 0 and img[:, -14:].max() == 0
    assert img[:, 20:76].min(axis=(0, 1)).max() > 0


def test_homography_mapping_renders():
    tree = run_tree(2, 2, frames=2)
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32([[0.1, 0.0], [2.0, 0.2], [1.8, 2.0], [0.0, 1.7]])
    m = cv2.getPerspectiveTransform(src, dst)
    img = NumpyRenderer(tree, (64, 64), TILE, norm_to_cell=m).render((181, 217, 253), wireframe=True)
    assert img.shape == (64, 64, 3)
    assert (img == 0).all(axis=2).mean() < 0.5


def test_engine_wrapper_matches_tree_plus_renderer():
    eng = OrnamentEngine(TILE, out_size=48, proc_size=32, max_level=8)
    tree = OrnamentTree(1, 1, 8, 32)
    rend = NumpyRenderer(tree, 48, TILE)
    for f in range(3):
        labels = wide_labels(1, 1, 32, f)
        eng.update(labels, PALETTE, 1 / 30)
        tree.update(labels, PALETTE, 1 / 30)
    assert np.array_equal(eng.render(), rend.render(eng.background))
