"""Matching a saved projector output to a connected display."""

import pytest

pytest.importorskip("PySide6")

from align_perform.runtime.screens import pick_screen
from core.scene import OutputSpec

EPSON = ("16843009", "EPSON PJ")          # placeholder serial: every Epson reports it
LAPTOP = ("", r"\\.\DISPLAY1")


def epson(index):
    return OutputSpec(kind="screen", screen_serial=EPSON[0], screen_name=EPSON[1], screen_index=index)


def test_two_identical_projectors_go_to_their_own_displays():
    keys = [LAPTOP, EPSON, EPSON]
    assert pick_screen(epson(1), keys) == 1
    assert pick_screen(epson(2), keys) == 2


def test_a_unique_serial_wins_over_the_index():
    keys = [LAPTOP, ("111", "PJ A"), ("222", "PJ B")]
    out = OutputSpec(kind="screen", screen_serial="222", screen_name="PJ B", screen_index=1)
    assert pick_screen(out, keys) == 2                     # display order changed since it was saved


def test_name_then_index():
    keys = [LAPTOP, ("", "PJ A"), ("", "PJ B")]
    assert pick_screen(OutputSpec(kind="screen", screen_name="PJ B", screen_index=1), keys) == 2
    assert pick_screen(OutputSpec(kind="screen", screen_name="GONE", screen_index=1), keys) == 1
    assert pick_screen(OutputSpec(kind="screen", screen_name="GONE", screen_index=5), keys) is None


def test_identical_displays_whose_index_moved_fall_back_to_the_first():
    assert pick_screen(epson(0), [EPSON, LAPTOP, EPSON]) == 0
    assert pick_screen(epson(1), [EPSON, LAPTOP, EPSON]) == 0


def test_without_index_fallback_only_a_serial_or_name_counts():
    keys = [LAPTOP, ("", "PJ A")]
    gone = OutputSpec(kind="screen", screen_name="GONE", screen_index=1)
    assert pick_screen(gone, keys, by_index=False) is None
    assert pick_screen(epson(2), [LAPTOP, EPSON, EPSON], by_index=False) == 2
