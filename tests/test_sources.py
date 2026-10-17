import os
import time

import numpy as np
import pytest

from core.scene import SourceSpec
from core.sources import (FrameReader, LiveSource, SourceError, SyncSource, kind_for_path, make_reader,
                          open_scene_source, probe)
from tests.engine_cases import ROOT

CLIP = os.path.join(ROOT, "scenes", "media", "cam_left.mp4")


def wait_for(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_probe_video_and_image():
    assert probe(SourceSpec(kind="video", path=CLIP)) == (480, 360, 15.0)
    w, h, _ = probe(SourceSpec(kind="image", path=os.path.join(ROOT, "me.jpg")))
    assert w > 0 and h > 0
    with pytest.raises(SourceError):
        probe(SourceSpec(kind="video", path="missing.mp4"))


def test_kind_for_path():
    assert kind_for_path("a.MP4") == "video"
    assert kind_for_path("b.gif") == "gif"
    assert kind_for_path("c.jpeg") == "image"


def test_live_video_plays_at_its_frame_rate():
    src = open_scene_source(SourceSpec(kind="video", path=CLIP), realtime=True)
    try:
        assert wait_for(lambda: src.latest()[1] > 0)
        start = src.latest()[1]
        time.sleep(1.0)
        got = src.latest()[1] - start
        assert 10 <= got <= 20          # 15 fps clip
        assert src.status == "ok"
    finally:
        src.stop()
    assert src.status == "stopped"


def test_sync_source_reads_one_frame_per_call():
    src = open_scene_source(SourceSpec(kind="video", path=CLIP), realtime=False)
    ids = [src.latest()[1] for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5]
    src.stop()


class FlakyReader(FrameReader):
    """Fails to open twice, then delivers 10 frames, then dies, then works again."""

    frame_interval = 0.01

    def __init__(self):
        self.opens = 0
        self.reads = 0

    def open(self):
        self.opens += 1
        if self.opens <= 2:
            raise SourceError("not plugged in")
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.opens == 3 and self.reads > 10:
            return None
        return np.full((4, 4, 3), self.reads, np.uint8)


def test_live_source_reconnects():
    reader = FlakyReader()
    src = LiveSource(reader, name="flaky", max_backoff=0.2)
    src.start()
    try:
        assert wait_for(lambda: reader.opens >= 4, timeout=10)
        assert wait_for(lambda: src.status == "ok")
        assert src.reconnects >= 1
        frame, fid = src.latest()
        assert frame is not None and fid > 10
    finally:
        src.stop()


def test_stalled_status():
    class Slow(FrameReader):
        def open(self):
            pass

        def read(self):
            time.sleep(0.05)
            return np.zeros((2, 2, 3), np.uint8) if not getattr(self, "hang", False) else None

    reader = Slow()
    src = LiveSource(reader, stall_timeout=0.3).start()
    try:
        assert wait_for(lambda: src.status == "ok")
        reader.hang = True
        assert wait_for(lambda: src.status in ("stalled", "reconnecting"), timeout=3)
    finally:
        src.stop()


def test_missing_file_keeps_retrying_without_crashing():
    src = open_scene_source(SourceSpec(kind="video", path="nope.mp4"), realtime=True)
    try:
        assert wait_for(lambda: src.status == "reconnecting")
        assert src.latest() == (None, 0)
        assert "not found" in src.error
    finally:
        src.stop()


def test_image_reader_noise():
    r = make_reader(SourceSpec(kind="image", path=os.path.join(ROOT, "me.jpg"), noise=4))
    r.open()
    a, b = r.read(), r.read()
    assert a.shape == b.shape and not np.array_equal(a, b)
