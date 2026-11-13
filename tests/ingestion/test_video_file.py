
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tvs.ingestion.video_file import VideoFileSource


class FakeCapture:
    def __init__(self, fps=30.0, frame_count=3, opened=True):
        self.fps = fps
        self.frame_count = frame_count
        self.opened = opened
        self.current = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def read(self):
        if self.current >= self.frame_count:
            return False, None

        self.current += 1
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        return True, image

    def release(self):
        self.released = True


def test_video_ingestion_smoke():
    source = VideoFileSource(
        "data/videos/01.mp4",
        max_frames=30,
    )

    frames = list(source.frames())

    assert len(frames) == 30
    assert frames[0].frame_id == 1
    assert frames[-1].frame_id == 30
    assert source.frames_total == 107934
    assert source.fps_effective == pytest.approx(29.9818, abs=0.01)
    assert frames[-1].elapsed_seconds == pytest.approx(29 / 29.9818, abs=0.001)


def test_fps_fallback_when_metadata_is_zero(tmp_path):
    video = tmp_path / "fake.mp4"
    video.touch()
    capture = FakeCapture(fps=0.0, frame_count=3)

    with patch("tvs.ingestion.video_file.cv2.VideoCapture", return_value=capture):
        source = VideoFileSource(video, fallback_fps=25.0)
        frames = list(source.frames())

    assert source.fps_effective == 25.0
    assert len(frames) == 3
    assert frames[1].elapsed_seconds == pytest.approx(1 / 25.0)
    assert capture.released


def test_stride_preserves_original_frame_ids(tmp_path):
    video = tmp_path / "fake.mp4"
    video.touch()
    capture = FakeCapture(fps=30.0, frame_count=6)

    with patch("tvs.ingestion.video_file.cv2.VideoCapture", return_value=capture):
        source = VideoFileSource(video, stride=2)
        frames = list(source.frames())

    assert [frame.frame_id for frame in frames] == [1, 3, 5]
    assert source.stats.frames_processed == 3
    assert capture.released



def test_missing_video_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoFileSource(tmp_path / "missing.mp4")


def test_video_that_cannot_be_opened_raises_error(tmp_path):
    video = tmp_path / "invalid.mp4"
    video.touch()
    capture = FakeCapture(opened=False)

    with patch("tvs.ingestion.video_file.cv2.VideoCapture", return_value=capture):
        source = VideoFileSource(video)

        with pytest.raises(RuntimeError, match="Could not open video"):
            list(source.frames())

    assert capture.released


@pytest.mark.parametrize("fps", [0.0, -1.0, 121.0])
def test_implausible_fps_uses_fallback(tmp_path, fps):
    video = tmp_path / "fake.mp4"
    video.touch()
    capture = FakeCapture(fps=fps, frame_count=2)

    with patch("tvs.ingestion.video_file.cv2.VideoCapture", return_value=capture):
        source = VideoFileSource(video, fallback_fps=24.0)
        frames = list(source.frames())

    assert source.fps_effective == 24.0
    assert frames[1].elapsed_seconds == pytest.approx(1 / 24.0)
