from pathlib import Path
from unittest.mock import patch
from src.tvs.ingestion.image_sequence import ImageSequenceSource
FIXTURE = Path("tests/fixtures/ingestion/MVI_20011")
DATASET = Path("data/ua_detrac/DETRAC-Images/DETRAC-Images/MVI_20011")


def test_mvi20011_ingestion():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        max_skip_ratio=0.05,
    )

    frames = list(source.frames())

    assert len(frames) == 664
    assert frames[0].frame_id == 1
    assert frames[-1].frame_id == 664

    assert source.stats.frames_discovered == 664
    assert source.stats.frames_processed == 664
    assert source.stats.frames_skipped == 0
    assert source.stats.skip_ratio == 0.0


def test_frame_timing():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
    )

    frames = list(source.frames())

    assert frames[0].elapsed_seconds == 0.0
    assert frames[1].elapsed_seconds == 1 / 30
    assert frames[-1].elapsed_seconds == 663 / 30


from unittest.mock import patch

import pytest


def test_unreadable_frame_is_skipped():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        max_skip_ratio=0.05,
    )

    original_imread = __import__("cv2").imread

    def fake_imread(path):
        if str(path).endswith("img00010.jpg"):
            return None
        return original_imread(path)

    with patch("tvs.ingestion.image_sequence.cv2.imread", side_effect=fake_imread):
        frames = list(source.frames())

    assert len(frames) == 663
    assert source.stats.frames_discovered == 664
    assert source.stats.frames_processed == 663
    assert source.stats.frames_skipped == 1
    assert source.stats.skip_ratio == 1 / 664

    frame_ids = [frame.frame_id for frame in frames]
    assert 10 not in frame_ids


def test_excessive_unreadable_frames_abort():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        max_skip_ratio=0.05,
    )

    original_imread = __import__("cv2").imread

    def fake_imread(path):
        return None if "img000" in str(path) else original_imread(path)

    with patch("tvs.ingestion.image_sequence.cv2.imread", side_effect=fake_imread):
        with pytest.raises(
            RuntimeError,
            match="Unreadable-frame ratio exceeded",
        ):
            list(source.frames())


def test_stride_preserves_original_frame_ids():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        stride=2,
    )

    frames = list(source.frames())

    assert len(frames) == 332
    assert frames[0].frame_id == 1
    assert frames[1].frame_id == 3
    assert frames[-1].frame_id == 663


def test_invalid_stride_is_rejected():
    with pytest.raises(ValueError, match="stride"):
        ImageSequenceSource(
            source_dir=DATASET,
            fps=30.0,
            stride=0,
        )


def test_frame_ids_are_monotonic():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
    )

    frames = list(source.frames())
    frame_ids = [frame.frame_id for frame in frames]

    assert frame_ids == sorted(frame_ids)

    for previous, current in zip(frame_ids, frame_ids[1:]):
        assert current > previous


def test_resolution_is_validated():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        expected_width=960,
        expected_height=540,
    )

    frames = list(source.frames())

    assert len(frames) == 664


def test_resolution_mismatch_does_not_abort():
    source = ImageSequenceSource(
        source_dir=DATASET,
        fps=30.0,
        expected_width=1280,
        expected_height=720,
    )

    frames = list(source.frames())

    assert len(frames) == 664
def test_real_corrupted_image_is_skipped():
    source = ImageSequenceSource(
        source_dir=FIXTURE,
        fps=30.0,
        max_skip_ratio=0.05,
    )

    frames = list(source.frames())

    assert len(frames) == 30
    assert source.stats.frames_discovered == 31
    assert source.stats.frames_processed == 30
    assert source.stats.frames_skipped == 1

    frame_ids = [frame.frame_id for frame in frames]

    assert 31 not in frame_ids
    assert frame_ids == list(range(1, 31))
def test_shuffled_input_raises_monotonicity_error():
    source = ImageSequenceSource(
        source_dir=FIXTURE,
        fps=30.0,
    )

    original_image_paths = source._image_paths()

    shuffled_paths = original_image_paths.copy()
    shuffled_paths[0], shuffled_paths[1] = (
        shuffled_paths[1],
        shuffled_paths[0],
    )

    with patch.object(
        source,
        "_image_paths",
        return_value=shuffled_paths,
    ):
        with pytest.raises(
            ValueError,
            match="Frame IDs must be strictly monotonic",
        ):
            list(source.frames())