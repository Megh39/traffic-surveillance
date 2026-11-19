
import pytest

from src.tvs.perception.tracker import TrackObservation
from evaluation.detection.annotations import (
    GroundTruthBox,
    IgnoredRegion,
)
from evaluation.tracking.matching import match_frame_tracks


def make_prediction(
    bbox,
    track_id=1,
    frame_id=1,
):
    return TrackObservation(
        frame_id=frame_id,
        internal_track_id=track_id,
        bbox=bbox,
        class_id=2,
        source_label="car",
        vehicle_type="car",
        confidence=0.9,
        detection_index=0,
        published_track_id=track_id,
    )


def make_ground_truth(
    bbox,
    target_id="1",
    frame_id=1,
):
    return GroundTruthBox(
        frame_id=frame_id,
        target_id=target_id,
        bbox=bbox,
        vehicle_type="car",
        truncation_ratio=0.1,
        occluded=False,
    )


class TestTrackingFrameMatching:

    def test_perfect_match(self):
        prediction = make_prediction((10, 20, 30, 40))
        gt = make_ground_truth((10, 20, 30, 40))

        result = match_frame_tracks([prediction], [gt])

        assert result.true_positive_count == 1
        assert result.false_positive_count == 0
        assert result.false_negative_count == 0
        assert result.matches[0].target_id == "1"
        assert result.matches[0].published_track_id == 1
        assert result.matches[0].iou == pytest.approx(1.0)

    def test_below_threshold_prediction_is_not_matched(self):
        prediction = make_prediction((100, 100, 20, 20))
        gt = make_ground_truth((10, 10, 20, 20))

        result = match_frame_tracks(
            [prediction],
            [gt],
            iou_threshold=0.5,
        )

        assert result.true_positive_count == 0
        assert result.false_positive_count == 1
        assert result.false_negative_count == 1

    def test_unmatched_prediction_in_ignored_region(self):
        prediction = make_prediction((10, 10, 20, 20))
        ignored = IgnoredRegion((0, 0, 50, 50))

        result = match_frame_tracks(
            [prediction],
            [],
            ignored_regions=[ignored],
        )

        assert result.ignored_prediction_count == 1
        assert result.false_positive_count == 0

    def test_prediction_outside_ignored_region_is_false_positive(self):
        prediction = make_prediction((100, 100, 20, 20))
        ignored = IgnoredRegion((0, 0, 50, 50))

        result = match_frame_tracks(
            [prediction],
            [],
            ignored_regions=[ignored],
        )

        assert result.ignored_prediction_count == 0
        assert result.false_positive_count == 1

    def test_rectangular_assignment_with_extra_predictions(self):
        predictions = [
            make_prediction((10, 10, 20, 20), track_id=1),
            make_prediction((100, 100, 20, 20), track_id=2),
        ]
        ground_truth = [
            make_ground_truth((10, 10, 20, 20), target_id="A"),
        ]

        result = match_frame_tracks(predictions, ground_truth)

        assert result.true_positive_count == 1
        assert result.false_positive_count == 1
        assert result.false_negative_count == 0

    def test_rectangular_assignment_with_extra_ground_truth(self):
        predictions = [
            make_prediction((10, 10, 20, 20), track_id=1),
        ]
        ground_truth = [
            make_ground_truth((10, 10, 20, 20), target_id="A"),
            make_ground_truth((100, 100, 20, 20), target_id="B"),
        ]

        result = match_frame_tracks(predictions, ground_truth)

        assert result.true_positive_count == 1
        assert result.false_positive_count == 0
        assert result.false_negative_count == 1

    def test_duplicate_prediction_ids_are_rejected(self):
        predictions = [
            make_prediction((10, 10, 20, 20), track_id=1),
            make_prediction((40, 40, 20, 20), track_id=1),
        ]

        with pytest.raises(ValueError, match="Duplicate published"):
            match_frame_tracks(predictions, [])

    def test_duplicate_ground_truth_ids_are_rejected(self):
        ground_truth = [
            make_ground_truth((10, 10, 20, 20), target_id="A"),
            make_ground_truth((40, 40, 20, 20), target_id="A"),
        ]

        with pytest.raises(ValueError, match="Duplicate ground-truth"):
            match_frame_tracks([], ground_truth)

    def test_unpublished_prediction_is_rejected(self):
        prediction = TrackObservation(
            frame_id=1,
            internal_track_id=1,
            bbox=(10, 10, 20, 20),
            class_id=2,
            source_label="car",
            vehicle_type="car",
            confidence=0.9,
            detection_index=0,
            published_track_id=None,
        )

        with pytest.raises(ValueError, match="published track ID"):
            match_frame_tracks([prediction], [])

    def test_mixed_frame_ids_are_rejected(self):
        prediction = make_prediction(
            (10, 10, 20, 20),
            frame_id=1,
        )
        gt = make_ground_truth(
            (10, 10, 20, 20),
            frame_id=2,
        )

        with pytest.raises(ValueError, match="one frame"):
            match_frame_tracks([prediction], [gt])

    def test_invalid_iou_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            match_frame_tracks([], [], iou_threshold=0)

    def test_invalid_ignore_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            match_frame_tracks(
                [],
                [],
                ignore_coverage_threshold=1.5,
            )

    def test_empty_frame(self):
        result = match_frame_tracks([], [])

        assert result.frame_id == 0
        assert result.true_positive_count == 0
        assert result.false_positive_count == 0
        assert result.false_negative_count == 0

    def test_empty_frame_preserves_explicit_frame_id(self):
        result = match_frame_tracks([], [], frame_id=542)

        assert result.frame_id == 542
        assert result.true_positive_count == 0
        assert result.false_positive_count == 0
        assert result.false_negative_count == 0
