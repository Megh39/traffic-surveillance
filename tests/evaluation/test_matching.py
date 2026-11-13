import pytest

from tvs.events.detection import Detection
from evaluation.detection.annotations import (
    GroundTruthBox,
    IgnoredRegion,
)
from evaluation.detection.matching import (
    intersection_over_union,
    intersection_over_prediction,
    match_frame_detections,
)


def make_detection(bbox, confidence=0.9, frame_id=1):
    return Detection(
        frame_id=frame_id,
        bbox=bbox,
        class_id=2,
        source_label="car",
        vehicle_type="car",
        confidence=confidence,
    )


def make_ground_truth(bbox, target_id="1", frame_id=1):
    return GroundTruthBox(
        frame_id=frame_id,
        target_id=target_id,
        bbox=bbox,
        vehicle_type="car",
        truncation_ratio=0.1,
        occluded=False,
    )


class TestIntersectionOverUnion:

    def test_identical_boxes_have_iou_one(self):
        box = (10.0, 20.0, 30.0, 40.0)

        assert intersection_over_union(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        assert intersection_over_union(
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 20.0, 10.0, 10.0),
        ) == 0.0

    def test_partial_overlap(self):
        assert intersection_over_union(
            (0.0, 0.0, 10.0, 10.0),
            (5.0, 0.0, 10.0, 10.0),
        ) == pytest.approx(1 / 3)

    def test_invalid_box_is_rejected(self):
        with pytest.raises(ValueError):
            intersection_over_union(
                (0.0, 0.0, 0.0, 10.0),
                (0.0, 0.0, 10.0, 10.0),
            )


class TestIgnoredRegionOverlap:

    def test_fully_covered_prediction(self):
        assert intersection_over_prediction(
            (10.0, 10.0, 20.0, 20.0),
            (0.0, 0.0, 50.0, 50.0),
        ) == pytest.approx(1.0)

    def test_half_covered_prediction(self):
        assert intersection_over_prediction(
            (0.0, 0.0, 20.0, 20.0),
            (0.0, 0.0, 10.0, 20.0),
        ) == pytest.approx(0.5)


class TestFrameMatching:

    def test_perfect_detection_is_true_positive(self):
        detection = make_detection((10.0, 20.0, 30.0, 40.0))
        gt = make_ground_truth((10.0, 20.0, 30.0, 40.0))

        result = match_frame_detections([detection], [gt])

        assert result.true_positive_count == 1
        assert result.false_positive_count == 0
        assert result.ground_truth_count == 1

    def test_duplicate_detection_is_false_positive(self):
        gt = make_ground_truth((10.0, 20.0, 30.0, 40.0))

        detections = [
            make_detection((10.0, 20.0, 30.0, 40.0), 0.95),
            make_detection((10.0, 20.0, 30.0, 40.0), 0.80),
        ]

        result = match_frame_detections(detections, [gt])

        assert result.true_positive_count == 1
        assert result.false_positive_count == 1

    def test_predictions_are_processed_by_confidence(self):
        gt = make_ground_truth((10.0, 20.0, 30.0, 40.0))

        detections = [
            make_detection((10.0, 20.0, 30.0, 40.0), 0.60),
            make_detection((10.0, 20.0, 30.0, 40.0), 0.95),
        ]

        result = match_frame_detections(detections, [gt])

        assert result.predictions[0].confidence == 0.95
        assert result.predictions[0].is_true_positive is True
        assert result.predictions[1].is_true_positive is False

    def test_unmatched_prediction_is_false_positive(self):
        detection = make_detection((100.0, 100.0, 20.0, 20.0))
        gt = make_ground_truth((10.0, 10.0, 20.0, 20.0))

        result = match_frame_detections([detection], [gt])

        assert result.false_positive_count == 1

    def test_prediction_in_ignored_region_is_ignored(self):
        detection = make_detection((10.0, 10.0, 20.0, 20.0))

        ignored = IgnoredRegion(
            bbox=(0.0, 0.0, 50.0, 50.0)
        )

        result = match_frame_detections(
            [detection],
            [],
            [ignored],
        )

        assert result.ignored_prediction_count == 1
        assert result.false_positive_count == 0

    def test_prediction_outside_ignored_region_is_false_positive(self):
        detection = make_detection((100.0, 100.0, 20.0, 20.0))

        ignored = IgnoredRegion(
            bbox=(0.0, 0.0, 50.0, 50.0)
        )

        result = match_frame_detections(
            [detection],
            [],
            [ignored],
        )

        assert result.false_positive_count == 1

    def test_class_agnostic_matching(self):
        detection = make_detection((10.0, 20.0, 30.0, 40.0))

        gt = GroundTruthBox(
            frame_id=1,
            target_id="1",
            bbox=(10.0, 20.0, 30.0, 40.0),
            vehicle_type="van",
            truncation_ratio=0.1,
            occluded=False,
        )

        result = match_frame_detections([detection], [gt])

        assert result.true_positive_count == 1

    def test_frame_id_mismatch_is_rejected(self):
        detection = make_detection(
            (10.0, 20.0, 30.0, 40.0),
            frame_id=1,
        )
        gt = make_ground_truth(
            (10.0, 20.0, 30.0, 40.0),
            frame_id=2,
        )

        with pytest.raises(ValueError, match="one frame"):
            match_frame_detections([detection], [gt])

    def test_invalid_iou_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            match_frame_detections([], [], iou_threshold=0)