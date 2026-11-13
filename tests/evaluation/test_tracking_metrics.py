
import pytest

from evaluation.detection.annotations import GroundTruthBox
from evaluation.tracking.matching import (
    FrameTrackingMatchResult,
    TrackMatch,
)
from evaluation.tracking.metrics import compute_tracking_metrics


def make_gt(target_id="A", frame_id=1):
    return GroundTruthBox(
        frame_id=frame_id,
        target_id=target_id,
        bbox=(10, 10, 20, 20),
        vehicle_type="car",
        truncation_ratio=0.0,
        occluded=False,
    )


def make_frame(
    frame_id,
    matches=(),
    unmatched_gt=(),
    unmatched_predictions=(),
):
    return FrameTrackingMatchResult(
        frame_id=frame_id,
        matches=tuple(matches),
        unmatched_ground_truth=tuple(unmatched_gt),
        unmatched_predictions=tuple(unmatched_predictions),
        ignored_predictions=(),
    )


class TestTrackingMetrics:

    def test_perfect_tracking(self):
        results = [
            make_frame(
                1,
                [TrackMatch("A", 1, 1.0)],
            ),
            make_frame(
                2,
                [TrackMatch("A", 1, 1.0)],
            ),
        ]

        metrics = compute_tracking_metrics(results)

        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.identity_switches == 0
        assert metrics.fragmentations == 0
        assert metrics.mota == pytest.approx(1.0)
        assert metrics.motp == pytest.approx(1.0)
        assert metrics.idf1 == pytest.approx(1.0)

    def test_identity_switch(self):
        results = [
            make_frame(1, [TrackMatch("A", 1, 1.0)]),
            make_frame(2, [TrackMatch("A", 2, 1.0)]),
        ]

        metrics = compute_tracking_metrics(results)

        assert metrics.identity_switches == 1
        assert metrics.fragmentations == 0
        assert metrics.mota == pytest.approx(0.5)
        assert metrics.idf1 == pytest.approx(0.5)

    def test_fragmentation(self):
        results = [
            make_frame(1, [TrackMatch("A", 1, 1.0)]),
            make_frame(2, unmatched_gt=[make_gt("A", 2)]),
            make_frame(3, [TrackMatch("A", 1, 1.0)]),
        ]

        metrics = compute_tracking_metrics(results)

        assert metrics.fragmentations == 1
        assert metrics.identity_switches == 0
        assert metrics.false_negatives == 1
        assert metrics.mota == pytest.approx(2 / 3)
        assert metrics.idf1 == pytest.approx(0.8)

    def test_false_positive_and_false_negative(self):
        results = [
            make_frame(
                1,
                unmatched_gt=[make_gt("A", 1)],
                unmatched_predictions=[object()],
            ),
        ]

        metrics = compute_tracking_metrics(results)

        assert metrics.true_positives == 0
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.mota == pytest.approx(-1.0)
        assert metrics.idf1 == 0.0

    def test_global_identity_assignment(self):
        results = [
            make_frame(
                1,
                [
                    TrackMatch("A", 1, 1.0),
                    TrackMatch("B", 2, 1.0),
                ],
            ),
            make_frame(
                2,
                [
                    TrackMatch("A", 2, 1.0),
                    TrackMatch("B", 1, 1.0),
                ],
            ),
        ]

        metrics = compute_tracking_metrics(results)

        assert metrics.true_positives == 4
        assert metrics.identity_switches == 2
        assert metrics.id_true_positives == 2
        assert metrics.id_false_positives == 2
        assert metrics.id_false_negatives == 2
        assert metrics.idf1 == pytest.approx(0.5)

    def test_empty_results(self):
        metrics = compute_tracking_metrics([])

        assert metrics.true_positives == 0
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.idf1 == 0.0
        assert metrics.mota == 0.0
        assert metrics.motp == 0.0

    def test_duplicate_frame_ids_are_rejected(self):
        results = [
            make_frame(1, [TrackMatch("A", 1, 1.0)]),
            make_frame(1, [TrackMatch("A", 1, 1.0)]),
        ]

        with pytest.raises(ValueError, match="Duplicate frame IDs"):
            compute_tracking_metrics(results)
    def test_ignored_predictions_are_counted(self):
        from evaluation.tracking.matching import (
            TrackObservation,
        )

        ignored = TrackObservation(
            frame_id=1,
            internal_track_id=1,
            bbox=(10, 10, 20, 20),
            class_id=2,
            source_label="car",
            vehicle_type="car",
            confidence=0.9,
            detection_index=0,
            published_track_id=1,
        )

        result = FrameTrackingMatchResult(
            frame_id=1,
            matches=(),
            unmatched_ground_truth=(),
            unmatched_predictions=(),
            ignored_predictions=(ignored,),
        )

        metrics = compute_tracking_metrics([result])

        assert metrics.ignored_predictions == 1
        assert metrics.prediction_detections == 0
        assert metrics.false_positives == 0