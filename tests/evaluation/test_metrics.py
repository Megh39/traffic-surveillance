import pytest

from evaluation.detection.matching import (
    FrameMatchResult,
    PredictionMatch,
)
from evaluation.detection.metrics import calculate_average_precision


def make_result(predictions=(), ground_truth_count=1):
    return FrameMatchResult(
        predictions=tuple(predictions),
        ground_truth_count=ground_truth_count,
    )


class TestAveragePrecision:

    def test_perfect_detection(self):
        result = make_result(
            [
                PredictionMatch(0.95, True, "1"),
            ]
        )

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == pytest.approx(1.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.false_negative_count == 0

    def test_false_positive_before_true_positive(self):
        result = make_result(
            [
                PredictionMatch(0.99, False),
                PredictionMatch(0.90, True, "1"),
            ]
        )

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == pytest.approx(0.5)
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(1.0)

    def test_false_positive_after_true_positive(self):
        result = make_result(
            [
                PredictionMatch(0.95, True, "1"),
                PredictionMatch(0.80, False),
            ]
        )

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == pytest.approx(1.0)
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == pytest.approx(1.0)

    def test_predictions_are_ranked_across_frames(self):
        frame_one = make_result(
            [
                PredictionMatch(0.60, True, "1"),
            ]
        )

        frame_two = make_result(
            [
                PredictionMatch(0.95, False),
            ],
            ground_truth_count=0,
        )
        metrics = calculate_average_precision(
            [
                frame_one,
                frame_two,
            ]
        )

        assert metrics.average_precision == pytest.approx(0.5)
        assert metrics.true_positive_count == 1
        assert metrics.false_positive_count == 1

    def test_ignored_predictions_are_excluded(self):
        result = make_result(
            [
                PredictionMatch(0.99, None),
                PredictionMatch(0.90, True, "1"),
            ]
        )

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == pytest.approx(1.0)
        assert metrics.prediction_count == 1
        assert metrics.ignored_prediction_count == 1

    def test_no_predictions_returns_zero_metrics(self):
        result = make_result([], ground_truth_count=5)

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.false_negative_count == 5

    def test_multiple_ground_truth_objects(self):
        result = make_result(
            [
                PredictionMatch(0.95, True, "1"),
                PredictionMatch(0.90, True, "2"),
            ],
            ground_truth_count=3,
        )

        metrics = calculate_average_precision([result])

        assert metrics.average_precision == pytest.approx(2 / 3)
        assert metrics.recall == pytest.approx(2 / 3)
        assert metrics.false_negative_count == 1

    def test_no_ground_truth_raises_error(self):
        result = make_result([], ground_truth_count=0)

        with pytest.raises(ValueError, match="ground-truth"):
            calculate_average_precision([result])

    def test_invalid_confidence_is_rejected(self):
        result = make_result(
            [
                PredictionMatch(1.5, True, "1"),
            ]
        )

        with pytest.raises(ValueError, match="confidence"):
            calculate_average_precision([result])

    def test_empty_frame_results_raises_error(self):
        with pytest.raises(ValueError, match="ground-truth"):
            calculate_average_precision([])
