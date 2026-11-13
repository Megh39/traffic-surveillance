
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from evaluation.detection.matching import FrameMatchResult


@dataclass(frozen=True)
class AveragePrecisionResult:
    average_precision: float
    precision: float
    recall: float
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    ground_truth_count: int
    prediction_count: int
    ignored_prediction_count: int
    precision_curve: tuple[float, ...]
    recall_curve: tuple[float, ...]


def calculate_average_precision(
    frame_results: Sequence[FrameMatchResult],
) -> AveragePrecisionResult:
    """
    Calculate all-point interpolated Average Precision across frames.

    Predictions are globally sorted by descending confidence.
    Ignored predictions are excluded from the precision-recall curve.

    Matching must already have been performed at the desired IoU
    threshold before calling this function.

    Raises:
        ValueError: If there are no ground-truth objects or if
                    matching results contain invalid values.
    """

    ground_truth_count = 0
    predictions = []
    ignored_prediction_count = 0

    for frame_result in frame_results:
        if frame_result.ground_truth_count < 0:
            raise ValueError("Ground-truth count cannot be negative")

        ground_truth_count += frame_result.ground_truth_count

        for prediction in frame_result.predictions:
            if (
                not isfinite(prediction.confidence)
                or not 0.0 <= prediction.confidence <= 1.0
            ):
                raise ValueError("Prediction confidence must be in [0, 1]")

            if prediction.is_true_positive is None:
                ignored_prediction_count += 1
            else:
                predictions.append(prediction)

    if ground_truth_count == 0:
        raise ValueError(
            "Average Precision is undefined without ground-truth objects"
        )

    # Python's sort is stable, preserving input order for equal scores.
    predictions.sort(
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )

    true_positives = 0
    false_positives = 0

    precision_curve = []
    recall_curve = []

    for prediction in predictions:
        if prediction.is_true_positive:
            true_positives += 1
        else:
            false_positives += 1

        precision_curve.append(
            true_positives / (true_positives + false_positives)
        )
        recall_curve.append(
            true_positives / ground_truth_count
        )

    # No predictions means AP, precision and recall are all zero.
    if not predictions:
        return AveragePrecisionResult(
            average_precision=0.0,
            precision=0.0,
            recall=0.0,
            true_positive_count=0,
            false_positive_count=0,
            false_negative_count=ground_truth_count,
            ground_truth_count=ground_truth_count,
            prediction_count=0,
            ignored_prediction_count=ignored_prediction_count,
            precision_curve=(),
            recall_curve=(),
        )

    # Add sentinel values and calculate the monotonically
    # non-increasing precision envelope.
    mrec = [0.0, *recall_curve, 1.0]
    mpre = [0.0, *precision_curve, 0.0]

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    # Integrate precision over recall changes.
    average_precision = 0.0

    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            average_precision += (
                (mrec[index] - mrec[index - 1]) * mpre[index]
            )

    total_predictions = true_positives + false_positives

    final_precision = (
        true_positives / total_predictions
        if total_predictions
        else 0.0
    )

    final_recall = true_positives / ground_truth_count

    return AveragePrecisionResult(
        average_precision=average_precision,
        precision=final_precision,
        recall=final_recall,
        true_positive_count=true_positives,
        false_positive_count=false_positives,
        false_negative_count=ground_truth_count - true_positives,
        ground_truth_count=ground_truth_count,
        prediction_count=total_predictions,
        ignored_prediction_count=ignored_prediction_count,
        precision_curve=tuple(precision_curve),
        recall_curve=tuple(recall_curve),
    )
