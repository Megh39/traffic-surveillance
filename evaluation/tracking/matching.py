
from dataclasses import dataclass

import numpy as np
from lap import lapjv

from evaluation.detection.annotations import (
    GroundTruthBox,
    IgnoredRegion,
)
from evaluation.detection.matching import (
    intersection_over_union,
    intersection_over_prediction,
)
from src.tvs.perception.tracker import TrackObservation


@dataclass(frozen=True, slots=True)
class TrackMatch:
    target_id: str
    published_track_id: int
    iou: float


@dataclass(frozen=True, slots=True)
class FrameTrackingMatchResult:
    frame_id: int
    matches: tuple[TrackMatch, ...]
    unmatched_ground_truth: tuple[GroundTruthBox, ...]
    unmatched_predictions: tuple[TrackObservation, ...]
    ignored_predictions: tuple[TrackObservation, ...]

    @property
    def true_positive_count(self) -> int:
        return len(self.matches)

    @property
    def false_positive_count(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def false_negative_count(self) -> int:
        return len(self.unmatched_ground_truth)

    @property
    def ignored_prediction_count(self) -> int:
        return len(self.ignored_predictions)


def match_frame_tracks(
    predictions: list[TrackObservation],
    ground_truth: list[GroundTruthBox],
    ignored_regions: tuple[IgnoredRegion, ...] | list[IgnoredRegion] = (),
    iou_threshold: float = 0.5,
    ignore_coverage_threshold: float = 0.5,
) -> FrameTrackingMatchResult:

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1].")

    if not 0.0 <= ignore_coverage_threshold <= 1.0:
        raise ValueError("Ignore coverage threshold must be in [0, 1].")

    if any(pred.published_track_id is None for pred in predictions):
        raise ValueError("All predictions must have a published track ID.")

    frame_ids = {
        item.frame_id for item in predictions
    } | {
        item.frame_id for item in ground_truth
    }

    if len(frame_ids) > 1:
        raise ValueError("All predictions and ground truth must belong to one frame.")

    frame_id = next(iter(frame_ids), 0)

    prediction_ids = [
        pred.published_track_id for pred in predictions
    ]

    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Duplicate published track IDs in one frame.")

    target_ids = [
        gt.target_id for gt in ground_truth
    ]

    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Duplicate ground-truth target IDs in one frame.")

    if not predictions or not ground_truth:
        matched_prediction_indices = set()
        matched_gt_indices = set()
        matches = []

    else:
        iou_matrix = np.zeros(
            (len(predictions), len(ground_truth)),
            dtype=np.float64,
        )

        for pred_index, prediction in enumerate(predictions):
            for gt_index, target in enumerate(ground_truth):
                iou_matrix[pred_index, gt_index] = (
                    intersection_over_union(
                        prediction.bbox,
                        target.bbox,
                    )
                )

        cost_matrix = 1.0 - iou_matrix

        _, row_assignments, _ = lapjv(
            cost_matrix,
            extend_cost=True,
            cost_limit=1.0 - iou_threshold,
        )

        matches = []
        matched_prediction_indices = set()
        matched_gt_indices = set()

        for pred_index, gt_index in enumerate(row_assignments):
            if gt_index < 0:
                continue

            iou = float(iou_matrix[pred_index, gt_index])

            if iou < iou_threshold:
                continue

            matches.append(
                TrackMatch(
                    target_id=ground_truth[gt_index].target_id,
                    published_track_id=predictions[
                        pred_index
                    ].published_track_id,
                    iou=iou,
                )
            )

            matched_prediction_indices.add(pred_index)
            matched_gt_indices.add(gt_index)

    unmatched_predictions = []
    ignored_predictions = []

    for index, prediction in enumerate(predictions):
        if index in matched_prediction_indices:
            continue

        covered_by_ignored_region = any(
            intersection_over_prediction(
                prediction.bbox,
                region.bbox,
            ) >= ignore_coverage_threshold
            for region in ignored_regions
        )

        if covered_by_ignored_region:
            ignored_predictions.append(prediction)
        else:
            unmatched_predictions.append(prediction)

    unmatched_ground_truth = [
        target
        for index, target in enumerate(ground_truth)
        if index not in matched_gt_indices
    ]

    return FrameTrackingMatchResult(
        frame_id=frame_id,
        matches=tuple(matches),
        unmatched_ground_truth=tuple(unmatched_ground_truth),
        unmatched_predictions=tuple(unmatched_predictions),
        ignored_predictions=tuple(ignored_predictions),
    )
