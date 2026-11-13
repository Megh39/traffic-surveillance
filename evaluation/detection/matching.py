from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from tvs.events.detection import Detection
from evaluation.detection.annotations import (
    GroundTruthBox,
    IgnoredRegion,
)


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class PredictionMatch:
    confidence: float
    is_true_positive: bool | None
    matched_target_id: str | None = None


@dataclass(frozen=True)
class FrameMatchResult:
    predictions: tuple[PredictionMatch, ...]
    ground_truth_count: int

    @property
    def true_positive_count(self) -> int:
        return sum(
            prediction.is_true_positive is True
            for prediction in self.predictions
        )

    @property
    def false_positive_count(self) -> int:
        return sum(
            prediction.is_true_positive is False
            for prediction in self.predictions
        )

    @property
    def ignored_prediction_count(self) -> int:
        return sum(
            prediction.is_true_positive is None
            for prediction in self.predictions
        )


def _box_area(box: Sequence[float]) -> float:
    if len(box) != 4:
        raise ValueError("Bounding box must contain four values")

    if not all(isfinite(value) for value in box):
        raise ValueError("Bounding box values must be finite")

    _, _, width, height = box

    if width <= 0 or height <= 0:
        raise ValueError("Bounding box dimensions must be positive")

    return width * height


def intersection_over_union(box_a: BBox, box_b: BBox) -> float:
    """
    Calculate IoU between two TLWH bounding boxes.

    IoU = intersection area / union area.
    """

    area_a = _box_area(box_a)
    area_b = _box_area(box_b)

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)

    intersection = intersection_width * intersection_height
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def intersection_over_prediction(
    prediction: BBox,
    region: BBox,
) -> float:
    """
    Calculate the fraction of a prediction covered by a region.

    This is used for ignored-region handling, not ordinary
    ground-truth matching.
    """

    prediction_area = _box_area(prediction)
    region_area = _box_area(region)

    px, py, pw, ph = prediction
    rx, ry, rw, rh = region

    left = max(px, rx)
    top = max(py, ry)
    right = min(px + pw, rx + rw)
    bottom = min(py + ph, ry + rh)

    intersection = (
        max(0.0, right - left)
        * max(0.0, bottom - top)
    )

    return intersection / prediction_area


def match_frame_detections(
    detections: Sequence[Detection],
    ground_truth: Sequence[GroundTruthBox],
    ignored_regions: Sequence[IgnoredRegion] = (),
    iou_threshold: float = 0.5,
    ignore_coverage_threshold: float = 0.5,
) -> FrameMatchResult:
    """
    Match predictions against ground truth for one frame.

    Matching is class-agnostic.

    Predictions are processed by descending confidence.
    Each ground-truth object can be matched only once.

    Unmatched predictions sufficiently covered by an ignored
    region are excluded from false-positive counting.

    is_true_positive:
        True  -> matched ground truth
        False -> false positive
        None  -> ignored prediction
    """

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    if not 0 < ignore_coverage_threshold <= 1:
        raise ValueError(
            "ignore_coverage_threshold must be in (0, 1]"
        )

    frame_ids = {
        detection.frame_id for detection in detections
    } | {
        annotation.frame_id for annotation in ground_truth
    }

    if len(frame_ids) > 1:
        raise ValueError(
            "All detections and ground truth must belong to one frame"
        )

    unmatched_gt = set(range(len(ground_truth)))

    ordered_detections = sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )

    results = []

    for detection in ordered_detections:
        best_gt_index = None
        best_iou = 0.0

        for gt_index in unmatched_gt:
            overlap = intersection_over_union(
                detection.bbox,
                ground_truth[gt_index].bbox,
            )

            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            unmatched_gt.remove(best_gt_index)

            results.append(
                PredictionMatch(
                    confidence=detection.confidence,
                    is_true_positive=True,
                    matched_target_id=ground_truth[
                        best_gt_index
                    ].target_id,
                )
            )
            continue

        covered_by_ignored_region = any(
            intersection_over_prediction(
                detection.bbox,
                region.bbox,
            ) >= ignore_coverage_threshold
            for region in ignored_regions
        )

        if covered_by_ignored_region:
            results.append(
                PredictionMatch(
                    confidence=detection.confidence,
                    is_true_positive=None,
                )
            )
        else:
            results.append(
                PredictionMatch(
                    confidence=detection.confidence,
                    is_true_positive=False,
                )
            )

    return FrameMatchResult(
        predictions=tuple(results),
        ground_truth_count=len(ground_truth),
    )
