
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from lap import lapjv

from evaluation.tracking.matching import FrameTrackingMatchResult


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    ground_truth_detections: int
    prediction_detections: int
    ignored_predictions: int

    identity_switches: int
    fragmentations: int

    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int

    precision: float
    recall: float
    mota: float
    motp: float

    id_precision: float
    id_recall: float
    idf1: float


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _calculate_identity_true_positives(
    cooccurrences: Counter,
) -> int:

    if not cooccurrences:
        return 0

    gt_ids = sorted({
        target_id
        for target_id, _ in cooccurrences
    })

    prediction_ids = sorted({
        track_id
        for _, track_id in cooccurrences
    })

    size = max(len(gt_ids), len(prediction_ids))

    if size == 0:
        return 0

    weights = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    gt_indices = {
        target_id: index
        for index, target_id in enumerate(gt_ids)
    }

    prediction_indices = {
        track_id: index
        for index, track_id in enumerate(prediction_ids)
    }

    for (target_id, track_id), count in cooccurrences.items():
        weights[
            gt_indices[target_id],
            prediction_indices[track_id],
        ] = count

    maximum_weight = max(1.0, float(weights.max()))

    cost_matrix = maximum_weight - weights

    _, row_assignments, _ = lapjv(cost_matrix)

    id_true_positives = 0

    for gt_index, prediction_index in enumerate(row_assignments):
        if prediction_index < 0:
            continue

        if gt_index >= len(gt_ids):
            continue

        if prediction_index >= len(prediction_ids):
            continue

        id_true_positives += int(
            weights[gt_index, prediction_index]
        )

    return id_true_positives


def compute_tracking_metrics(
    frame_results: Sequence[FrameTrackingMatchResult],
) -> TrackingMetrics:

    ordered_results = sorted(
        frame_results,
        key=lambda result: result.frame_id,
    )

    frame_ids = [
        result.frame_id
        for result in ordered_results
    ]

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Duplicate frame IDs in tracking results.")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    ignored_predictions = 0
    total_iou = 0.0

    identity_switches = 0
    fragmentations = 0

    cooccurrences = Counter()

    previous_track_by_target = {}

    previously_matched = set()
    interrupted_targets = set()

    for result in ordered_results:

        true_positives += result.true_positive_count
        false_positives += result.false_positive_count
        false_negatives += result.false_negative_count
        ignored_predictions += len(result.ignored_predictions)
        for match in result.matches:

            target_id = match.target_id
            track_id = match.published_track_id

            total_iou += match.iou

            cooccurrences[(target_id, track_id)] += 1

            previous_track = previous_track_by_target.get(
                target_id
            )

            if (
                previous_track is not None
                and previous_track != track_id
            ):
                identity_switches += 1

            previous_track_by_target[target_id] = track_id

        matched_targets = {
            match.target_id
            for match in result.matches
        }

        visible_targets = (
            matched_targets
            | {
                target.target_id
                for target in result.unmatched_ground_truth
            }
        )

        for target_id in visible_targets:

            if target_id in matched_targets:

                if target_id in interrupted_targets:
                    fragmentations += 1
                    interrupted_targets.remove(target_id)

                previously_matched.add(target_id)

            elif target_id in previously_matched:

                interrupted_targets.add(target_id)

    ground_truth_detections = (
        true_positives + false_negatives
    )

    prediction_detections = (
        true_positives + false_positives
    )

    id_true_positives = _calculate_identity_true_positives(
        cooccurrences
    )

    id_false_positives = (
        prediction_detections - id_true_positives
    )

    id_false_negatives = (
        ground_truth_detections - id_true_positives
    )

    precision = _safe_divide(
        true_positives,
        prediction_detections,
    )

    recall = _safe_divide(
        true_positives,
        ground_truth_detections,
    )

    mota = (
        1.0 - _safe_divide(
            false_negatives
            + false_positives
            + identity_switches,
            ground_truth_detections,
        )
        if ground_truth_detections > 0
        else 0.0
    )

    motp = _safe_divide(
        total_iou,
        true_positives,
    )

    id_precision = _safe_divide(
        id_true_positives,
        id_true_positives + id_false_positives,
    )

    id_recall = _safe_divide(
        id_true_positives,
        id_true_positives + id_false_negatives,
    )

    idf1 = _safe_divide(
        2 * id_true_positives,
        2 * id_true_positives
        + id_false_positives
        + id_false_negatives,
    )

    return TrackingMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ground_truth_detections=ground_truth_detections,
        prediction_detections=prediction_detections,
        ignored_predictions=ignored_predictions,
        identity_switches=identity_switches,
        fragmentations=fragmentations,
        id_true_positives=id_true_positives,
        id_false_positives=id_false_positives,
        id_false_negatives=id_false_negatives,
        precision=precision,
        recall=recall,
        mota=mota,
        motp=motp,
        id_precision=id_precision,
        id_recall=id_recall,
        idf1=idf1,
    )