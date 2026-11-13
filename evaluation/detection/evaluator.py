
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from evaluation.detection.annotations import AnnotationSequence
from evaluation.detection.matching import match_frame_detections
from evaluation.detection.metrics import (
    AveragePrecisionResult,
    calculate_average_precision,
)


@dataclass(frozen=True)
class SequenceEvaluationResult:
    frames_processed: int
    total_detections: int
    average_detections_per_frame: float
    metrics_by_iou: dict[float, AveragePrecisionResult]
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    peak_gpu_memory_mb: float
    inference_failures: int
    discarded_non_vehicle_count: int
    clipped_box_count: int
    degenerate_box_count: int


def evaluate_sequence(
    frames: Iterable,
    detector,
    annotations: AnnotationSequence,
    iou_thresholds: tuple[float, ...] = (0.5, 0.7),
    ignore_coverage_threshold: float = 0.5,
    warmup_frames: int = 1,
) -> SequenceEvaluationResult:
    """
    Evaluate a vehicle detector against one annotated sequence.

    Matching is class-agnostic. AP is calculated independently
    for every requested ground-truth matching IoU threshold.

    The first warmup_frames are excluded from latency statistics,
    but are included in detection metrics.
    """

    if not iou_thresholds:
        raise ValueError("At least one IoU threshold is required")

    if len(set(iou_thresholds)) != len(iou_thresholds):
        raise ValueError("IoU thresholds must be unique")

    if any(not 0 < value <= 1 for value in iou_thresholds):
        raise ValueError("IoU thresholds must be in (0, 1]")

    if warmup_frames < 0:
        raise ValueError("warmup_frames cannot be negative")

    frame_results = {
        threshold: [] for threshold in iou_thresholds
    }

    frame_count = 0
    total_detections = 0
    latency_samples = []
    used_cuda = False

    initial_failures = detector.inference_failure_count
    initial_discarded = detector.discarded_non_vehicle_count
    initial_clipped = detector.clipped_box_count
    initial_degenerate = detector.degenerate_box_count

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for frame in frames:
        frame_id = frame.frame_id

        if frame_id not in annotations.frames:
            raise ValueError(
                f"No ground-truth annotations found for frame {frame_id}"
            )

        # Synchronize CUDA before timing so asynchronous GPU work
        # from a previous frame is not included in this measurement.
        if (
            torch.cuda.is_available()
            and detector.last_inference_device != "cpu"
        ):
            torch.cuda.synchronize()

        start_time = perf_counter()
        detections = detector.detect(frame)

        # Synchronize again to measure completed GPU inference.
        if (
            torch.cuda.is_available()
            and detector.last_inference_device != "cpu"
        ):
            torch.cuda.synchronize()
            used_cuda = True

        elapsed_ms = (perf_counter() - start_time) * 1000

        if frame_count >= warmup_frames:
            latency_samples.append(elapsed_ms)

        ground_truth = annotations.frames[frame_id]
        total_detections += len(detections)

        for threshold in iou_thresholds:
            match_result = match_frame_detections(
                detections=detections,
                ground_truth=ground_truth,
                ignored_regions=annotations.ignored_regions,
                iou_threshold=threshold,
                ignore_coverage_threshold=ignore_coverage_threshold,
            )

            frame_results[threshold].append(match_result)

        frame_count += 1

    if frame_count == 0:
        raise ValueError("No frames were processed")

    metrics_by_iou = {
        threshold: calculate_average_precision(results)
        for threshold, results in frame_results.items()
    }

    if latency_samples:
        latency_p50 = float(np.percentile(latency_samples, 50))
        latency_p95 = float(np.percentile(latency_samples, 95))
    else:
        latency_p50 = 0.0
        latency_p95 = 0.0

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if used_cuda
        else 0.0
    )

    return SequenceEvaluationResult(
        frames_processed=frame_count,
        total_detections=total_detections,
        average_detections_per_frame=total_detections / frame_count,
        metrics_by_iou=metrics_by_iou,
        inference_latency_p50_ms=latency_p50,
        inference_latency_p95_ms=latency_p95,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        inference_failures=(
            detector.inference_failure_count - initial_failures
        ),
        discarded_non_vehicle_count=(
            detector.discarded_non_vehicle_count - initial_discarded
        ),
        clipped_box_count=(
            detector.clipped_box_count - initial_clipped
        ),
        degenerate_box_count=(
            detector.degenerate_box_count - initial_degenerate
        ),
    )
