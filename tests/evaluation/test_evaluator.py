
from dataclasses import dataclass

import pytest

from tvs.events.detection import Detection
from evaluation.detection.annotations import (
    AnnotationSequence,
    GroundTruthBox,
)
from evaluation.detection.evaluator import evaluate_sequence


@dataclass
class FakeFrame:
    frame_id: int


class FakeDetector:
    def __init__(self, detections_by_frame):
        self.detections_by_frame = detections_by_frame
        self.inference_failure_count = 0
        self.discarded_non_vehicle_count = 0
        self.clipped_box_count = 0
        self.degenerate_box_count = 0
        self.last_inference_device = "cpu"

    def detect(self, frame):
        return self.detections_by_frame.get(frame.frame_id, [])


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


def make_annotations(frame_data):
    return AnnotationSequence(
        frames={
            frame_id: tuple(annotations)
            for frame_id, annotations in frame_data.items()
        },
        ignored_regions=(),
        total_annotations=sum(
            len(annotations) for annotations in frame_data.values()
        ),
        excluded_truncated_annotations=0,
    )


class TestSequenceEvaluator:

    def test_processes_all_frames(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        frames = [FakeFrame(1), FakeFrame(2)]

        annotations = make_annotations({
            1: [make_ground_truth((10, 10, 20, 20))],
            2: [make_ground_truth((30, 30, 20, 20), frame_id=2)],
        })

        detector = FakeDetector({
            1: [make_detection((10, 10, 20, 20))],
            2: [make_detection((30, 30, 20, 20), frame_id=2)],
        })

        result = evaluate_sequence(
            frames,
            detector,
            annotations,
            warmup_frames=0,
        )

        assert result.frames_processed == 2
        assert result.total_detections == 2
        assert result.average_detections_per_frame == 1.0

    def test_evaluates_multiple_iou_thresholds(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        frames = [FakeFrame(1)]

        annotations = make_annotations({
            1: [make_ground_truth((0, 0, 10, 10))],
        })

        # This prediction has IoU 0.6 with the ground truth.
        detector = FakeDetector({
            1: [make_detection((2.5, 0, 10, 10))],
        })

        result = evaluate_sequence(
            frames,
            detector,
            annotations,
            iou_thresholds=(0.5, 0.7),
            warmup_frames=0,
        )

        assert result.metrics_by_iou[0.5].true_positive_count == 1
        assert result.metrics_by_iou[0.7].false_positive_count == 1
        assert result.metrics_by_iou[0.7].false_negative_count == 1

    def test_latency_statistics_are_non_negative(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        frames = [FakeFrame(1), FakeFrame(2)]

        annotations = make_annotations({
            1: [make_ground_truth((0, 0, 10, 10))],
            2: [make_ground_truth((20, 20, 10, 10), frame_id=2)],
        })

        detector = FakeDetector({
            1: [make_detection((0, 0, 10, 10))],
            2: [make_detection((20, 20, 10, 10), frame_id=2)],
        })

        result = evaluate_sequence(
            frames,
            detector,
            annotations,
            warmup_frames=1,
        )

        assert result.inference_latency_p50_ms >= 0
        assert result.inference_latency_p95_ms >= 0

    def test_missing_annotation_frame_raises_error(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        detector = FakeDetector({})

        with pytest.raises(ValueError, match="No ground-truth"):
            evaluate_sequence(
                [FakeFrame(2)],
                detector,
                make_annotations({
                    1: [make_ground_truth((0, 0, 10, 10))],
                }),
            )

    def test_empty_frame_sequence_raises_error(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        detector = FakeDetector({})

        with pytest.raises(ValueError, match="No frames"):
            evaluate_sequence(
                [],
                detector,
                make_annotations({
                    1: [make_ground_truth((0, 0, 10, 10))],
                }),
            )

    def test_invalid_iou_threshold_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        with pytest.raises(ValueError, match="IoU"):
            evaluate_sequence(
                [],
                FakeDetector({}),
                make_annotations({}),
                iou_thresholds=(0.0,),
            )

    def test_duplicate_iou_thresholds_are_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.detection.evaluator.torch.cuda.is_available",
            lambda: False,
        )

        with pytest.raises(ValueError, match="unique"):
            evaluate_sequence(
                [],
                FakeDetector({}),
                make_annotations({}),
                iou_thresholds=(0.5, 0.5),
            )
