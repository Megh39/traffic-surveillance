from pathlib import Path
from unittest.mock import Mock
import numpy as np

from src.tvs.events.frame import Frame
from src.tvs.events.detection import Detection
import pytest

from src.tvs.perception.detector import (
    DetectorConfig,
    VehicleDetector,
)


def make_config(weights_path, device="auto", half=True):
    return DetectorConfig(
        weights_path=Path(weights_path),
        device=device,
        half=half,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 5, 7),
    )


def make_frame():
    return Frame(
        frame_id=1,
        elapsed_seconds=0.0,
        image=np.zeros((540, 960, 3), dtype=np.uint8),
        source_path=Path("frame_000001.jpg"),
        fps_effective=30.0,
    )


def make_box(class_id, confidence, xyxy):
    box = Mock()
    box.cls = [Mock(item=Mock(return_value=class_id))]
    box.conf = [Mock(item=Mock(return_value=confidence))]
    box.xyxy = [Mock(tolist=Mock(return_value=list(xyxy)))]
    return box


def make_detector(tmp_path, monkeypatch, vehicle_class_ids=(2, 3, 5, 7)):
    weights = tmp_path / "yolo11n.pt"
    weights.touch()

    model = Mock()
    model.names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    monkeypatch.setattr(
        "tvs.perception.detector.YOLO",
        Mock(return_value=model),
    )
    monkeypatch.setattr(
        "tvs.perception.detector.torch.cuda.is_available",
        lambda: False,
    )

    config = DetectorConfig(
        weights_path=weights,
        device="cpu",
        half=False,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=vehicle_class_ids,
    )

    detector = VehicleDetector(config)
    return detector, model

def make_cuda_detector(tmp_path, monkeypatch):
    weights = tmp_path / "yolo11n.pt"
    weights.touch()

    model = Mock()
    model.names = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    monkeypatch.setattr(
        "tvs.perception.detector.YOLO",
        Mock(return_value=model),
    )
    monkeypatch.setattr(
        "tvs.perception.detector.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "tvs.perception.detector.torch.cuda.empty_cache",
        lambda: None,
    )

    detector = VehicleDetector(
        make_config(weights, device="cuda", half=True)
    )

    return detector, model
class TestDeviceSelection:

    def test_auto_selects_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        assert VehicleDetector._resolve_device("auto") == "0"

    def test_auto_selects_cpu_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: False,
        )

        assert VehicleDetector._resolve_device("auto") == "cpu"

    def test_explicit_cpu_selection(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        assert VehicleDetector._resolve_device("cpu") == "cpu"

    def test_explicit_cuda_selection(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        assert VehicleDetector._resolve_device("cuda") == "0"

    def test_explicit_gpu_zero_selection(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        assert VehicleDetector._resolve_device("0") == "0"

    def test_cuda_request_fails_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="CUDA was requested"):
            VehicleDetector._resolve_device("cuda")

    def test_gpu_zero_request_fails_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="CUDA was requested"):
            VehicleDetector._resolve_device("0")

    def test_unsupported_device_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported"):
            VehicleDetector._resolve_device("gpu")


class TestModelLoading:

    def test_missing_weights_are_rejected(self, tmp_path, monkeypatch):
        mock_yolo = Mock()

        monkeypatch.setattr(
            "tvs.perception.detector.YOLO",
            mock_yolo,
        )

        config = make_config(tmp_path / "missing.pt")

        with pytest.raises(FileNotFoundError, match="weights not found"):
            VehicleDetector(config)

        mock_yolo.assert_not_called()

    def test_model_is_initialized_with_correct_weights(self, tmp_path, monkeypatch):
        weights = tmp_path / "yolo11n.pt"
        weights.touch()

        mock_model = Mock()
        mock_yolo = Mock(return_value=mock_model)

        monkeypatch.setattr(
            "tvs.perception.detector.YOLO",
            mock_yolo,
        )
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: False,
        )

        detector = VehicleDetector(make_config(weights, device="cpu"))

        mock_yolo.assert_called_once_with(str(weights.resolve()))
        assert detector.model is mock_model
        assert detector.device == "cpu"

    def test_fp16_is_disabled_on_cpu(self, tmp_path, monkeypatch):
        weights = tmp_path / "yolo11n.pt"
        weights.touch()

        monkeypatch.setattr(
            "tvs.perception.detector.YOLO",
            Mock(),
        )
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: False,
        )

        detector = VehicleDetector(make_config(weights, device="cpu", half=True))

        assert detector.use_half is False

    def test_fp16_is_enabled_on_cuda(self, tmp_path, monkeypatch):
        weights = tmp_path / "yolo11n.pt"
        weights.touch()

        monkeypatch.setattr(
            "tvs.perception.detector.YOLO",
            Mock(),
        )
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        detector = VehicleDetector(make_config(weights, device="cuda", half=True))

        assert detector.device == "0"
        assert detector.use_half is True

    def test_fp16_remains_disabled_when_not_configured(self, tmp_path, monkeypatch):
        weights = tmp_path / "yolo11n.pt"
        weights.touch()

        monkeypatch.setattr(
            "tvs.perception.detector.YOLO",
            Mock(),
        )
        monkeypatch.setattr(
            "tvs.perception.detector.torch.cuda.is_available",
            lambda: True,
        )

        detector = VehicleDetector(make_config(weights, device="cuda", half=False))

        assert detector.use_half is False

class TestVehicleDetection:

    def test_vehicle_detection_returns_detection_objects(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [100, 120, 200, 220]),
                    make_box(5, 0.75, [300, 150, 450, 300]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert len(detections) == 2
        assert all(isinstance(d, Detection) for d in detections)
        assert detections[0].vehicle_type == "car"
        assert detections[1].vehicle_type == "bus"

    def test_non_vehicle_classes_are_discarded(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(0, 0.90, [100, 100, 200, 200]),
                    make_box(2, 0.85, [300, 100, 400, 200]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert len(detections) == 1
        assert detections[0].vehicle_type == "car"
        assert detector.discarded_non_vehicle_count == 1

    def test_configured_vehicle_classes_are_respected(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(
            tmp_path,
            monkeypatch,
            vehicle_class_ids=(2,),
        )

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [100, 100, 200, 200]),
                    make_box(3, 0.75, [300, 100, 400, 200]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert len(detections) == 1
        assert detections[0].vehicle_type == "car"
        assert detector.discarded_vehicle_class_count == 1

    def test_bounding_box_is_converted_to_tlwh(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [100, 120, 250, 300]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert len(detections) == 1
        assert detections[0].bbox == (100.0, 120.0, 150.0, 180.0)

    def test_bounding_box_is_clipped_to_frame_dimensions(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [-10, -5, 1000, 550]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert len(detections) == 1
        assert detections[0].bbox == (0.0, 0.0, 960.0, 540.0)
        assert detector.clipped_box_count == 1

    def test_degenerate_box_after_clipping_is_discarded(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [1000, 100, 1100, 200]),
                ]
            )
        ]

        detections = detector.detect(make_frame())

        assert detections == []
        assert detector.clipped_box_count == 1
        assert detector.degenerate_box_count == 1

    def test_empty_frame_returns_empty_detection_list(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(boxes=[])
        ]

        detections = detector.detect(make_frame())

        assert detections == []

    def test_none_boxes_returns_empty_detection_list(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(boxes=None)
        ]

        detections = detector.detect(make_frame())

        assert detections == []

    def test_frame_id_is_preserved(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.return_value = [
            Mock(
                boxes=[
                    make_box(2, 0.85, [100, 100, 200, 200]),
                ]
            )
        ]

        frame = make_frame()
        frame = Frame(
            frame_id=42,
            elapsed_seconds=1.366,
            image=frame.image,
            source_path=frame.source_path,
            fps_effective=30.0,
        )

        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].frame_id == 42

    def test_inference_retries_once_after_failure(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.side_effect = [
            RuntimeError("Temporary inference failure"),
            [Mock(boxes=[])],
        ]

        detections = detector.detect(make_frame())

        assert detections == []
        assert model.predict.call_count == 2
        assert detector.inference_failure_count == 0

    def test_inference_failure_returns_empty_after_retry(
        self, tmp_path, monkeypatch
    ):
        detector, model = make_detector(tmp_path, monkeypatch)

        model.predict.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
        ]

        detections = detector.detect(make_frame())

        assert detections == []
        assert model.predict.call_count == 2
        assert detector.inference_failure_count == 1


class TestPublishableDetections:

    def test_publish_threshold_filters_without_mutating_original(
        self, tmp_path, monkeypatch
    ):
        detector, _ = make_detector(tmp_path, monkeypatch)

        detections = [
            Detection(
                frame_id=1,
                bbox=(10.0, 20.0, 50.0, 40.0),
                class_id=2,
                source_label="car",
                vehicle_type="car",
                confidence=0.15,
            ),
            Detection(
                frame_id=1,
                bbox=(100.0, 120.0, 60.0, 40.0),
                class_id=5,
                source_label="bus",
                vehicle_type="bus",
                confidence=0.45,
            ),
            Detection(
                frame_id=1,
                bbox=(200.0, 220.0, 70.0, 50.0),
                class_id=7,
                source_label="truck",
                vehicle_type="truck",
                confidence=0.30,
            ),
        ]

        original_count = len(detections)

        publishable = detector.get_publishable_detections(
            detections
        )

        assert len(detections) == original_count
        assert len(publishable) == 2
        assert all(
            d.confidence >= 0.30
            for d in publishable
        )
        assert detections[0].confidence == 0.15

class TestCudaFallback:

    def test_cuda_oom_retries_with_reduced_resolution(
        self, tmp_path, monkeypatch
    ):
        detector, _ = make_cuda_detector(tmp_path, monkeypatch)

        empty_results = [Mock(boxes=[])]

        monkeypatch.setattr(
            detector,
            "_run_prediction",
            Mock(
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    empty_results,
                ]
            ),
        )

        detections = detector.detect(make_frame())

        assert detections == []
        assert detector._run_prediction.call_count == 2

        calls = detector._run_prediction.call_args_list

        assert [
            (call.kwargs["device"], call.kwargs["image_size"])
            for call in calls
        ] == [
            ("0", 640),
            ("0", 320),
        ]

        assert detector.last_inference_device == "0"
        assert detector.effective_image_size == 320

    def test_reduced_cuda_failure_falls_back_to_cpu(
        self, tmp_path, monkeypatch
    ):
        detector, _ = make_cuda_detector(tmp_path, monkeypatch)

        empty_results = [Mock(boxes=[])]

        monkeypatch.setattr(
            detector,
            "_run_prediction",
            Mock(
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    RuntimeError("CUDA out of memory"),
                    empty_results,
                ]
            ),
        )

        detections = detector.detect(make_frame())

        assert detections == []
        assert detector._run_prediction.call_count == 3

        calls = detector._run_prediction.call_args_list

        assert [
            (call.kwargs["device"], call.kwargs["image_size"])
            for call in calls
        ] == [
            ("0", 640),
            ("0", 320),
            ("cpu", 320),
        ]

        assert detector.last_inference_device == "cpu"
        assert detector.last_inference_image_size == 320

    def test_cpu_failure_is_retried_and_recorded(
        self, tmp_path, monkeypatch
    ):
        detector, _ = make_cuda_detector(tmp_path, monkeypatch)

        monkeypatch.setattr(
            detector,
            "_run_prediction",
            Mock(
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    RuntimeError("CUDA out of memory"),
                    RuntimeError("CPU inference failed"),
                    RuntimeError("CPU inference failed again"),
                ]
            ),
        )

        detections = detector.detect(make_frame())

        assert detections == []
        assert detector._run_prediction.call_count == 4
        assert detector.inference_failure_count == 1

    def test_successful_fallback_is_reused_for_next_frame(
        self, tmp_path, monkeypatch
    ):
        detector, _ = make_cuda_detector(tmp_path, monkeypatch)

        empty_results = [Mock(boxes=[])]

        monkeypatch.setattr(
            detector,
            "_run_prediction",
            Mock(
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    empty_results,
                    empty_results,
                ]
            ),
        )

        detector.detect(make_frame())
        detector.detect(make_frame())

        calls = detector._run_prediction.call_args_list

        assert [
            (call.kwargs["device"], call.kwargs["image_size"])
            for call in calls
        ] == [
            ("0", 640),
            ("0", 320),
            ("0", 320),
        ]

        assert detector.effective_image_size == 320
        assert detector.inference_failure_count == 0
