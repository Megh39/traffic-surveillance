from pathlib import Path

import pytest
import yaml

from tvs.perception.detector import DetectorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_existing_detector_config():
    config_path = PROJECT_ROOT / "config" / "detector.yaml"

    config = DetectorConfig.from_yaml(config_path, PROJECT_ROOT)

    assert config.weights_path == (PROJECT_ROOT / "yolo11n.pt").resolve()
    assert config.device == "auto"
    assert config.half is True
    assert config.image_size == 640
    assert config.confidence_threshold == 0.10
    assert config.publish_confidence_threshold == 0.30
    assert config.iou_threshold == 0.45
    assert config.class_agnostic_nms is False
    assert config.vehicle_class_ids == (2, 3, 5, 7)


def test_relative_weights_path_is_resolved(tmp_path):
    config_data = {
        "model": {
            "weights": "models/custom.pt",
            "device": "cpu",
            "half": False,
        },
        "inference": {
            "image_size": 640,
            "confidence_threshold": 0.10,
            "publish_confidence_threshold": 0.30,
            "iou_threshold": 0.45,
            "class_agnostic_nms": False,
        },
        "classes": {
            "vehicle_class_ids": [2, 3, 5, 7],
        },
    }

    config_path = tmp_path / "detector.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = DetectorConfig.from_yaml(config_path, tmp_path)

    assert config.weights_path == (tmp_path / "models" / "custom.pt").resolve()


def test_missing_config_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        DetectorConfig.from_yaml(
            tmp_path / "missing.yaml",
            tmp_path,
        )


def test_invalid_device_is_rejected():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="gpu",
        half=True,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 5, 7),
    )

    with pytest.raises(ValueError, match="device"):
        config.validate()


def test_publish_threshold_cannot_be_lower_than_inference_threshold():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="auto",
        half=True,
        image_size=640,
        confidence_threshold=0.30,
        publish_confidence_threshold=0.10,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 5, 7),
    )

    with pytest.raises(ValueError, match="publish_confidence_threshold"):
        config.validate()


def test_invalid_iou_threshold_is_rejected():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="auto",
        half=True,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=1.5,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 5, 7),
    )

    with pytest.raises(ValueError, match="iou_threshold"):
        config.validate()


def test_empty_vehicle_class_list_is_rejected():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="auto",
        half=True,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(),
    )

    with pytest.raises(ValueError, match="vehicle_class_ids"):
        config.validate()


def test_duplicate_vehicle_class_ids_are_rejected():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="auto",
        half=True,
        image_size=640,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 3, 7),
    )

    with pytest.raises(ValueError, match="unique"):
        config.validate()


def test_non_positive_image_size_is_rejected():
    config = DetectorConfig(
        weights_path=Path("yolo11n.pt"),
        device="auto",
        half=True,
        image_size=0,
        confidence_threshold=0.10,
        publish_confidence_threshold=0.30,
        iou_threshold=0.45,
        class_agnostic_nms=False,
        vehicle_class_ids=(2, 3, 5, 7),
    )

    with pytest.raises(ValueError, match="image_size"):
        config.validate()
