import pytest

from tvs.events.detection import (
    COCO_VEHICLE_CLASS_MAP,
    Detection,
    map_coco_vehicle_class,
)


class TestVehicleClassMapping:

    @pytest.mark.parametrize(
        ("class_id", "label", "expected"),
        [
            (2, "car", "car"),
            (3, "motorcycle", "two_wheeler"),
            (5, "bus", "bus"),
            (7, "truck", "truck"),
        ],
    )
    def test_supported_vehicle_classes(
        self, class_id, label, expected
    ):
        assert map_coco_vehicle_class(class_id, label) == expected

    def test_non_vehicle_class_is_ignored(self):
        assert map_coco_vehicle_class(0, "person") is None

    def test_unknown_class_is_ignored(self):
        assert map_coco_vehicle_class(80, "unknown") is None

    def test_mismatched_class_label_raises_error(self):
        with pytest.raises(ValueError):
            map_coco_vehicle_class(2, "truck")

    def test_vehicle_mapping_contains_expected_classes(self):
        assert set(COCO_VEHICLE_CLASS_MAP) == {2, 3, 5, 7}


class TestDetection:

    def test_valid_detection(self):
        detection = Detection(
            frame_id=10,
            bbox=(100.0, 150.0, 80.0, 60.0),
            class_id=2,
            source_label="car",
            vehicle_type="car",
            confidence=0.95,
        )

        assert detection.frame_id == 10
        assert detection.bbox == (100.0, 150.0, 80.0, 60.0)
        assert detection.vehicle_type == "car"
        assert detection.confidence == 0.95

    def test_zero_width_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (100.0, 150.0, 0.0, 60.0), 2, "car", "car", 0.95)

    def test_zero_height_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (100.0, 150.0, 80.0, 0.0), 2, "car", "car", 0.95)

    def test_negative_coordinates_are_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (-1.0, 150.0, 80.0, 60.0), 2, "car", "car", 0.95)

    def test_confidence_above_one_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (100.0, 150.0, 80.0, 60.0), 2, "car", "car", 1.5)

    def test_negative_confidence_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (100.0, 150.0, 80.0, 60.0), 2, "car", "car", -0.1)

    def test_zero_frame_id_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(0, (100.0, 150.0, 80.0, 60.0), 2, "car", "car", 0.95)

    def test_empty_source_label_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (100.0, 150.0, 80.0, 60.0), 2, "", "car", 0.95)

    def test_nan_bbox_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(10, (float("nan"), 150.0, 80.0, 60.0), 2, "car", "car", 0.95)

    def test_confidence_boundaries_are_valid(self):
        for confidence in (0.0, 1.0):
            detection = Detection(
                10, (100.0, 150.0, 80.0, 60.0),
                2, "car", "car", confidence
            )
            assert detection.confidence == confidence
