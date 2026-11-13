from dataclasses import dataclass
from math import isfinite
from typing import Literal


VehicleType = Literal[
    "car",
    "van",
    "bus",
    "truck",
    "two_wheeler",
    "three_wheeler",
    "other",
]


# Verified against the loaded YOLO11n COCO class mapping.
# Each entry contains: class ID -> (original model label, internal category).
COCO_VEHICLE_CLASS_MAP: dict[int, tuple[str, VehicleType]] = {
    2: ("car", "car"),
    3: ("motorcycle", "two_wheeler"),
    5: ("bus", "bus"),
    7: ("truck", "truck"),
}


def map_coco_vehicle_class(
    class_id: int,
    source_label: str,
) -> VehicleType | None:
    """
    Map a COCO model class to the internal vehicle taxonomy.

    Returns None for classes that are not supported road-vehicle
    detections. Raises ValueError if the supplied label does not
    match the verified COCO class ID.
    """
    mapping = COCO_VEHICLE_CLASS_MAP.get(class_id)

    if mapping is None:
        return None

    expected_label, vehicle_type = mapping

    if source_label.casefold() != expected_label:
        raise ValueError(
            f"Class ID {class_id} expects label '{expected_label}', "
            f"received '{source_label}'"
        )

    return vehicle_type


@dataclass(frozen=True, slots=True)
class Detection:
    """
    A vehicle detection produced for one source frame.

    bbox uses pixel coordinates in TLWH format:
    (top-left x, top-left y, width, height).
    """

    frame_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    source_label: str
    vehicle_type: VehicleType
    confidence: float

    def __post_init__(self) -> None:
        if self.frame_id < 1:
            raise ValueError("frame_id must be at least 1")

        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")

        if not self.source_label.strip():
            raise ValueError("source_label cannot be empty")

        if self.vehicle_type not in {
            "car",
            "van",
            "bus",
            "truck",
            "two_wheeler",
            "three_wheeler",
            "other",
        }:
            raise ValueError(
                f"Unsupported vehicle_type: {self.vehicle_type}"
            )

        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four values")

        if not all(isfinite(value) for value in self.bbox):
            raise ValueError("bbox values must be finite numbers")

        x, y, width, height = self.bbox

        if x < 0 or y < 0:
            raise ValueError("bbox coordinates cannot be negative")

        if width <= 0 or height <= 0:
            raise ValueError("bbox width and height must be positive")

        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
