from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruthBox:
    frame_id: int
    target_id: str
    bbox: BBox
    vehicle_type: str
    truncation_ratio: float | None
    occluded: bool


@dataclass(frozen=True)
class IgnoredRegion:
    bbox: BBox


@dataclass(frozen=True)
class AnnotationSequence:
    frames: dict[int, tuple[GroundTruthBox, ...]]
    ignored_regions: tuple[IgnoredRegion, ...]
    total_annotations: int
    excluded_truncated_annotations: int
    invalid_truncation_annotations: int = 0

    @property
    def retained_annotations(self) -> int:
        return sum(len(annotations) for annotations in self.frames.values())


def _parse_bbox(element: ET.Element, context: str) -> BBox:
    try:
        left = float(element.attrib["left"])
        top = float(element.attrib["top"])
        width = float(element.attrib["width"])
        height = float(element.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid bounding box: {context}") from exc

    values = (left, top, width, height)

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite bounding box: {context}")

    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive bounding box dimensions: {context}")

    return values


def load_detrac_annotations(
    xml_path: str | Path,
    max_truncation_ratio: float = 0.5,
) -> AnnotationSequence:
    """
    Load UA-DETRAC annotations from an XML file.

    Annotations with truncation_ratio >= max_truncation_ratio
    are excluded.

    Bounding boxes use TLWH format:
    (left, top, width, height).

    Ignored regions are sequence-level regions.
    """

    if not 0 <= max_truncation_ratio <= 1:
        raise ValueError(
            "max_truncation_ratio must be between 0 and 1"
        )

    xml_path = Path(xml_path)

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_path}"
        )

    root = ET.parse(xml_path).getroot()

    frames: dict[int, tuple[GroundTruthBox, ...]] = {}
    total_annotations = 0
    excluded_annotations = 0
    invalid_truncation_annotations = 0

    for frame_element in root.findall("./frame"):
        try:
            frame_id = int(frame_element.attrib["num"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid or missing frame number") from exc

        if frame_id < 1:
            raise ValueError(f"Invalid frame ID: {frame_id}")

        if frame_id in frames:
            raise ValueError(f"Duplicate frame ID: {frame_id}")

        frame_annotations = []

        for target in frame_element.findall("./target_list/target"):
            total_annotations += 1

            target_id = target.get("id")

            if target_id is None:
                raise ValueError(
                    f"Missing target ID in frame {frame_id}"
                )

            box_element = target.find("box")
            attribute = target.find("attribute")

            if box_element is None or attribute is None:
                raise ValueError(
                    f"Missing box or attribute for target "
                    f"{target_id} in frame {frame_id}"
                )

            bbox = _parse_bbox(
                box_element,
                f"frame={frame_id}, target={target_id}",
            )

            vehicle_type = attribute.get("vehicle_type")

            if not vehicle_type:
                raise ValueError(
                    f"Missing vehicle_type for target {target_id}"
                )

            try:
                truncation_ratio = float(
                    attribute.attrib["truncation_ratio"]
                )
            except (KeyError, ValueError):
                truncation_ratio = None

            if truncation_ratio is not None and (
                not math.isfinite(truncation_ratio)
                or not 0 <= truncation_ratio <= 1
            ):
                truncation_ratio = None

            if truncation_ratio is None:
                invalid_truncation_annotations += 1
            elif truncation_ratio >= max_truncation_ratio:
                excluded_annotations += 1
                continue

            frame_annotations.append(
                GroundTruthBox(
                    frame_id=frame_id,
                    target_id=target_id,
                    bbox=bbox,
                    vehicle_type=vehicle_type,
                    truncation_ratio=truncation_ratio,
                    occluded=target.find("occlusion") is not None,
                )
            )

        frames[frame_id] = tuple(frame_annotations)

    ignored_regions = []

    ignored_element = root.find("./ignored_region")

    if ignored_element is not None:
        for index, box_element in enumerate(
            ignored_element.findall("./box")
        ):
            ignored_regions.append(
                IgnoredRegion(
                    bbox=_parse_bbox(
                        box_element,
                        f"ignored_region={index}",
                    )
                )
            )

    return AnnotationSequence(
        frames=frames,
        ignored_regions=tuple(ignored_regions),
        total_annotations=total_annotations,
        excluded_truncated_annotations=excluded_annotations,
        invalid_truncation_annotations=invalid_truncation_annotations,
    )
