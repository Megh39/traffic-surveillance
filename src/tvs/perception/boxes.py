from math import isfinite
from typing import Sequence


Box = tuple[float, float, float, float]


def _validate_box(box: Sequence[float]) -> None:
    if len(box) != 4:
        raise ValueError("A bounding box must contain exactly four values")

    if not all(isfinite(value) for value in box):
        raise ValueError("Bounding box values must be finite numbers")


def xyxy_to_tlwh(box: Sequence[float]) -> Box:
    """
    Convert XYXY coordinates to TLWH coordinates.

    XYXY: (x1, y1, x2, y2)
    TLWH: (top-left x, top-left y, width, height)

    Coordinates are in image pixels. This function does not
    clip coordinates to image boundaries.
    """
    _validate_box(box)

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError("XYXY box must have positive width and height")

    return (float(x1), float(y1), float(width), float(height))


def tlwh_to_xyxy(box: Sequence[float]) -> Box:
    """
    Convert TLWH coordinates to XYXY coordinates.

    TLWH: (top-left x, top-left y, width, height)
    XYXY: (x1, y1, x2, y2)
    """
    _validate_box(box)

    x, y, width, height = box

    if width <= 0 or height <= 0:
        raise ValueError("TLWH box must have positive width and height")

    return (
        float(x),
        float(y),
        float(x + width),
        float(y + height),
    )
