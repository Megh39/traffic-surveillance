from __future__ import annotations

import numpy as np


def extract_vehicle_crop(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    minimum_width: int = 8,
    minimum_height: int = 8,
) -> np.ndarray | None:
    """Clip an XYXY box to the frame and return its BGR crop.

    Returns None for an invalid, empty, or too-small crop.
    The function does not resize or change colour channels.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a three-channel BGR image")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))

    if right - left < minimum_width or bottom - top < minimum_height:
        return None

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)
