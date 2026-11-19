from __future__ import annotations

import numpy as np

from .crop_extractor import extract_vehicle_crop
from .model import VehicleReIDModel


class VehicleEmbedder:
    """Extract normalized appearance descriptors from tracked vehicle boxes."""

    def __init__(self, model: VehicleReIDModel):
        self.model = model

    def from_crop(self, vehicle_crop_bgr: np.ndarray) -> np.ndarray:
        return self.model.extract(vehicle_crop_bgr)

    def from_frame_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        crop = extract_vehicle_crop(frame_bgr, bbox_xyxy)
        if crop is None:
            return None
        return self.from_crop(crop)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity for two finite, non-zero feature vectors."""
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Embedding dimensions differ: {a.shape} vs {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Embeddings must contain only finite values")

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        raise ValueError("Cannot compare a zero-norm embedding")

    return float(np.dot(a, b) / denominator)
