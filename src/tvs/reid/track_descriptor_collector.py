from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.tvs.perception.boxes import tlwh_to_xyxy


class EmbedderProtocol(Protocol):
    def from_frame_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray | None: ...


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    minimum_confidence: float = 0.50
    minimum_crop_area: int = 32 * 32
    maximum_samples_per_track: int = 20
    minimum_samples_per_track: int = 3

    def __post_init__(self):
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.minimum_crop_area < 1 or self.maximum_samples_per_track < 1:
            raise ValueError("area and maximum sample count must be positive")
        if not 1 <= self.minimum_samples_per_track <= self.maximum_samples_per_track:
            raise ValueError("minimum_samples_per_track must be between 1 and maximum_samples_per_track")


@dataclass(slots=True)
class _TrackSamples:
    descriptors: list[np.ndarray] = field(default_factory=list)
    seen_frame_ids: set[int] = field(default_factory=set)


class TrackDescriptorCollector:
    """Collect multiple frame descriptors and aggregate one descriptor per published track."""

    def __init__(self, embedder: EmbedderProtocol, config: CollectorConfig | None = None):
        self._embedder = embedder
        self.config = config or CollectorConfig()
        self._tracks: dict[int, _TrackSamples] = {}

    def add(self, frame, observation) -> bool:
        track_id = observation.published_track_id
        if track_id is None or observation.confidence < self.config.minimum_confidence:
            return False

        samples = self._tracks.setdefault(int(track_id), _TrackSamples())
        if observation.frame_id in samples.seen_frame_ids:
            return False
        if len(samples.descriptors) >= self.config.maximum_samples_per_track:
            return False

        x, y, width, height = observation.bbox
        if width <= 0 or height <= 0 or width * height < self.config.minimum_crop_area:
            return False

        bbox_xyxy = tlwh_to_xyxy(observation.bbox)
        descriptor = self._embedder.from_frame_bbox(frame.image, bbox_xyxy)
        if descriptor is None:
            return False

        vector = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        if vector.size != 512:
            raise ValueError(f"Expected a 512-dimensional descriptor, got {vector.size}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Descriptor contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Descriptor has zero norm")

        samples.descriptors.append(vector / norm)
        samples.seen_frame_ids.add(int(observation.frame_id))
        return True

    def finalize(self, published_track_id: int) -> np.ndarray | None:
        samples = self._tracks.pop(int(published_track_id), None)
        if samples is None or len(samples.descriptors) < self.config.minimum_samples_per_track:
            return None
        mean = np.mean(np.stack(samples.descriptors), axis=0)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return np.asarray(mean / norm, dtype=np.float32)

    def finalize_all(self) -> dict[int, np.ndarray]:
        result = {}
        for track_id in list(self._tracks):
            descriptor = self.finalize(track_id)
            if descriptor is not None:
                result[track_id] = descriptor
        return result

    def discard(self, published_track_id: int) -> None:
        self._tracks.pop(int(published_track_id), None)

    def sample_count(self, published_track_id: int) -> int:
        samples = self._tracks.get(int(published_track_id))
        return 0 if samples is None else len(samples.descriptors)
