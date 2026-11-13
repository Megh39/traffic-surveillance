from pathlib import Path
from typing import Iterator

import cv2

from tvs.events.frame import Frame
from tvs.ingestion.models import IngestionStats


class ImageSequenceSource:
    def __init__(
        self,
        source_dir: str | Path,
        fps: float,
        max_skip_ratio: float = 0.05,
        stride: int = 1,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.fps = fps
        self.max_skip_ratio = max_skip_ratio
        self.stride = stride
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.stats = IngestionStats()

        if self.fps <= 0:
            raise ValueError("fps must be greater than zero")

        if not 0 <= self.max_skip_ratio <= 1:
            raise ValueError("max_skip_ratio must be between 0 and 1")

        if self.stride < 1:
            raise ValueError("stride must be at least 1")

        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"Image sequence directory not found: {self.source_dir}"
            )

    def _image_paths(self) -> list[Path]:
        paths = list(self.source_dir.glob("*.jpg"))

        def frame_id_from_path(path: Path) -> int:
            stem = path.stem
            if not stem.startswith("img"):
                raise ValueError(f"Unexpected image filename: {path.name}")

            try:
                return int(stem[3:])
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse frame_id from filename: {path.name}"
                ) from exc

        return sorted(paths, key=frame_id_from_path)

    def frames(self) -> Iterator[Frame]:
        image_paths = self._image_paths()

        self.stats = IngestionStats()
        self.stats.frames_discovered = len(image_paths)

        previous_frame_id: int | None = None

        for path_index, image_path in enumerate(image_paths):
            stem = image_path.stem

            if not stem.startswith("img"):
                raise ValueError(f"Unexpected image filename: {image_path.name}")

            frame_id = int(stem[3:])

            if previous_frame_id is not None and frame_id <= previous_frame_id:
                raise ValueError(
                    "Frame IDs must be strictly monotonic: "
                    f"{previous_frame_id} followed by {frame_id}"
                )

            previous_frame_id = frame_id

            # Sampling uses the original frame index.
            if path_index % self.stride != 0:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                self.stats.frames_skipped += 1

                if self.stats.skip_ratio > self.max_skip_ratio:
                    raise RuntimeError(
                        "Unreadable-frame ratio exceeded the configured "
                        f"maximum of {self.max_skip_ratio:.2%}"
                    )

                continue

            height, width = image.shape[:2]

            if self.expected_width is not None and self.expected_height is not None:
                if width != self.expected_width or height != self.expected_height:
                    print(
                        "WARNING: frame resolution mismatch: "
                        f"expected {self.expected_width}x{self.expected_height}, "
                        f"got {width}x{height} "
                        f"(frame_id={frame_id})"
                    )

            self.stats.frames_processed += 1

            elapsed_seconds = (frame_id - 1) / self.fps

            yield Frame(
                frame_id=frame_id,
                elapsed_seconds=elapsed_seconds,
                image=image,
                source_path=image_path,
                fps_effective=self.fps,
            )
