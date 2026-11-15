
from pathlib import Path
from typing import Iterator

import cv2

from src.tvs.events.frame import Frame
from src.tvs.ingestion.models import IngestionStats


class VideoFileSource:
    def __init__(
        self,
        video_path: str | Path,
        fallback_fps: float = 30.0,
        max_frames: int | None = None,
        stride: int = 1,
        expected_width: int | None = None,
        expected_height: int | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.fallback_fps = fallback_fps
        self.max_frames = max_frames
        self.stride = stride
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.stats = IngestionStats()
        self.frames_total: int | None = None
        self.fps_effective: float | None = None

        if fallback_fps <= 0:
            raise ValueError("fallback_fps must be greater than zero")

        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        if stride < 1:
            raise ValueError("stride must be at least 1")

        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        try:
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # Reject missing or implausible FPS metadata.
            if 1.0 <= reported_fps <= 120.0:
                self.fps_effective = reported_fps
            else:
                self.fps_effective = self.fallback_fps

            if reported_count > 0:
                self.frames_total = int(reported_count)
            else:
                self.frames_total = None

            self.stats = IngestionStats()
            self.stats.frames_discovered = self.frames_total or 0

            frame_id = 0

            while True:
                success, image = cap.read()

                if not success:
                    break

                frame_id += 1

                if (frame_id - 1) % self.stride != 0:
                    continue

                height, width = image.shape[:2]

                if (
                    self.expected_width is not None
                    and self.expected_height is not None
                    and (
                        width != self.expected_width
                        or height != self.expected_height
                    )
                ):
                    print(
                        "WARNING: frame resolution mismatch: "
                        f"expected {self.expected_width}x"
                        f"{self.expected_height}, got {width}x{height} "
                        f"(frame_id={frame_id})"
                    )

                self.stats.frames_processed += 1

                yield Frame(
                    frame_id=frame_id,
                    elapsed_seconds=(
                        (frame_id - 1) / self.fps_effective
                    ),
                    image=image,
                    source_path=self.video_path,
                    fps_effective=self.fps_effective,
                )

                if (
                    self.max_frames is not None
                    and self.stats.frames_processed >= self.max_frames
                ):
                    break

        finally:
            cap.release()
