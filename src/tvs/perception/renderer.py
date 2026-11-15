from collections import deque

import cv2
import numpy as np

from src.tvs.events.frame import Frame
from src.tvs.perception.tracker import TrackObservation


class TrackRenderer:
    """Render tracked vehicles, published IDs, and trajectory trails."""

    def __init__(
        self,
        trail_length: int = 30,
        box_thickness: int = 2,
        trail_thickness: int = 2,
    ):
        if trail_length < 1:
            raise ValueError("trail_length must be at least 1")

        if box_thickness < 1 or trail_thickness < 1:
            raise ValueError("Drawing thickness must be at least 1")

        self.trail_length = trail_length
        self.box_thickness = box_thickness
        self.trail_thickness = trail_thickness

        self.trajectories: dict[int, deque] = {}

    @staticmethod
    def _get_color(track_id: int) -> tuple[int, int, int]:
        """Generate a deterministic BGR colour for a published track ID."""

        return (
            (track_id * 67 + 50) % 256,
            (track_id * 131 + 100) % 256,
            (track_id * 197 + 150) % 256,
        )

    def reset(self) -> None:
        """Clear all stored trajectory histories."""

        self.trajectories.clear()

    def render(
        self,
        frame: Frame,
        observations: list[TrackObservation],
    ) -> np.ndarray:
        """
        Draw published vehicle tracks on a copy of the supplied frame.

        Only observations with published tracking IDs are rendered.
        """

        image = frame.image.copy()

        for observation in observations:
            if observation.frame_id != frame.frame_id:
                raise ValueError(
                    "All track observations must belong to the supplied frame"
                )

            track_id = observation.published_track_id

            if track_id is None:
                continue

            x, y, width, height = observation.bbox

            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + width))
            y2 = int(round(y + height))

            center = (
                int(round(x + width / 2.0)),
                int(round(y + height / 2.0)),
            )

            color = self._get_color(track_id)

            if track_id not in self.trajectories:
                self.trajectories[track_id] = deque(
                    maxlen=self.trail_length
                )

            trajectory = self.trajectories[track_id]
            trajectory.append(center)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                self.box_thickness,
            )

            label = f"ID: {track_id}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            for index in range(1, len(trajectory)):
                cv2.line(
                    image,
                    trajectory[index - 1],
                    trajectory[index],
                    color,
                    self.trail_thickness,
                    cv2.LINE_AA,
                )

            cv2.circle(
                image,
                center,
                3,
                color,
                -1,
            )

        return image
