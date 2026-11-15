from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from types import SimpleNamespace
from math import hypot, isclose, isfinite
import numpy as np
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.tvs.events.detection import Detection
from src.tvs.events.frame import Frame
from src.tvs.perception.boxes import tlwh_to_xyxy, xyxy_to_tlwh


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    match_thresh: float
    fuse_score: bool
    lost_buffer_seconds: float
    minimum_box_area: float

    confirmation_frames: int
    minimum_track_points: int
    minimum_displacement_pixels: float

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrackerConfig":
        config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(f"Tracker configuration not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Tracker configuration must be a YAML mapping")

        try:
            tracker = config["tracker"]
            lifecycle = config["lifecycle"]

            tracker_config = cls(
                track_high_thresh=float(tracker["track_high_thresh"]),
                track_low_thresh=float(tracker["track_low_thresh"]),
                new_track_thresh=float(tracker["new_track_thresh"]),
                match_thresh=float(tracker["match_thresh"]),
                fuse_score=bool(tracker["fuse_score"]),
                lost_buffer_seconds=float(tracker["lost_buffer_seconds"]),
                minimum_box_area=float(tracker["minimum_box_area"]),
                confirmation_frames=int(lifecycle["confirmation_frames"]),
                minimum_track_points=int(lifecycle["minimum_track_points"]),
                minimum_displacement_pixels=float(
                    lifecycle["minimum_displacement_pixels"]
                ),
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid tracker configuration: {exc}") from exc

        tracker_config.validate()
        return tracker_config

    def validate(self) -> None:
        for name, value in (
            ("track_high_thresh", self.track_high_thresh),
            ("track_low_thresh", self.track_low_thresh),
            ("new_track_thresh", self.new_track_thresh),
            ("match_thresh", self.match_thresh),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.track_low_thresh >= self.track_high_thresh:
            raise ValueError("track_low_thresh must be lower than track_high_thresh")

        if self.lost_buffer_seconds <= 0:
            raise ValueError("lost_buffer_seconds must be positive")

        if self.minimum_box_area < 0:
            raise ValueError("minimum_box_area cannot be negative")

        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")

        if self.minimum_track_points < 1:
            raise ValueError("minimum_track_points must be at least 1")

        if self.minimum_displacement_pixels < 0:
            raise ValueError("minimum_displacement_pixels cannot be negative")

@dataclass(frozen=True, slots=True)

class TrackObservation:
    """One vehicle observation in a source frame."""

    frame_id: int
    internal_track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    source_label: str
    vehicle_type: str
    confidence: float
    detection_index: int
    published_track_id: int | None = None


@dataclass(slots=True)
class _TrackConfirmation:
    consecutive_frames: int
    first_center: tuple[float, float]
    last_update_index: int
    published_track_id: int | None = None

@dataclass(frozen=True, slots=True)
class AssociationStats:
    """Association statistics for one processed frame."""

    frame_id: int
    matched_tracks: int
    unmatched_tracks: int
    unmatched_detections: int


class _InstrumentedBYTETracker(BYTETracker):
    """BYTETracker that records association statistics without changing matching."""

    def __init__(self, args):
        super().__init__(args)
        self.initial_centers: dict[int, tuple[float, float]] = {}
        self.last_association_stats = AssociationStats(
            frame_id=0,
            matched_tracks=0,
            unmatched_tracks=0,
            unmatched_detections=0,
        )

        self._matched_track_ids = set()
        self._matched_detection_indices = set()
    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        existing_ids = {
            track.track_id for track in activated
        }

        super()._init_new_tracks(
            u_detection,
            detections,
            activated,
            refind,
        )

        for track in activated:
            if track.track_id not in existing_ids:
                x, y, width, height = track.tlwh

                self.initial_centers[track.track_id] = (
                    float(x + width / 2.0),
                    float(y + height / 2.0),
                )
    def update(self, results, img=None, feats=None, **kwargs):
        existing_track_ids = {
            track.track_id
            for track in self.tracked_stracks + self.lost_stracks
        }

        self._matched_track_ids = set()
        self._matched_detection_indices = set()
        detection_count = len(results)

        output = super().update(
            results,
            img=img,
            feats=feats,
            **kwargs,
        )

        self.last_association_stats = AssociationStats(
            frame_id=self.frame_id,
            matched_tracks=len(self._matched_track_ids),
            unmatched_tracks=len(
                existing_track_ids - self._matched_track_ids
            ),
            unmatched_detections=(
                detection_count - len(self._matched_detection_indices)
            ),
        )

        return output

    def _apply_match(self, track, det, activated, refind):
        super()._apply_match(
            track,
            det,
            activated,
            refind,
        )

        self._matched_track_ids.add(track.track_id)
        self._matched_detection_indices.add(int(det.idx))

    def _unconfirmed_association(
        self,
        unconfirmed,
        u_detection,
        detections,
        activated,
        removed,
    ):
        result = super()._unconfirmed_association(
            unconfirmed,
            u_detection,
            detections,
            activated,
            removed,
        )

        activated_ids = {
            track.track_id
            for track in activated
        }

        for track in unconfirmed:
            if (
                track.track_id in activated_ids
                and track.frame_id == self.frame_id
            ):
                self._matched_track_ids.add(track.track_id)
                self._matched_detection_indices.add(int(track.idx))

        return result
class Tracker:
    """
    ByteTrack adapter.

    Accepts application Detection objects and returns raw observations.
    Internal ByteTrack IDs are not published IDs.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._tracker: BYTETracker | None = None
        self.effective_fps: float | None = None
        self.effective_track_buffer: int | None = None
        self._confirmation_states: dict[int, _TrackConfirmation] = {}
        self._next_published_id = 1
        self._update_index = 0
    @property
    def association_stats(self) -> AssociationStats | None:
        """Return association statistics for the most recently processed frame."""

        if self._tracker is None:
            return None

        return self._tracker.last_association_stats
    def _initialize(self, fps: float) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Frame FPS must be a positive finite number")

        track_buffer = max(
            1,
            round(self.config.lost_buffer_seconds * fps),
        )

        args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
            track_buffer=track_buffer,
        )

        self._tracker = _InstrumentedBYTETracker(args)
        self.effective_fps = fps
        self.effective_track_buffer = track_buffer

    def _build_boxes(
        self,
        detections: list[Detection],
        frame: Frame,
    ) -> tuple[Boxes, list[int]]:
        rows = []
        eligible_indices = []

        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = tlwh_to_xyxy(detection.bbox)

            area = detection.bbox[2] * detection.bbox[3]

            if area < self.config.minimum_box_area:
                continue

            rows.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    detection.confidence,
                    detection.class_id,
                ]
            )

            eligible_indices.append(index)

        data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

        boxes = Boxes(
            data,
            orig_shape=frame.image.shape[:2],
        )

        return boxes, eligible_indices

    def update(
        self,
        detections: list[Detection],
        frame: Frame,
    ) -> list[TrackObservation]:
        """
        Update ByteTrack with detections from one frame.

        Returns active track observations for the current frame.
        """

        if any(detection.frame_id != frame.frame_id for detection in detections):
            raise ValueError("All detections must belong to the supplied frame")

        fps = float(frame.fps_effective)

        if self._tracker is None:
            self._initialize(fps)

        elif not isclose(
            fps,
            self.effective_fps,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Frame FPS changed during tracking. "
                "A tracker reset is required before changing FPS."
            )

        boxes, eligible_indices = self._build_boxes(
            detections,
            frame,
        )
        self._update_index += 1
        results = self._tracker.update(
            boxes,
            img=frame.image,
        )

        observations = []

        for row in results:
            x1, y1, x2, y2 = row[:4]

            internal_track_id = int(row[4])
            confidence = float(row[5])
            detection_index = int(row[7])

            if not 0 <= detection_index < len(eligible_indices):
                raise RuntimeError("ByteTrack returned an invalid detection index")

            original_index = eligible_indices[detection_index]
            detection = detections[original_index]

            observations.append(
                TrackObservation(
                    frame_id=frame.frame_id,
                    internal_track_id=internal_track_id,
                    bbox=xyxy_to_tlwh((x1, y1, x2, y2)),
                    class_id=detection.class_id,
                    source_label=detection.source_label,
                    vehicle_type=detection.vehicle_type,
                    confidence=confidence,
                    detection_index=original_index,
                )
            )

        published_observations = []

        for observation in observations:
            internal_id = observation.internal_track_id

            source_detection = detections[observation.detection_index]

            x, y, width, height = source_detection.bbox

            center = (
                x + width / 2.0,
                y + height / 2.0,
            )

            state = self._confirmation_states.get(internal_id)

            if state is None:
                # ByteTrack's tracklet_len counts successful updates after
                # activation. Adding one includes the initial detection.
                tracklet_length = next(
                    (
                        track.tracklet_len
                        for track in self._tracker.tracked_stracks
                        if track.track_id == internal_id
                    ),
                    0,
                )

                initial_center = self._tracker.initial_centers.pop(
                    internal_id,
                    center,
                )

                state = _TrackConfirmation(
                    consecutive_frames=tracklet_length + 1,
                    first_center=initial_center,
                    last_update_index=self._update_index,
)

                self._confirmation_states[internal_id] = state

            elif state.published_track_id is not None:
                # Confirmed identities survive temporary missed detections
                # as long as ByteTrack successfully recovers the track.
                state.last_update_index = self._update_index

            else:
                if state.last_update_index == self._update_index - 1:
                    state.consecutive_frames += 1
                else:
                    # A missed processed frame resets tentative confirmation.
                    state.consecutive_frames = 1
                    state.first_center = center

                state.last_update_index = self._update_index

            displacement = hypot(
                center[0] - state.first_center[0],
                center[1] - state.first_center[1],
            )

            if state.published_track_id is None:
                if (
                    state.consecutive_frames >= self.config.confirmation_frames
                    and displacement >= self.config.minimum_displacement_pixels
                ):
                    state.published_track_id = self._next_published_id
                    self._next_published_id += 1

            if state.published_track_id is not None:
                published_observations.append(
                    TrackObservation(
                        frame_id=observation.frame_id,
                        internal_track_id=observation.internal_track_id,
                        published_track_id=state.published_track_id,
                        bbox=observation.bbox,
                        class_id=observation.class_id,
                        source_label=observation.source_label,
                        vehicle_type=observation.vehicle_type,
                        confidence=observation.confidence,
                        detection_index=observation.detection_index,
                    )
                )

        # Remove confirmation records for tracks that have exceeded
        # ByteTrack's recovery buffer.
        stale_ids = [
            internal_id
            for internal_id, state in self._confirmation_states.items()
            if (
                self._update_index - state.last_update_index
                > self.effective_track_buffer
            )
        ]

        for internal_id in stale_ids:
            del self._confirmation_states[internal_id]

        return published_observations
