
from __future__ import annotations

import time
from itertools import chain
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2

from src.tvs.db.repository import TrafficRepository
from src.tvs.perception.detector import DetectorConfig, VehicleDetector
from src.tvs.perception.renderer import TrackRenderer
from src.tvs.perception.tracker import Tracker, TrackerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"
TRACKER_CONFIG = PROJECT_ROOT / "config" / "tracker.yaml"


def process_video(
    source: Any,
    output_path: str | Path,
    camera_id: str,
) -> dict[str, Any]:
    """
    Process an image-sequence or video source through detection,
    tracking, rendering, and PostgreSQL persistence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector_config = DetectorConfig.from_yaml(
        DETECTOR_CONFIG,
        PROJECT_ROOT,
    )
    detector = VehicleDetector(detector_config)

    tracker_config = TrackerConfig.from_yaml(TRACKER_CONFIG)
    tracker = Tracker(tracker_config)

    renderer = TrackRenderer(
        trail_length=30,
        box_thickness=2,
        trail_thickness=2,
    )

    repository = TrafficRepository()

    writer = None
    run_id: UUID | None = None

    frames_processed = 0
    total_detections = 0
    total_publishable_detections = 0
    published_observations = 0

    published_track_ids: set[int] = set()
    inference_latencies: list[float] = []

    total_matched_tracks = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    started_at = time.perf_counter()

    try:
        repository.register_camera(camera_id)

        frame_iterator = iter(source.frames())
        first_frame = next(frame_iterator, None)

        if first_frame is None:
            raise ValueError(
                "The supplied source contains no readable frames."
            )

        fps = first_frame.fps_effective
        height, width = first_frame.image.shape[:2]

        if fps <= 0:
            raise ValueError(f"Invalid source FPS: {fps}")

        run_id = repository.create_processing_run(
            camera_id=camera_id,
            source_fps=fps,
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {output_path}"
            )

        for frame in chain([first_frame], frame_iterator):
            frame_height, frame_width = frame.image.shape[:2]

            if frame_width != width or frame_height != height:
                raise ValueError(
                    "Frame resolution changed during processing. "
                    f"Expected {width}x{height}, got "
                    f"{frame_width}x{frame_height}."
                )

            detections = detector.detect(frame)

            publishable_detections = (
                detector.get_publishable_detections(detections)
            )

            inference_latencies.append(
                detector.last_inference_latency_ms
            )

            observations = tracker.update(detections, frame)

            saved_count = repository.save_observations(
                run_id=run_id,
                frame=frame,
                observations=observations,
            )

            published_observations += saved_count

            for observation in observations:
                if observation.published_track_id is not None:
                    published_track_ids.add(
                        observation.published_track_id
                    )

            association = tracker.association_stats

            if association is not None:
                total_matched_tracks += association.matched_tracks
                total_unmatched_tracks += association.unmatched_tracks
                total_unmatched_detections += (
                    association.unmatched_detections
                )

            annotated_frame = renderer.render(
                frame,
                observations,
            )

            writer.write(annotated_frame)

            frames_processed += 1
            total_detections += len(detections)
            total_publishable_detections += len(
                publishable_detections
            )

        elapsed_seconds = time.perf_counter() - started_at

        repository.complete_processing_run(
            run_id=run_id,
            frames_processed=frames_processed,
            total_detections=total_detections,
            published_observations=published_observations,
            unique_tracks=len(published_track_ids),
        )

        average_inference_ms = (
            sum(inference_latencies) / len(inference_latencies)
            if inference_latencies
            else 0.0
        )

        return {
            "run_id": str(run_id),
            "camera_id": camera_id,
            "status": "completed",
            "output_path": str(output_path),
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "publishable_detections": total_publishable_detections,
            "published_observations": published_observations,
            "unique_tracks": len(published_track_ids),
            "fps": fps,
            "width": width,
            "height": height,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "processing_fps": round(
                frames_processed / elapsed_seconds,
                2,
            ) if elapsed_seconds > 0 else 0.0,
            "average_inference_ms": round(
                average_inference_ms,
                2,
            ),
            "association_stats": {
                "matched_tracks": total_matched_tracks,
                "unmatched_tracks": total_unmatched_tracks,
                "unmatched_detections": total_unmatched_detections,
            },
        }

    except Exception:
        if run_id is not None:
            repository.fail_processing_run(run_id)
        raise

    finally:
        if writer is not None:
            writer.release()

        repository.close()
