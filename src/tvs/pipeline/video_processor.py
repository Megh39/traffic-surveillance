from __future__ import annotations

import subprocess
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

from src.tvs.pipeline.reid_integration import (
    build_reid_collector,
    persist_completed_run_reid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"
TRACKER_CONFIG = PROJECT_ROOT / "config" / "tracker.yaml"
DATABASE_BATCH_FRAMES = 100


def _encode_h264(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and ensure ffmpeg.exe "
            "is available on PATH."
        ) from exc

    if result.returncode != 0 or not output_path.exists():
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"FFmpeg H.264 encoding failed (exit {result.returncode}): "
            f"{details[-3000:]}"
        )


def process_video(
    source: Any, output_path: str | Path, camera_id: str
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f"{output_path.stem}.intermediate.mp4")

    detector_config = DetectorConfig.from_yaml(DETECTOR_CONFIG, PROJECT_ROOT)
    detector = VehicleDetector(detector_config)
    tracker_config = TrackerConfig.from_yaml(TRACKER_CONFIG)
    tracker = Tracker(tracker_config)
    renderer = TrackRenderer(trail_length=30, box_thickness=2, trail_thickness=2)
    repository = TrafficRepository()
    reid_collector = build_reid_collector(PROJECT_ROOT)
    reid_result = {
        "persisted_descriptors": 0,
        "created_new_identities": 0,
        "candidate_count": 0,
        "skipped_missing_track": 0,
        "candidates": [],
    }

    writer = None
    run_id: UUID | None = None
    pending = []

    frames_processed = 0
    total_detections = 0
    total_publishable_detections = 0
    published_observations = 0
    published_track_ids: set[int] = set()
    inference_latencies: list[float] = []

    total_matched_tracks = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    stage_timings = {
        "detection_seconds": 0.0,
        "tracking_seconds": 0.0,
        "database_seconds": 0.0,
        "rendering_seconds": 0.0,
        "opencv_writing_seconds": 0.0,
        "h264_encoding_seconds": 0.0,
        "reid_persistence_seconds": 0.0,
    }

    started_at = time.perf_counter()

    def flush_batch():
        nonlocal published_observations
        if not pending:
            return
        batch_started = time.perf_counter()
        saved = repository.save_observations_batch(run_id, pending)
        stage_timings["database_seconds"] += time.perf_counter() - batch_started
        published_observations += saved
        pending.clear()

    try:
        repository.register_camera(camera_id)
        frame_iterator = iter(source.frames())
        first_frame = next(frame_iterator, None)
        if first_frame is None:
            raise ValueError("The supplied source contains no readable frames.")

        fps = first_frame.fps_effective
        height, width = first_frame.image.shape[:2]
        if fps <= 0:
            raise ValueError(f"Invalid source FPS: {fps}")

        run_id = repository.create_processing_run(camera_id, fps)

        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create intermediate video: {intermediate_path}"
            )

        for frame in chain([first_frame], frame_iterator):
            frame_height, frame_width = frame.image.shape[:2]
            if frame_width != width or frame_height != height:
                raise ValueError(
                    "Frame resolution changed during processing. "
                    f"Expected {width}x{height}, got "
                    f"{frame_width}x{frame_height}."
                )

            stage_started = time.perf_counter()
            detections = detector.detect(frame)
            publishable = detector.get_publishable_detections(detections)
            stage_timings["detection_seconds"] += time.perf_counter() - stage_started
            inference_latencies.append(detector.last_inference_latency_ms)

            stage_started = time.perf_counter()
            observations = tracker.update(detections, frame)
            for observation in observations:
                reid_collector.add(frame, observation)
            stage_timings["tracking_seconds"] += time.perf_counter() - stage_started

            pending.append((frame, observations))
            for observation in observations:
                if observation.published_track_id is not None:
                    published_track_ids.add(observation.published_track_id)

            association = tracker.association_stats
            if association is not None:
                total_matched_tracks += association.matched_tracks
                total_unmatched_tracks += association.unmatched_tracks
                total_unmatched_detections += association.unmatched_detections

            if len(pending) >= DATABASE_BATCH_FRAMES:
                flush_batch()

            stage_started = time.perf_counter()
            annotated_frame = renderer.render(frame, observations)
            stage_timings["rendering_seconds"] += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            writer.write(annotated_frame)
            stage_timings["opencv_writing_seconds"] += (
                time.perf_counter() - stage_started
            )

            frames_processed += 1
            total_detections += len(detections)
            total_publishable_detections += len(publishable)

        flush_batch()
        reid_started = time.perf_counter()
        reid_result = persist_completed_run_reid(
            repository=repository,
            run_id=run_id,
            collector=reid_collector,
        )
        stage_timings["reid_persistence_seconds"] = time.perf_counter() - reid_started

        writer.release()
        writer = None

        encoding_started = time.perf_counter()
        _encode_h264(intermediate_path, output_path)
        stage_timings["h264_encoding_seconds"] = time.perf_counter() - encoding_started
        intermediate_path.unlink(missing_ok=True)

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
            "processing_fps": (
                round(frames_processed / elapsed_seconds, 2)
                if elapsed_seconds > 0
                else 0.0
            ),
            "average_inference_ms": round(average_inference_ms, 2),
            "stage_timings_seconds": {
                key: round(value, 3) for key, value in stage_timings.items()
            },
            "reid": {
                "persisted_descriptors": reid_result["persisted_descriptors"],
                "created_new_identities": reid_result["created_new_identities"],
                "candidate_count": reid_result["candidate_count"],
                "skipped_missing_track": reid_result["skipped_missing_track"],
                "candidates": reid_result["candidates"],
            },
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
        if intermediate_path.exists():
            intermediate_path.unlink(missing_ok=True)
        repository.close()
