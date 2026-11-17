
from pathlib import Path
from time import perf_counter
from statistics import mean

import cv2
import yaml

from src.tvs.ingestion.image_sequence import ImageSequenceSource
from src.tvs.perception.detector import DetectorConfig, VehicleDetector
from src.tvs.perception.tracker import Tracker, TrackerConfig
from src.tvs.perception.renderer import TrackRenderer
from src.tvs.db.repository import TrafficRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMERA_CONFIG = PROJECT_ROOT / "config" / "camera_mvi20011.yaml"
DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"
TRACKER_CONFIG = PROJECT_ROOT / "config" / "tracker.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_VIDEO = OUTPUT_DIR / "mvi20011_tracked.mp4"


def main():
    print("=" * 65)
    print("TRAFFIC VIDEO SURVEILLANCE - DETECTION AND TRACKING")
    print("=" * 65)

    with CAMERA_CONFIG.open("r", encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    source_path = PROJECT_ROOT / camera_config["source_path"]
    fps = float(camera_config["fps"])
    camera_id = camera_config["camera_id"]

    if camera_config["source_type"] != "image_sequence":
        raise ValueError("Unsupported source type")

    if fps <= 0:
        raise ValueError("Camera FPS must be positive")

    print(f"Camera ID       : {camera_id}")
    print(f"Source directory: {source_path}")
    print(f"Configured FPS  : {fps}")

    source = ImageSequenceSource(
        source_dir=source_path,
        fps=fps,
        max_skip_ratio=camera_config["ingestion"]["max_skip_ratio"],
    )

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    repository = TrafficRepository()
    run_id = None
    run_completed = False

    video_writer = None

    frames_processed = 0
    total_detections = 0
    total_publishable_detections = 0
    total_persisted_observations = 0

    published_track_ids = set()

    total_matched_tracks = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    inference_latencies = []

    start_time = perf_counter()

    try:
        repository.register_camera(camera_id)

        run_id = repository.create_processing_run(
            camera_id=camera_id,
            source_fps=fps,
        )

        print(f"Database run ID: {run_id}")

        for frame in source.frames():

            if video_writer is None:
                height, width = frame.image.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    str(OUTPUT_VIDEO),
                    fourcc,
                    fps,
                    (width, height),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {OUTPUT_VIDEO}"
                    )

                print("-" * 65)
                print(f"Output resolution: {width}x{height}")
                print(f"Output video     : {OUTPUT_VIDEO}")
                print("-" * 65)

            elif frame.image.shape[:2] != (height, width):
                raise RuntimeError(
                    "Frame resolution changed during processing"
                )

            detections = detector.detect(frame)

            total_detections += len(detections)

            publishable = detector.get_publishable_detections(
                detections
            )

            total_publishable_detections += len(publishable)

            inference_latencies.append(
                detector.last_inference_latency_ms
            )

            observations = tracker.update(
                detections,
                frame,
            )

            # Persist published vehicle tracks and trajectory points.
            saved_count = repository.save_observations(
                run_id,
                frame,
                observations,
            )

            total_persisted_observations += saved_count

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

            video_writer.write(annotated_frame)

            frames_processed += 1

            if frames_processed % 50 == 0:
                print(
                    f"Processed: {frames_processed} frames | "
                    f"Frame ID: {frame.frame_id} | "
                    f"Detections: {len(detections)} | "
                    f"Active published tracks: {len(observations)} | "
                    f"Unique IDs: {len(published_track_ids)} | "
                    f"DB observations: {total_persisted_observations}"
                )

        repository.complete_processing_run(
            run_id=run_id,
            frames_processed=frames_processed,
            total_detections=total_detections,
            published_observations=total_persisted_observations,
            unique_tracks=len(published_track_ids),
        )

        run_completed = True

    except Exception:
        if run_id is not None:
            repository.fail_processing_run(run_id)
        raise

    finally:
        if video_writer is not None:
            video_writer.release()

        repository.close()

    end_time = perf_counter()
    elapsed_time = end_time - start_time

    throughput = (
        frames_processed / elapsed_time
        if elapsed_time > 0
        else 0.0
    )

    average_inference_latency = (
        mean(inference_latencies)
        if inference_latencies
        else 0.0
    )

    print("\n" + "=" * 65)
    print("MVI_20011 TRACKING SUMMARY")
    print("=" * 65)

    print("\nINGESTION")
    print("-" * 65)
    print(f"Frames discovered       : {source.stats.frames_discovered}")
    print(f"Frames processed        : {source.stats.frames_processed}")
    print(f"Frames skipped          : {source.stats.frames_skipped}")
    print(f"Skip ratio              : {source.stats.skip_ratio:.2%}")

    print("\nDETECTION")
    print("-" * 65)
    print(f"Total vehicle detections: {total_detections}")
    print(f"Publishable detections  : {total_publishable_detections}")
    print(
        f"Discarded non-vehicles  : "
        f"{detector.discarded_non_vehicle_count}"
    )
    print(
        f"Discarded vehicle class : "
        f"{detector.discarded_vehicle_class_count}"
    )
    print(f"Clipped bounding boxes  : {detector.clipped_box_count}")
    print(f"Degenerate bounding boxes: {detector.degenerate_box_count}")
    print(f"Inference failures      : {detector.inference_failure_count}")

    print("\nTRACKING")
    print("-" * 65)
    print(f"Unique published IDs    : {len(published_track_ids)}")
    print(f"Matched track events    : {total_matched_tracks}")
    print(f"Unmatched track events  : {total_unmatched_tracks}")
    print(f"Unmatched detection events: {total_unmatched_detections}")

    print("\nDATABASE")
    print("-" * 65)
    print(f"Processing run ID       : {run_id}")
    print(f"Persisted observations  : {total_persisted_observations}")
    print(f"Processing run status   : {'completed' if run_completed else 'failed'}")

    print("\nPERFORMANCE")
    print("-" * 65)
    print(f"Total frames            : {frames_processed}")
    print(f"Total processing time   : {elapsed_time:.2f} seconds")
    print(f"Pipeline throughput     : {throughput:.2f} FPS")
    print(
        f"Average inference time  : "
        f"{average_inference_latency:.2f} ms"
    )
    print(f"Configured FPS          : {fps}")

    print("\nOUTPUT")
    print("-" * 65)

    if video_writer is not None:
        print(f"Annotated video         : {OUTPUT_VIDEO}")
        print(
            f"Output video size       : "
            f"{OUTPUT_VIDEO.stat().st_size / (1024 * 1024):.2f} MB"
        )
    else:
        print("No video was generated.")

    print("=" * 65)


if __name__ == "__main__":
    main()
