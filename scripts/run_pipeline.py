from pathlib import Path
from time import perf_counter
from statistics import mean

import cv2
import yaml

from tvs.ingestion.image_sequence import ImageSequenceSource
from tvs.perception.detector import DetectorConfig, VehicleDetector
from tvs.perception.tracker import Tracker, TrackerConfig
from tvs.perception.renderer import TrackRenderer


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

    # ---------------------------------------------------------
    # 1. Load camera configuration
    # ---------------------------------------------------------

    with CAMERA_CONFIG.open("r", encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    source_path = PROJECT_ROOT / camera_config["source_path"]
    fps = float(camera_config["fps"])

    if camera_config["source_type"] != "image_sequence":
        raise ValueError("Unsupported source type")

    if fps <= 0:
        raise ValueError("Camera FPS must be positive")

    print(f"Camera ID       : {camera_config['camera_id']}")
    print(f"Source directory: {source_path}")
    print(f"Configured FPS  : {fps}")

    # ---------------------------------------------------------
    # 2. Initialize ingestion
    # ---------------------------------------------------------

    source = ImageSequenceSource(
        source_dir=source_path,
        fps=fps,
        max_skip_ratio=camera_config["ingestion"]["max_skip_ratio"],
    )

    # ---------------------------------------------------------
    # 3. Initialize detector
    # ---------------------------------------------------------

    detector_config = DetectorConfig.from_yaml(
        DETECTOR_CONFIG,
        PROJECT_ROOT,
    )

    detector = VehicleDetector(detector_config)

    # ---------------------------------------------------------
    # 4. Initialize tracker
    # ---------------------------------------------------------

    tracker_config = TrackerConfig.from_yaml(TRACKER_CONFIG)
    tracker = Tracker(tracker_config)

    # ---------------------------------------------------------
    # 5. Initialize renderer
    # ---------------------------------------------------------

    renderer = TrackRenderer(
        trail_length=30,
        box_thickness=2,
        trail_thickness=2,
    )

    # ---------------------------------------------------------
    # 6. Initialize output and counters
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_writer = None

    frames_processed = 0
    total_detections = 0
    total_publishable_detections = 0

    published_track_ids = set()

    total_matched_tracks = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    inference_latencies = []

    start_time = perf_counter()

    try:
        # -----------------------------------------------------
        # 7. Process frames
        # -----------------------------------------------------

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

            # YOLO detection.
            detections = detector.detect(frame)

            total_detections += len(detections)

            publishable = detector.get_publishable_detections(
                detections
            )

            total_publishable_detections += len(publishable)

            inference_latencies.append(
                detector.last_inference_latency_ms
            )

            # IMPORTANT:
            # Pass ALL detections to ByteTrack.
            # Do not pass only publishable detections.
            observations = tracker.update(
                detections,
                frame,
            )

            # Collect published vehicle IDs.
            for observation in observations:
                if observation.published_track_id is not None:
                    published_track_ids.add(
                        observation.published_track_id
                    )

            # Collect association statistics.
            association = tracker.association_stats

            if association is not None:
                total_matched_tracks += association.matched_tracks
                total_unmatched_tracks += association.unmatched_tracks
                total_unmatched_detections += (
                    association.unmatched_detections
                )

            # Render tracked vehicles.
            annotated_frame = renderer.render(
                frame,
                observations,
            )

            # Write annotated frame.
            video_writer.write(annotated_frame)

            frames_processed += 1

            # Progress report every 50 frames.
            if frames_processed % 50 == 0:
                print(
                    f"Processed: {frames_processed} frames | "
                    f"Frame ID: {frame.frame_id} | "
                    f"Detections: {len(detections)} | "
                    f"Active published tracks: {len(observations)} | "
                    f"Unique IDs: {len(published_track_ids)}"
                )

    finally:
        if video_writer is not None:
            video_writer.release()

    end_time = perf_counter()
    elapsed_time = end_time - start_time

    # ---------------------------------------------------------
    # 8. Calculate performance statistics
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 9. Print final report
    # ---------------------------------------------------------

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
        print(f"Output video size       : {OUTPUT_VIDEO.stat().st_size / (1024 * 1024):.2f} MB")
    else:
        print("No video was generated.")

    print("=" * 65)


if __name__ == "__main__":
    main()
