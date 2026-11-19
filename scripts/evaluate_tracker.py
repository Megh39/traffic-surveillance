import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import yaml

from evaluation.detection.annotations import load_detrac_annotations
from evaluation.tracking.matching import match_frame_tracks
from evaluation.tracking.metrics import compute_tracking_metrics

from src.tvs.ingestion.image_sequence import ImageSequenceSource
from src.tvs.perception.detector import DetectorConfig, VehicleDetector
from src.tvs.perception.tracker import Tracker, TrackerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "config" / "camera_mvi20011.yaml"

DEFAULT_DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"

DEFAULT_TRACKER_CONFIG = PROJECT_ROOT / "config" / "tracker.yaml"

DEFAULT_ANNOTATION = (
    PROJECT_ROOT
    / "data"
    / "ua_detrac"
    / "DETRAC-Train-Annotations-XML"
    / "DETRAC-Train-Annotations-XML"
    / "MVI_20011.xml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO and ByteTrack against UA-DETRAC."
    )

    parser.add_argument(
        "--sequence",
        default="MVI_20011",
        help="Sequence identifier.",
    )

    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG,
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )

    parser.add_argument(
        "--detector-config",
        type=Path,
        default=DEFAULT_DETECTOR_CONFIG,
    )

    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=DEFAULT_TRACKER_CONFIG,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--matching-iou",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "reports",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.matching_iou <= 1.0:
        raise ValueError("Matching IoU must be in (0, 1].")

    if args.matching_iou == 0.0:
        raise ValueError("Matching IoU must be greater than zero.")

    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

    if args.nms_iou is not None:
        if not 0.0 <= args.nms_iou <= 1.0:
            raise ValueError("NMS IoU must be between 0 and 1.")

    print("=" * 65)
    print("TRAFFIC SURVEILLANCE - TRACKING EVALUATION")
    print("=" * 65)

    # 1. Load camera configuration.

    with args.camera_config.open("r", encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    if camera_config["source_type"] != "image_sequence":
        raise ValueError("Only image sequences are supported.")

    fps = float(camera_config["fps"])

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    source_path = PROJECT_ROOT / camera_config["source_path"]

    # 2. Load annotations.

    annotations = load_detrac_annotations(args.annotation)

    print(f"Sequence             : {args.sequence}")
    print(f"Source               : {source_path}")
    print(f"Annotation file      : {args.annotation}")
    print(f"FPS                  : {fps}")
    print(f"Ground-truth boxes   : {annotations.retained_annotations}")
    print(f"Ignored regions      : {len(annotations.ignored_regions)}")
    print(f"Matching IoU         : {args.matching_iou}")

    # 3. Initialize frame ingestion.

    source = ImageSequenceSource(
        source_dir=source_path,
        fps=fps,
        max_skip_ratio=camera_config["ingestion"]["max_skip_ratio"],
    )

    # 4. Initialize detector.

    base_detector_config = DetectorConfig.from_yaml(
        args.detector_config,
        PROJECT_ROOT,
    )

    overrides = {}

    if args.confidence is not None:
        overrides["confidence_threshold"] = args.confidence

    if args.nms_iou is not None:
        overrides["iou_threshold"] = args.nms_iou

    detector_config = replace(
        base_detector_config,
        **overrides,
    )

    detector = VehicleDetector(detector_config)

    # 5. Initialize tracker.

    tracker_config = TrackerConfig.from_yaml(args.tracker_config)

    tracker = Tracker(tracker_config)

    # 6. Prepare output files.

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = args.output_dir / f"{args.sequence}_tracking_predictions.jsonl"

    report_path = args.output_dir / f"{args.sequence}_tracking_report.json"

    frame_results = []

    frames_processed = 0
    total_detections = 0
    total_published_observations = 0

    total_matched_tracks = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    unique_track_ids = set()

    start_time = perf_counter()

    # 7. Run detection and tracking.

    print("\nStarting detection and tracking...\n")

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as prediction_file:

        for frame in source.frames():

            detections = detector.detect(frame)

            total_detections += len(detections)

            observations = tracker.update(
                detections,
                frame,
            )
            association = tracker.association_stats

            if association is not None:
                total_matched_tracks += association.matched_tracks
                total_unmatched_tracks += association.unmatched_tracks
                total_unmatched_detections += association.unmatched_detections
            # Only published vehicle identities are evaluated.

            published_observations = [
                observation
                for observation in observations
                if observation.published_track_id is not None
            ]

            total_published_observations += len(published_observations)

            for observation in published_observations:
                unique_track_ids.add(observation.published_track_id)

            # Save predictions for this frame.

            prediction_record = {
                "frame_id": frame.frame_id,
                "predictions": [
                    {
                        "track_id": observation.published_track_id,
                        "bbox": list(observation.bbox),
                        "confidence": observation.confidence,
                        "class_id": observation.class_id,
                        "vehicle_type": observation.vehicle_type,
                    }
                    for observation in published_observations
                ],
            }

            prediction_file.write(json.dumps(prediction_record) + "\n")

            # Preserve tracker continuity, but evaluate only annotated frames.
            if frame.frame_id not in annotations.frames:
                continue

            # Evaluate this frame.

            frame_result = match_frame_tracks(
                predictions=published_observations,
                ground_truth=list(annotations.frames[frame.frame_id]),
                ignored_regions=annotations.ignored_regions,
                iou_threshold=args.matching_iou,
                frame_id=frame.frame_id,
            )

            frame_results.append(frame_result)

            frames_processed += 1

            if frames_processed % 50 == 0:
                print(
                    f"Processed {frames_processed} frames | "
                    f"Detections: {total_detections} | "
                    f"Unique IDs: {len(unique_track_ids)}"
                )

    # 8. Calculate tracking metrics.

    metrics = compute_tracking_metrics(frame_results)

    elapsed_time = perf_counter() - start_time

    throughput = frames_processed / elapsed_time if elapsed_time > 0 else 0.0

    # 9. Save evaluation report.

    report = {
        "sequence": args.sequence,
        "model": detector_config.weights_path.name,
        "fps": fps,
        "confidence_threshold": detector_config.confidence_threshold,
        "nms_iou_threshold": detector_config.iou_threshold,
        "matching_iou_threshold": args.matching_iou,
        "frames_processed": frames_processed,
        "total_ground_truth_detections": (metrics.ground_truth_detections),
        "total_yolo_detections": total_detections,
        "published_track_observations": (total_published_observations),
        "unique_published_track_ids": len(unique_track_ids),
        "association_statistics": {
            "matched_tracks": total_matched_tracks,
            "unmatched_tracks": total_unmatched_tracks,
            "unmatched_detections": total_unmatched_detections,
        },
        "processing_time_seconds": elapsed_time,
        "pipeline_throughput_fps": throughput,
        "metrics": asdict(metrics),
        "prediction_file": predictions_path.name,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # 10. Print final report.

    print("\n" + "=" * 65)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 65)

    print(f"Sequence             : {args.sequence}")
    print(f"Frames processed     : {frames_processed}")
    print(f"Ground-truth boxes   : {metrics.ground_truth_detections}")
    print(f"YOLO detections      : {total_detections}")
    print(f"Published observations: {total_published_observations}")
    print(f"Unique track IDs     : {len(unique_track_ids)}")

    print("\nDETECTION AND TRACKING")
    print("-" * 65)
    print(f"TP                   : {metrics.true_positives}")
    print(f"FP                   : {metrics.false_positives}")
    print(f"Ignored predictions  : {metrics.ignored_predictions}")
    print(f"FN                   : {metrics.false_negatives}")
    print(f"Precision            : {metrics.precision:.4f}")
    print(f"Recall               : {metrics.recall:.4f}")

    print("\nTRACKING QUALITY")
    print("-" * 65)
    print(f"MOTA                 : {metrics.mota:.4f}")
    print(f"MOTP                 : {metrics.motp:.4f}")
    print(f"IDF1                 : {metrics.idf1:.4f}")
    print(f"ID Precision         : {metrics.id_precision:.4f}")
    print(f"ID Recall            : {metrics.id_recall:.4f}")
    print(f"ID Switches          : {metrics.identity_switches}")
    print(f"Fragmentations       : {metrics.fragmentations}")

    print("\nPERFORMANCE")
    print("-" * 65)
    print(f"Processing time      : {elapsed_time:.2f} seconds")
    print(f"Pipeline throughput  : {throughput:.2f} FPS")

    print("\nOUTPUT")
    print("-" * 65)
    print(f"Predictions          : {predictions_path}")
    print(f"Evaluation report    : {report_path}")

    print("=" * 65)


if __name__ == "__main__":
    main()
