
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from evaluation.detection.annotations import load_detrac_annotations
from evaluation.detection.evaluator import evaluate_sequence
from tvs.ingestion.image_sequence import ImageSequenceSource
from tvs.perception.detector import DetectorConfig, VehicleDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMERA_CONFIG = PROJECT_ROOT / "config" / "camera_mvi20011.yaml"
DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "ua_detrac"
    / "DETRAC-Train-Annotations-XML"
    / "DETRAC-Train-Annotations-XML"
    / "MVI_20011.xml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO vehicle detection on MVI_20011."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="YOLO inference confidence threshold.",
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="YOLO NMS IoU threshold.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("NMS IoU must be between 0 and 1.")

    print("=" * 65)
    print("YOLO VEHICLE DETECTION EVALUATION")
    print("=" * 65)

    with CAMERA_CONFIG.open("r", encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    source_path = PROJECT_ROOT / camera_config["source_path"]

    print(f"Camera          : {camera_config['camera_id']}")
    print(f"Sequence        : MVI_20011")
    print(f"Confidence      : {args.confidence}")
    print(f"NMS IoU         : {args.nms_iou}")
    print(f"Matching IoU    : 0.5, 0.7")

    annotations = load_detrac_annotations(ANNOTATION_PATH)

    print(f"Ground truth    : {annotations.retained_annotations}")
    print(f"Ignored regions : {len(annotations.ignored_regions)}")

    source = ImageSequenceSource(
        source_dir=source_path,
        fps=camera_config["fps"],
        max_skip_ratio=camera_config["ingestion"]["max_skip_ratio"],
    )

    base_config = DetectorConfig.from_yaml(
        DETECTOR_CONFIG,
        PROJECT_ROOT,
    )

    evaluation_config = replace(
        base_config,
        confidence_threshold=args.confidence,
        iou_threshold=args.nms_iou,
    )

    detector = VehicleDetector(evaluation_config)

    print("\nStarting sequence evaluation...\n")

    result = evaluate_sequence(
        frames=source.frames(),
        detector=detector,
        annotations=annotations,
        iou_thresholds=(0.5, 0.7),
        warmup_frames=1,
    )

    report = {
        "sequence": "MVI_20011",
        "model": str(evaluation_config.weights_path.name),
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.nms_iou,
        "matching_iou_thresholds": [0.5, 0.7],
        "image_size": detector.effective_image_size,
        "device": detector.last_inference_device,
        "fp16": detector.use_half,
        "evaluation": asdict(result),
    }

    report_directory = PROJECT_ROOT / "evaluation" / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)

    report_path = report_directory / (
        f"mvi20011_conf_{args.confidence:.2f}_"
        f"nms_{args.nms_iou:.2f}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Frames processed       : {result.frames_processed}")
    print(f"Total detections       : {result.total_detections}")
    print(
        f"Average detections/frame: "
        f"{result.average_detections_per_frame:.2f}"
    )

    for threshold, metrics in result.metrics_by_iou.items():
        print(f"\nAP @ IoU {threshold:.2f}")
        print(f"  AP       : {metrics.average_precision:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall   : {metrics.recall:.4f}")
        print(f"  TP       : {metrics.true_positive_count}")
        print(f"  FP       : {metrics.false_positive_count}")
        print(f"  FN       : {metrics.false_negative_count}")

    print("\nPERFORMANCE")
    print(f"Latency p50           : {result.inference_latency_p50_ms:.2f} ms")
    print(f"Latency p95           : {result.inference_latency_p95_ms:.2f} ms")
    print(
        f"Peak GPU allocated   : "
        f"{result.peak_gpu_memory_mb:.2f} MB"
    )
    print(f"Inference failures    : {result.inference_failures}")
    print(f"Non-vehicle discarded : {result.discarded_non_vehicle_count}")
    print(f"Clipped boxes         : {result.clipped_box_count}")
    print(f"Degenerate boxes      : {result.degenerate_box_count}")

    print(f"\nReport saved to: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
