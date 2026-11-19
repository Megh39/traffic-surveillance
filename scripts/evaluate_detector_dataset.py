import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from evaluation.detection.annotations import load_detrac_annotations
from evaluation.detection.evaluator import evaluate_sequence
from src.tvs.ingestion.image_sequence import ImageSequenceSource
from src.tvs.perception.detector import DetectorConfig, VehicleDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG = PROJECT_ROOT / "config" / "camera_mvi20011.yaml"
DETECTOR_CONFIG = PROJECT_ROOT / "config" / "detector.yaml"

DATA_ROOT = PROJECT_ROOT / "data" / "ua_detrac"
IMAGE_ROOT = DATA_ROOT / "DETRAC-Images" / "DETRAC-Images"
ANNOTATION_ROOTS = {
    "train": DATA_ROOT / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML",
    "test": DATA_ROOT / "DETRAC-Test-Annotations-XML" / "DETRAC-Test-Annotations-XML",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO vehicle detection across a UA-DETRAC split."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to evaluate (default: test).",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sequences, useful for a short smoke test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("NMS IoU must be between 0 and 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    annotation_root = ANNOTATION_ROOTS[args.split]
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {IMAGE_ROOT}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_root}")

    annotation_paths = sorted(annotation_root.glob("*.xml"))
    if not annotation_paths:
        raise FileNotFoundError(f"No XML annotations found in: {annotation_root}")

    sequence_names = [path.stem for path in annotation_paths]
    if args.limit is not None:
        sequence_names = sequence_names[:args.limit]

    with CAMERA_CONFIG.open("r", encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    base_config = DetectorConfig.from_yaml(DETECTOR_CONFIG, PROJECT_ROOT)
    evaluation_config = replace(
        base_config,
        confidence_threshold=args.confidence,
        iou_threshold=args.nms_iou,
    )
    detector = VehicleDetector(evaluation_config)

    print("=" * 72)
    print("UA-DETRAC YOLO DETECTION EVALUATION")
    print("=" * 72)
    print(f"Split           : {args.split}")
    print(f"Sequences       : {len(sequence_names)}")
    print(f"Confidence      : {args.confidence}")
    print(f"NMS IoU         : {args.nms_iou}")
    print("Matching IoU    : 0.5, 0.7")
    print(f"Model           : {evaluation_config.weights_path.name}")
    print(f"Image size      : {evaluation_config.image_size}")
    print()

    sequence_reports = []
    aggregate = {
        threshold: {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for threshold in (0.5, 0.7)
    }
    totals = {
        "frames_processed": 0,
        "total_detections": 0,
        "inference_failures": 0,
        "discarded_non_vehicle_count": 0,
        "clipped_box_count": 0,
        "degenerate_box_count": 0,
    }

    for index, sequence_name in enumerate(sequence_names, start=1):
        image_dir = IMAGE_ROOT / sequence_name
        annotation_path = annotation_root / f"{sequence_name}.xml"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory missing for {sequence_name}: {image_dir}"
            )

        annotations = load_detrac_annotations(annotation_path)
        source = ImageSequenceSource(
            source_dir=image_dir,
            fps=camera_config["fps"],
            max_skip_ratio=camera_config["ingestion"]["max_skip_ratio"],
        )

        print(
            f"[{index}/{len(sequence_names)}] {sequence_name} "
            f"({annotations.retained_annotations} ground-truth boxes)"
        )

        result = evaluate_sequence(
            frames=source.frames(),
            detector=detector,
            annotations=annotations,
            iou_thresholds=(0.5, 0.7),
            warmup_frames=1,
        )

        sequence_reports.append({
            "sequence": sequence_name,
            "frames_processed": result.frames_processed,
            "total_detections": result.total_detections,
            "ground_truth_detections": annotations.retained_annotations,
            "metrics_by_iou": {
                str(threshold): asdict(metrics)
                for threshold, metrics in result.metrics_by_iou.items()
            },
            "inference_latency_p50_ms": result.inference_latency_p50_ms,
            "inference_latency_p95_ms": result.inference_latency_p95_ms,
            "inference_failures": result.inference_failures,
        })

        totals["frames_processed"] += result.frames_processed
        totals["total_detections"] += result.total_detections
        totals["inference_failures"] += result.inference_failures
        totals["discarded_non_vehicle_count"] += result.discarded_non_vehicle_count
        totals["clipped_box_count"] += result.clipped_box_count
        totals["degenerate_box_count"] += result.degenerate_box_count

        for threshold, metrics in result.metrics_by_iou.items():
            aggregate[threshold]["true_positives"] += metrics.true_positive_count
            aggregate[threshold]["false_positives"] += metrics.false_positive_count
            aggregate[threshold]["false_negatives"] += metrics.false_negative_count

        print(
            f"    frames={result.frames_processed}, "
            f"detections={result.total_detections}, "
            f"recall@0.5={result.metrics_by_iou[0.5].recall:.4f}"
        )

    aggregate_metrics = {}
    for threshold, counts in aggregate.items():
        tp = counts["true_positives"]
        fp = counts["false_positives"]
        fn = counts["false_negatives"]
        aggregate_metrics[str(threshold)] = {
            **counts,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    report = {
        "split": args.split,
        "sequences_evaluated": len(sequence_reports),
        "model": evaluation_config.weights_path.name,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.nms_iou,
        "matching_iou_thresholds": [0.5, 0.7],
        "image_size": detector.effective_image_size,
        "device": detector.last_inference_device,
        "fp16": detector.use_half,
        "totals": totals,
        "aggregate_metrics": aggregate_metrics,
        "sequence_reports": sequence_reports,
    }

    report_directory = PROJECT_ROOT / "evaluation" / "reports" / "detector_dataset"
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / (
        f"{args.split}_conf_{args.confidence:.2f}_nms_{args.nms_iou:.2f}.json"
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 72)
    print("AGGREGATE DETECTION RESULTS")
    print("=" * 72)
    print(f"Sequences evaluated : {len(sequence_reports)}")
    print(f"Frames processed    : {totals['frames_processed']}")
    print(f"Total detections    : {totals['total_detections']}")
    for threshold, metrics in aggregate_metrics.items():
        print(f"\nIoU {float(threshold):.2f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  TP / FP / FN: {metrics['true_positives']} / "
              f"{metrics['false_positives']} / {metrics['false_negatives']}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
