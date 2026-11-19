import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = ROOT / "data/ua_detrac/DETRAC-Images/DETRAC-Images"
TRAIN_ROOT = ROOT / "data/ua_detrac/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML"
TEST_ROOT = ROOT / "data/ua_detrac/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential UA-DETRAC tracking evaluation."
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--camera-config", type=Path, default=ROOT / "config/camera_mvi20011.yaml")
    parser.add_argument("--detector-config", type=Path, default=ROOT / "config/detector.yaml")
    parser.add_argument("--tracker-config", type=Path, default=ROOT / "config/tracker.yaml")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--matching-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/reports/dataset")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 < args.matching_iou <= 1:
        parser.error("--matching-iou must be in (0, 1]")
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.nms_iou is not None and not 0 <= args.nms_iou <= 1:
        parser.error("--nms-iou must be between 0 and 1")

    return args


def configuration_fingerprint(args, fps, max_skip_ratio):
    config = {
        "detector": args.detector_config.read_text(encoding="utf-8"),
        "tracker": args.tracker_config.read_text(encoding="utf-8"),
        "fps": fps,
        "max_skip_ratio": max_skip_ratio,
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "matching_iou": args.matching_iou,
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_reports(reports):
    metric_names = [
        "true_positives",
        "false_positives",
        "false_negatives",
        "ground_truth_detections",
        "prediction_detections",
        "ignored_predictions",
        "identity_switches",
        "fragmentations",
        "id_true_positives",
        "id_false_positives",
        "id_false_negatives",
    ]

    totals = {name: sum(r["metrics"][name] for r in reports) for name in metric_names}

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    gt = totals["ground_truth_detections"]

    idtp = totals["id_true_positives"]
    idfp = totals["id_false_positives"]
    idfn = totals["id_false_negatives"]

    total_iou = sum(
        r["metrics"]["motp"] * r["metrics"]["true_positives"]
        for r in reports
    )

    totals.update({
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "mota": 1 - (fn + fp + totals["identity_switches"]) / gt if gt else 0.0,
        "motp": total_iou / tp if tp else 0.0,
        "id_precision": idtp / (idtp + idfp) if idtp + idfp else 0.0,
        "id_recall": idtp / (idtp + idfn) if idtp + idfn else 0.0,
        "idf1": (2 * idtp) / (2 * idtp + idfp + idfn)
        if 2 * idtp + idfp + idfn else 0.0,
    })

    processing_time = sum(r["processing_time_seconds"] for r in reports)
    frames = sum(r["frames_processed"] for r in reports)

    return {
        "sequences_evaluated": len(reports),
        "frames_processed": frames,
        "total_ground_truth_detections": gt,
        "total_yolo_detections": sum(r["total_yolo_detections"] for r in reports),
        "published_track_observations": sum(
            r["published_track_observations"] for r in reports
        ),
        "unique_published_track_ids": sum(
            r["unique_published_track_ids"] for r in reports
        ),
        "processing_time_seconds": processing_time,
        "pipeline_throughput_fps": frames / processing_time if processing_time else 0.0,
        "metrics": totals,
        "sequence_reports": [r["sequence"] for r in reports],
    }


def evaluate_split(args, split, annotation_root, fps, max_skip_ratio, fingerprint):
    image_names = {p.name for p in IMAGE_ROOT.iterdir() if p.is_dir()}
    annotation_files = sorted(annotation_root.glob("*.xml"))
    annotation_names = {p.stem for p in annotation_files}

    if image_names != annotation_names | (image_names - annotation_names):
        raise RuntimeError("Unable to inspect image sequence directories.")

    missing = annotation_names - image_names
    if missing:
        raise RuntimeError(f"Missing image directories: {sorted(missing)}")

    sequences = sorted(annotation_names)

    if args.limit is not None:
        sequences = sequences[:args.limit]

    output_dir = args.output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    if manifest.get("fingerprint") != fingerprint or args.no_resume:
        manifest = {"fingerprint": fingerprint, "completed": []}

    completed = set(manifest.get("completed", []))
    reports = []

    print(f"\n{'=' * 65}\n{split.upper()} SEQUENCES: {len(sequences)}\n{'=' * 65}")

    for index, sequence in enumerate(sequences, start=1):
        report_path = output_dir / f"{sequence}_tracking_report.json"
        prediction_path = output_dir / f"{sequence}_tracking_predictions.jsonl"

        if (
            sequence in completed
            and report_path.is_file()
            and prediction_path.is_file()
        ):
            print(f"[{index}/{len(sequences)}] SKIP {sequence} (already completed)")
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue

        image_dir = IMAGE_ROOT / sequence
        annotation_path = annotation_root / f"{sequence}.xml"

        camera_config = {
            "source_type": "image_sequence",
            "source_path": str(image_dir.resolve()),
            "fps": fps,
            "ingestion": {"max_skip_ratio": max_skip_ratio},
        }

        camera_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                encoding="utf-8",
                delete=False,
            ) as camera_file:
                yaml.safe_dump(camera_config, camera_file)
                camera_file_path = Path(camera_file.name)

            command = [
                sys.executable,
                "-m",
                "scripts.evaluate_tracker",
                "--sequence", sequence,
                "--camera-config", str(camera_file_path),
                "--annotation", str(annotation_path),
                "--detector-config", str(args.detector_config.resolve()),
                "--tracker-config", str(args.tracker_config.resolve()),
                "--matching-iou", str(args.matching_iou),
                "--output-dir", str(output_dir.resolve()),
            ]

            if args.confidence is not None:
                command += ["--confidence", str(args.confidence)]
            if args.nms_iou is not None:
                command += ["--nms-iou", str(args.nms_iou)]

            print(f"\n[{index}/{len(sequences)}] PROCESSING {sequence}")

            subprocess.run(command, cwd=ROOT, check=True)

            if not report_path.is_file() or not prediction_path.is_file():
                raise RuntimeError(f"Incomplete output for {sequence}")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            reports.append(report)

            completed.add(sequence)
            manifest = {
                "fingerprint": fingerprint,
                "completed": sorted(completed),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

        finally:
            if camera_file_path is not None:
                camera_file_path.unlink(missing_ok=True)

    if not reports:
        raise RuntimeError(f"No completed reports available for {split}")

    aggregate = aggregate_reports(reports)
    aggregate_path = output_dir / f"{split}_aggregate_report.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print(f"\n{split.upper()} AGGREGATE")
    print(f"Sequences       : {aggregate['sequences_evaluated']}")
    print(f"Frames          : {aggregate['frames_processed']}")
    print(f"MOTA            : {aggregate['metrics']['mota']:.4f}")
    print(f"MOTP            : {aggregate['metrics']['motp']:.4f}")
    print(f"IDF1            : {aggregate['metrics']['idf1']:.4f}")
    print(f"Aggregate report: {aggregate_path}")


def main():
    args = parse_args()

    with args.camera_config.open("r", encoding="utf-8") as file:
        camera = yaml.safe_load(file)

    fps = float(camera["fps"])
    max_skip_ratio = float(camera["ingestion"]["max_skip_ratio"])

    fingerprint = configuration_fingerprint(args, fps, max_skip_ratio)

    splits = {
        "train": TRAIN_ROOT,
        "test": TEST_ROOT,
    }

    selected = ["train", "test"] if args.split == "all" else [args.split]

    for split in selected:
        evaluate_split(
            args,
            split,
            splits[split],
            fps,
            max_skip_ratio,
            fingerprint,
        )


if __name__ == "__main__":
    main()
