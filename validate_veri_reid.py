import argparse
import csv
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description="Pairwise VeRi vehicle Re-ID validation (CPU only).")
    p.add_argument("--dataset", type=Path, default=Path("data/VeRi"))
    p.add_argument("--model", type=Path, default=Path("models/reid/osnet_ain_x1_0_vehicle_reid.onnx"))
    p.add_argument("--output", type=Path, default=Path("reid_validation_output"))
    p.add_argument("--max-vehicles", type=int, default=50, help="0 means all eligible identities")
    p.add_argument("--images-per-vehicle", type=int, default=8)
    p.add_argument("--max-positive-pairs", type=int, default=2000)
    p.add_argument("--max-negative-pairs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    xml_path = args.dataset / "test_label.xml"
    image_dir = args.dataset / "image_test"
    for path in (xml_path, image_dir, args.model):
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")

    root = ET.fromstring(xml_path.read_bytes().decode("gb2312"))
    groups = defaultdict(list)
    missing = 0
    for item in root.findall(".//Item"):
        name, vid, cam = item.get("imageName"), item.get("vehicleID"), item.get("cameraID")
        if not name or not vid or not cam:
            continue
        image_path = image_dir / name
        if image_path.is_file():
            groups[vid].append((name, cam, image_path))
        else:
            missing += 1

    rng = random.Random(args.seed)
    eligible = sorted(v for v, rows in groups.items() if len(rows) >= 2)
    if args.max_vehicles > 0 and len(eligible) > args.max_vehicles:
        eligible = sorted(rng.sample(eligible, args.max_vehicles))

    selected = {}
    for vid in eligible:
        camera_rows = defaultdict(list)
        for row in sorted(groups[vid], key=lambda x: (x[1], x[0])):
            camera_rows[row[1]].append(row)
        cameras = sorted(camera_rows)
        chosen = []
        while cameras and len(chosen) < args.images_per_vehicle:
            remaining = []
            for cam in cameras:
                if camera_rows[cam] and len(chosen) < args.images_per_vehicle:
                    chosen.append(camera_rows[cam].pop(0))
                if camera_rows[cam]:
                    remaining.append(cam)
            cameras = remaining
        if len(chosen) >= 2:
            selected[vid] = chosen

    ids = sorted(selected)
    positives = []
    for vid in ids:
        rows = selected[vid]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][1] != rows[j][1]:
                    positives.append((vid, rows[i], vid, rows[j], 1))
    if len(positives) > args.max_positive_pairs:
        positives = rng.sample(positives, args.max_positive_pairs)

    negatives, seen = [], set()
    attempts = 0
    while len(negatives) < args.max_negative_pairs and attempts < args.max_negative_pairs * 30:
        attempts += 1
        if len(ids) < 2:
            break
        va, vb = rng.sample(ids, 2)
        ra, rb = rng.choice(selected[va]), rng.choice(selected[vb])
        key = (va, ra[0], vb, rb[0])
        if key not in seen:
            seen.add(key)
            negatives.append((va, ra, vb, rb, 0))
    pairs = positives + negatives
    if not positives or not negatives:
        raise SystemExit("Could not generate both positive and negative pairs. Increase --max-vehicles.")

    unique = {}
    for _, a, _, b, _ in pairs:
        unique[a[0]], unique[b[0]] = a[2], b[2]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    embeddings = {}
    print(f"Selected identities: {len(selected)} | images: {len(unique)}")
    print(f"Positive pairs: {len(positives)} | negative pairs: {len(negatives)}")
    for i, (name, path) in enumerate(sorted(unique.items()), 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        resized = cv2.resize(image, (208, 208), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32), (2, 0, 1))[None, ...]
        raw = session.run(None, {input_name: tensor})[0]
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.size != 512 or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"Unexpected output for {name}: {np.asarray(raw).shape}")
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise RuntimeError(f"Zero-norm embedding: {name}")
        embeddings[name] = vector / norm
        if i % 50 == 0 or i == len(unique):
            print(f"Embedded {i}/{len(unique)}")

    rows = []
    for va, a, vb, b, label in pairs:
        score = float(np.dot(embeddings[a[0]], embeddings[b[0]]))
        rows.append((va, a[0], a[1], vb, b[0], b[1], label, score))

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "pair_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vehicle_id_a", "image_a", "camera_a", "vehicle_id_b",
                    "image_b", "camera_b", "same_vehicle", "cosine_similarity"])
        w.writerows(rows)

    pos = [r[7] for r in rows if r[6] == 1]
    neg = [r[7] for r in rows if r[6] == 0]
    candidates = sorted(set(pos + neg))
    best = None
    for threshold in candidates:
        tp = sum(r[6] == 1 and r[7] >= threshold for r in rows)
        fp = sum(r[6] == 0 and r[7] >= threshold for r in rows)
        fn = sum(r[6] == 1 and r[7] < threshold for r in rows)
        tn = sum(r[6] == 0 and r[7] < threshold for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(rows)
        result = (f1, precision, threshold, tp, fp, fn, tn, recall, accuracy)
        if best is None or result[:2] > best[:2]:
            best = result

    f1, precision, threshold, tp, fp, fn, tn, recall, accuracy = best
    report = args.output / "validation_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("VeRi-776 vehicle Re-ID pairwise validation\\n")
        f.write("==========================================\\n")
        f.write(f"Dataset: {args.dataset.resolve()}\\nModel: {args.model.resolve()}\\n")
        f.write("Inference provider: CPUExecutionProvider\\n")
        f.write(f"Selected identities: {len(selected)}\\nEmbedded images: {len(unique)}\\n")
        f.write(f"Missing image files skipped: {missing}\\n")
        f.write(f"Positive cross-camera pairs: {len(pos)}\\nNegative pairs: {len(neg)}\\n\\n")
        f.write(f"Positive mean/min/max: {np.mean(pos):.6f} / {np.min(pos):.6f} / {np.max(pos):.6f}\\n")
        f.write(f"Negative mean/min/max: {np.mean(neg):.6f} / {np.min(neg):.6f} / {np.max(neg):.6f}\\n\\n")
        f.write("Best F1 on these sampled pairs (exploratory only):\\n")
        f.write(f"Candidate threshold: {threshold:.6f}\\nTP={tp}, FP={fp}, FN={fn}, TN={tn}\\n")
        f.write(f"Precision={precision:.6f}\\nRecall={recall:.6f}\\nF1={f1:.6f}\\nAccuracy={accuracy:.6f}\\n\\n")
        f.write("LIMITATIONS:\\n")
        f.write("- Pairwise metrics are not standard VeRi Rank-1/mAP.\\n")
        f.write("- The threshold is optimized and evaluated on the same sampled pairs; it is exploratory, not deployable.\\n")
        f.write("- VeRi may overlap the model's training/evaluation provenance; it is not an independent test of your cameras.\\n")
        f.write("- Calibrate on held-out footage from your own cameras before deployment.\\n")
        f.write("- VeRi README states research-only use; check terms before commercial use or redistribution.\\n")

    print("\\nValidation complete.")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Report: {report.resolve()}")
    print(f"Positive mean: {np.mean(pos):.6f} | Negative mean: {np.mean(neg):.6f}")
    print(f"Exploratory best-F1 threshold: {threshold:.6f} (DO NOT deploy directly)")


if __name__ == "__main__":
    main()
