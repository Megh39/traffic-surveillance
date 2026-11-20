#!/usr/bin/env python
"""Cross-camera vehicle ReID evaluation for VeRi XML annotations.

Protocol:
- Every annotated image is used once as a query.
- For a query, gallery images from the same camera are excluded.
- A gallery image is relevant when vehicleID matches the query vehicleID.
- Similarity is cosine similarity (the model output is L2-normalized).
- Reports CMC Rank-1 and mean Average Precision (mAP).

This is an all-annotated-images cross-camera retrieval protocol, not a claim
that it reproduces every official VeRi-776 benchmark split/protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _record_value(element: ET.Element, names: set[str]) -> str | None:
    for key, value in element.attrib.items():
        if _local_name(key) in names:
            return value.strip()
    for child in element:
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return None


def parse_annotations(xml_path: Path) -> list[dict[str, Any]]:
    raw = xml_path.read_bytes()
    # The VeRi test_label.xml uses the GB2312 multibyte encoding.
    # Decode it explicitly to Unicode before passing it to ElementTree.
    # ElementTree's bytes parser rejects some multibyte encodings on Windows.
    try:
        xml_text = raw.decode("gb2312")
    except UnicodeDecodeError:
        # Some redistributed copies are saved as UTF-8 instead.
        xml_text = raw.decode("utf-8-sig")

    root = ET.fromstring(xml_text)

    rows: list[dict[str, Any]] = []
    for element in root.iter():
        image_name = _record_value(element, {"imagename", "image_name", "name", "filename"})
        vehicle_id = _record_value(element, {"vehicleid", "vehicle_id", "id"})
        camera_id = _record_value(element, {"cameraid", "camera_id", "camid"})
        if image_name is None or vehicle_id is None or camera_id is None:
            continue
        rows.append({
            "image_name": image_name,
            "vehicle_id": str(vehicle_id),
            "camera_id": str(camera_id),
            "color_id": _record_value(element, {"colorid", "color_id"}),
            "type_id": _record_value(element, {"typeid", "type_id"}),
        })

    # Prevent accidental duplicate records from nested XML structures.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["image_name"], row["vehicle_id"], row["camera_id"])
        unique[key] = row
    rows = list(unique.values())
    if not rows:
        raise ValueError(
            "No annotation records found. Expected imageName, vehicleID and "
            "cameraID attributes or child elements in the XML."
        )
    return rows


def resolve_image(image_root: Path, image_name: str, index: dict[str, Path]) -> Path | None:
    direct = image_root / image_name
    if direct.is_file():
        return direct
    # XMLs sometimes contain paths relative to the dataset root.
    normalized = image_name.replace("\\", "/")
    direct = image_root / normalized
    if direct.is_file():
        return direct
    return index.get(Path(normalized).name)


def build_image_index(image_root: Path) -> dict[str, Path]:
    print(f"Indexing image files under: {image_root}")
    index: dict[str, Path] = {}
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            # Keep the first deterministic match if duplicate basenames exist.
            index.setdefault(path.name, path)
    print(f"Indexed {len(index)} image files.")
    return index


def preprocess_batch(paths: list[Path], width: int, height: int) -> np.ndarray:
    batch = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read image: {path}")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32).transpose(2, 0, 1)
        batch.append(image)
    return np.stack(batch, axis=0)


def create_session(model_path: Path, intra_threads: int, inter_threads: int):
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = inter_threads
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    # Use fixed model dimensions when available; otherwise use VeRi model defaults.
    height = int(shape[2]) if isinstance(shape[2], int) else 208
    width = int(shape[3]) if isinstance(shape[3], int) else 208
    return session, input_meta.name, session.get_outputs()[0].name, width, height


def extract_embeddings(
    records: list[dict[str, Any]],
    session,
    input_name: str,
    output_name: str,
    width: int,
    height: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_embeddings: list[np.ndarray] = []
    successful: list[dict[str, Any]] = []
    started = time.perf_counter()

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        paths = [row["path"] for row in chunk]
        tensor = preprocess_batch(paths, width, height)
        output = session.run([output_name], {input_name: tensor})[0]
        output = np.asarray(output, dtype=np.float32).reshape(len(chunk), -1)
        if output.shape[1] != 512:
            raise ValueError(
                f"Expected 512-dimensional descriptors, got {output.shape[1]}."
            )
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        valid = np.isfinite(output).all(axis=1) & (norms[:, 0] > 1e-12)
        output = output[valid] / norms[valid]
        for row, embedding in zip(
            [row for row, ok in zip(chunk, valid) if ok], output
        ):
            successful.append(row)
            all_embeddings.append(embedding)

        done = min(start + len(chunk), len(records))
        if done == len(records) or done % max(batch_size * 10, batch_size) == 0:
            elapsed = time.perf_counter() - started
            print(f"Embedded {done}/{len(records)} images ({elapsed:.1f}s)")

    if not all_embeddings:
        raise RuntimeError("No valid embeddings were generated.")
    return np.stack(all_embeddings).astype(np.float32), successful


def evaluate_cross_camera(
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
    query_batch_size: int,
) -> dict[str, Any]:
    ids = np.asarray([r["vehicle_id"] for r in records], dtype=object)
    cams = np.asarray([r["camera_id"] for r in records], dtype=object)
    n = len(records)

    aps: list[float] = []
    rank1_hits = 0
    valid_queries = 0
    no_positive_queries = 0
    start_time = time.perf_counter()

    for start in range(0, n, query_batch_size):
        end = min(start + query_batch_size, n)
        sims = embeddings[start:end] @ embeddings.T
        for local_index, query_index in enumerate(range(start, end)):
            eligible = cams != cams[query_index]  # strictly cross-camera gallery
            relevant = (ids == ids[query_index]) & eligible
            if not np.any(relevant):
                no_positive_queries += 1
                continue

            candidate_indices = np.flatnonzero(eligible)
            order = candidate_indices[
                np.argsort(-sims[local_index, candidate_indices], kind="stable")
            ]
            ranked_relevant = relevant[order]
            hit_positions = np.flatnonzero(ranked_relevant)
            if len(hit_positions) == 0:
                no_positive_queries += 1
                continue

            valid_queries += 1
            if ranked_relevant[0]:
                rank1_hits += 1

            precisions = (
                np.arange(1, len(hit_positions) + 1, dtype=np.float64)
                / (hit_positions + 1)
            )
            aps.append(float(np.mean(precisions)))

        processed = end
        if processed == n or processed % max(query_batch_size * 10, query_batch_size) == 0:
            print(f"Evaluated {processed}/{n} query images")

    elapsed = time.perf_counter() - start_time
    return {
        "protocol": (
            "All annotated images are queries; gallery excludes images from "
            "the query camera; positives share vehicleID across a different camera."
        ),
        "num_images_with_embeddings": n,
        "num_valid_queries": valid_queries,
        "num_queries_without_cross_camera_positive": no_positive_queries,
        "rank1_accuracy": rank1_hits / valid_queries if valid_queries else None,
        "mAP": float(np.mean(aps)) if aps else None,
        "evaluation_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate vehicle-reid-0001 on VeRi XML annotations."
    )
    parser.add_argument("--xml", required=True, type=Path, help="Path to test_label.xml")
    parser.add_argument("--image-root", required=True, type=Path, help="VeRi image directory")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/reid/osnet_ain_x1_0_vehicle_reid.onnx"),
        help="Path to the existing ONNX ReID model",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--intra-threads", type=int, default=2)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/veri_cross_camera_results.json"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/veri_cross_camera_embeddings.npz"),
        help="Where to save embeddings and metadata for reproducibility",
    )
    args = parser.parse_args()

    for name, value in [
        ("XML", args.xml),
        ("image root", args.image_root),
        ("ONNX model", args.model),
    ]:
        if not value.exists():
            raise FileNotFoundError(f"{name} path does not exist: {value}")
    if args.batch_size < 1 or args.query_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers.")

    print("Parsing XML annotations...")
    annotations = parse_annotations(args.xml)
    print(f"XML records: {len(annotations)}")
    print(f"Vehicle identities: {len(set(r['vehicle_id'] for r in annotations))}")
    print(f"Camera IDs: {len(set(r['camera_id'] for r in annotations))}")

    identity_cameras: dict[str, set[str]] = defaultdict(set)
    for row in annotations:
        identity_cameras[row["vehicle_id"]].add(row["camera_id"])
    multi_camera_ids = {vid for vid, camera_set in identity_cameras.items() if len(camera_set) > 1}
    print(f"Identities present in multiple cameras: {len(multi_camera_ids)}")

    image_index = build_image_index(args.image_root)
    records = []
    missing = []
    for row in annotations:
        path = resolve_image(args.image_root, row["image_name"], image_index)
        if path is None:
            missing.append(row["image_name"])
            continue
        records.append({**row, "path": path})

    print(f"Images resolved: {len(records)}")
    print(f"Images missing: {len(missing)}")
    if missing:
        print("First missing filenames:")
        for name in missing[:20]:
            print(f"  {name}")
    if not records:
        raise RuntimeError("No XML image filenames could be resolved under --image-root.")

    session, input_name, output_name, width, height = create_session(
        args.model, args.intra_threads, args.inter_threads
    )
    print(
        f"ONNX model loaded on CPU; input={width}x{height}; "
        f"provider={session.get_providers()[0]}"
    )

    embeddings, successful_records = extract_embeddings(
        records, session, input_name, output_name,
        width, height, args.batch_size
    )
    metrics = evaluate_cross_camera(
        embeddings, successful_records, args.query_batch_size
    )

    successful_ids = {r["vehicle_id"] for r in successful_records}
    successful_cameras = {r["camera_id"] for r in successful_records}
    result = {
        "dataset": "VeRi test_label.xml",
        "model": str(args.model),
        "xml_records": len(annotations),
        "resolved_images": len(records),
        "missing_images": len(missing),
        "embedded_images": len(successful_records),
        "vehicle_identities_in_xml": len(set(r["vehicle_id"] for r in annotations)),
        "camera_ids_in_xml": sorted(set(r["camera_id"] for r in annotations)),
        "identities_with_multiple_cameras_in_xml": len(multi_camera_ids),
        "identities_in_embedded_subset": len(successful_ids),
        "camera_ids_in_embedded_subset": sorted(successful_cameras),
        "metrics": metrics,
        "caveats": [
            "This is an explicitly defined all-images cross-camera retrieval protocol.",
            "It is not necessarily identical to the official VeRi-776 query/gallery protocol.",
            "Confirm the model's training-data provenance before interpreting this as an independent benchmark.",
            "A retrieval match is not a calibrated probability and must not directly trigger production identity merges."
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.embeddings,
        embeddings=embeddings,
        vehicle_ids=np.asarray([r["vehicle_id"] for r in successful_records]),
        camera_ids=np.asarray([r["camera_id"] for r in successful_records]),
        image_names=np.asarray([r["image_name"] for r in successful_records]),
    )

    # Write a per-image manifest so missing/unreadable records can be audited.
    manifest_path = args.output.with_name(args.output.stem + "_manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "vehicle_id", "camera_id", "resolved_path"])
        for row in successful_records:
            writer.writerow([row["image_name"], row["vehicle_id"], row["camera_id"], str(row["path"])])

    print("\n=== CROSS-CAMERA REID RESULTS ===")
    print(f"Embedded images: {metrics['num_images_with_embeddings']}")
    print(f"Valid queries: {metrics['num_valid_queries']}")
    print(f"Queries without cross-camera positives: {metrics['num_queries_without_cross_camera_positive']}")
    print(f"Rank-1 accuracy: {metrics['rank1_accuracy']}")
    print(f"mAP: {metrics['mAP']}")
    print(f"JSON report: {args.output}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
