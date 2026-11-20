#!/usr/bin/env python
"""Memory-conscious camera-pair and hard-negative audit for VeRi embeddings.

Reads the NPZ produced by evaluate_veri_cross_camera.py. Does not load images,
run ONNX, modify the project, or construct a full N x N similarity matrix.

Outputs:
- camera_pair_metrics.csv: ordered query-camera -> gallery-camera metrics
- hard_negative_examples.csv: top incorrect cross-camera result for each
  query whose top result is incorrect in the full cross-camera gallery
- identity_metrics.csv: per-identity Rank-1 and AP in the full cross-camera gallery
- audit_summary.json: dataset counts and protocol description
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_embeddings(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"embeddings", "vehicle_ids", "camera_ids", "image_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        vehicle_ids = data["vehicle_ids"].astype(str)
        camera_ids = data["camera_ids"].astype(str)
        image_names = data["image_names"].astype(str)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Expected embeddings shaped (N, 512), got {embeddings.shape}")
    if not (len(embeddings) == len(vehicle_ids) == len(camera_ids) == len(image_names)):
        raise ValueError("Embedding and metadata array lengths do not match.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("At least one embedding has near-zero norm.")
    # Normalize again defensively; cosine similarity becomes a dot product.
    embeddings = embeddings / norms[:, None]
    return embeddings, vehicle_ids, camera_ids, image_names


def rank_metrics(similarities: np.ndarray, gallery_ids: np.ndarray, query_id: str):
    relevant = gallery_ids == query_id
    positives = int(np.count_nonzero(relevant))
    if positives == 0:
        return None, None, 0, None

    order = np.argsort(-similarities, kind="stable")
    ranked_relevant = relevant[order]
    hits = np.flatnonzero(ranked_relevant)
    rank1 = float(bool(ranked_relevant[0]))
    precisions = np.arange(1, len(hits) + 1, dtype=np.float64) / (hits + 1)
    ap = float(np.mean(precisions))
    top_index = int(order[0])
    return rank1, ap, positives, top_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/veri_cross_camera_embeddings.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/veri_embedding_audit"))
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--max-hard-negatives", type=int, default=500)
    args = parser.parse_args()

    if not args.embeddings.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")
    if args.query_batch_size < 1:
        raise ValueError("--query-batch-size must be positive.")

    started = time.perf_counter()
    print(f"Loading embeddings: {args.embeddings}")
    x, ids, cams, names = load_embeddings(args.embeddings)
    n = len(ids)
    camera_values = sorted(set(cams.tolist()))
    identity_values = sorted(set(ids.tolist()))
    print(f"Loaded {n} embeddings, {len(identity_values)} identities, {len(camera_values)} cameras.")
    print("No ONNX inference or image loading will be performed.")

    camera_indices = {cam: np.flatnonzero(cams == cam) for cam in camera_values}
    identity_query_indices = {vid: np.flatnonzero(ids == vid) for vid in identity_values}

    # Per-query full cross-camera metrics and top incorrect result.
    per_identity = defaultdict(lambda: {"queries": 0, "rank1_hits": 0, "ap_sum": 0.0})
    hard_negatives = []
    query_count = 0
    rank1_total = 0
    ap_total = 0.0

    for start in range(0, n, args.query_batch_size):
        end = min(start + args.query_batch_size, n)
        sim_block = x[start:end] @ x.T
        for local, qi in enumerate(range(start, end)):
            eligible = cams != cams[qi]
            gallery = np.flatnonzero(eligible)
            sims = sim_block[local, gallery]
            gallery_ids = ids[gallery]
            rank1, ap, positive_count, top_local = rank_metrics(sims, gallery_ids, ids[qi])
            if rank1 is None:
                continue

            query_count += 1
            rank1_total += int(rank1)
            ap_total += ap
            stats = per_identity[ids[qi]]
            stats["queries"] += 1
            stats["rank1_hits"] += int(rank1)
            stats["ap_sum"] += ap

            if not rank1:
                top_gallery_index = int(gallery[top_local])
                hard_negatives.append({
                    "query_image": names[qi],
                    "query_vehicle_id": ids[qi],
                    "query_camera_id": cams[qi],
                    "top_wrong_image": names[top_gallery_index],
                    "top_wrong_vehicle_id": ids[top_gallery_index],
                    "top_wrong_camera_id": cams[top_gallery_index],
                    "cosine_similarity": float(sims[top_local]),
                    "positive_gallery_images": positive_count,
                })

        if end == n or end % max(args.query_batch_size * 100, args.query_batch_size) == 0:
            print(f"Full-gallery audit: {end}/{n} queries")

    # Ordered camera-pair metrics: query camera A, gallery camera B only.
    pair_rows = []
    total_pairs = len(camera_values) * (len(camera_values) - 1)
    pair_number = 0
    for query_cam in camera_values:
        q_indices = camera_indices[query_cam]
        for gallery_cam in camera_values:
            if gallery_cam == query_cam:
                continue
            g_indices = camera_indices[gallery_cam]
            g_ids = ids[g_indices]
            pair_aps = []
            pair_rank1 = 0
            valid = 0
            no_positive = 0

            for q_start in range(0, len(q_indices), args.query_batch_size):
                q_chunk = q_indices[q_start:q_start + args.query_batch_size]
                sims_block = x[q_chunk] @ x[g_indices].T
                for local, qi in enumerate(q_chunk):
                    rank1, ap, _, _ = rank_metrics(sims_block[local], g_ids, ids[qi])
                    if rank1 is None:
                        no_positive += 1
                        continue
                    valid += 1
                    pair_rank1 += int(rank1)
                    pair_aps.append(ap)

            pair_rows.append({
                "query_camera": query_cam,
                "gallery_camera": gallery_cam,
                "query_images": len(q_indices),
                "gallery_images": len(g_indices),
                "valid_queries": valid,
                "queries_without_positive": no_positive,
                "rank1_accuracy": pair_rank1 / valid if valid else "",
                "mAP": float(np.mean(pair_aps)) if pair_aps else "",
            })
            pair_number += 1
        print(f"Camera-pair analysis: {pair_number}/{total_pairs} ordered pairs")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_path = args.output_dir / "camera_pair_metrics.csv"
    with pair_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    hard_negatives.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    hard_path = args.output_dir / "hard_negative_examples.csv"
    hard_fields = [
        "query_image", "query_vehicle_id", "query_camera_id",
        "top_wrong_image", "top_wrong_vehicle_id", "top_wrong_camera_id",
        "cosine_similarity", "positive_gallery_images",
    ]
    with hard_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=hard_fields)
        writer.writeheader()
        writer.writerows(hard_negatives[:args.max_hard_negatives])

    identity_rows = []
    for vid in identity_values:
        stats = per_identity[vid]
        count = stats["queries"]
        identity_rows.append({
            "vehicle_id": vid,
            "valid_queries": count,
            "rank1_accuracy": stats["rank1_hits"] / count if count else "",
            "mAP": stats["ap_sum"] / count if count else "",
        })
    identity_rows.sort(key=lambda row: (
        row["rank1_accuracy"] if row["rank1_accuracy"] != "" else -1,
        row["mAP"] if row["mAP"] != "" else -1
    ))
    identity_path = args.output_dir / "identity_metrics.csv"
    with identity_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicle_id", "valid_queries", "rank1_accuracy", "mAP"])
        writer.writeheader()
        writer.writerows(identity_rows)

    summary = {
        "embedding_file": str(args.embeddings),
        "images": n,
        "identities": len(identity_values),
        "cameras": camera_values,
        "full_cross_camera_valid_queries": query_count,
        "full_cross_camera_rank1_accuracy": rank1_total / query_count if query_count else None,
        "full_cross_camera_mAP": ap_total / query_count if query_count else None,
        "full_cross_camera_top1_errors": len(hard_negatives),
        "hard_negative_rows_written": min(len(hard_negatives), args.max_hard_negatives),
        "ordered_camera_pairs": total_pairs,
        "protocol": (
            "For full-gallery metrics, all images are queries and same-camera gallery "
            "images are excluded. For camera-pair metrics, queries come from one camera "
            "and gallery images come only from one different camera."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "outputs": {
            "camera_pair_metrics": str(pair_path),
            "hard_negative_examples": str(hard_path),
            "identity_metrics": str(identity_path),
        },
    }
    summary_path = args.output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== EMBEDDING AUDIT COMPLETE ===")
    print(f"Full-gallery Rank-1: {summary['full_cross_camera_rank1_accuracy']}")
    print(f"Full-gallery mAP: {summary['full_cross_camera_mAP']}")
    print(f"Full-gallery top-1 errors: {len(hard_negatives)}")
    print(f"Camera pairs evaluated: {total_pairs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")


if __name__ == "__main__":
    main()
