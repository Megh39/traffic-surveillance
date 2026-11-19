from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.tvs.reid.config import ReIDConfig
from src.tvs.reid.embedder import VehicleEmbedder, cosine_similarity
from src.tvs.reid.model import VehicleReIDModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPU smoke test for pretrained vehicle Re-ID."
    )
    parser.add_argument("image_a", type=Path)
    parser.add_argument("image_b", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/reid/osnet_ain_x1_0_vehicle_reid.onnx"),
    )
    args = parser.parse_args()

    config = ReIDConfig(model_path=args.model)
    model = VehicleReIDModel(config)
    embedder = VehicleEmbedder(model)

    image_a = cv2.imread(str(args.image_a))
    image_b = cv2.imread(str(args.image_b))
    if image_a is None:
        raise FileNotFoundError(f"Could not read image: {args.image_a}")
    if image_b is None:
        raise FileNotFoundError(f"Could not read image: {args.image_b}")

    embedding_a = embedder.from_crop(image_a)
    embedding_b = embedder.from_crop(image_b)
    score = cosine_similarity(embedding_a, embedding_b)

    print(f"Embedding A shape: {embedding_a.shape}")
    print(f"Embedding B shape: {embedding_b.shape}")
    print(f"Embedding A norm:  {np.linalg.norm(embedding_a):.6f}")
    print(f"Embedding B norm:  {np.linalg.norm(embedding_b):.6f}")
    print(f"Cosine similarity: {score:.6f}")
    print(
        "Note: this score is not a calibrated probability or a final "
        "same-vehicle decision."
    )


if __name__ == "__main__":
    main()
