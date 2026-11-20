from __future__ import annotations

"""Small integration hooks for adding ReID to the existing video processor."""

from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from src.tvs.db.track_id_lookup import get_track_id_map
from src.tvs.reid.config import ReIDConfig
from src.tvs.reid.embedder import VehicleEmbedder
from src.tvs.reid.model import VehicleReIDModel
from src.tvs.reid.run_persistence import persist_finalized_track_descriptors
from src.tvs.reid.track_descriptor_collector import (
    CollectorConfig,
    TrackDescriptorCollector,
)


def build_reid_collector(project_root: Path) -> TrackDescriptorCollector:
    """Load config/reid.yaml and initialize the pretrained CPU ONNX embedder."""
    config_path = Path(project_root) / "config" / "reid.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"ReID configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    model_settings = raw.get("model", {})
    runtime_settings = raw.get("runtime", {})
    weights = model_settings.get("weights")
    if not weights:
        raise ValueError("config/reid.yaml is missing model.weights")

    model_path = Path(weights)
    if not model_path.is_absolute():
        model_path = Path(project_root) / model_path

    reid_config = ReIDConfig(
        model_path=model_path,
        input_width=int(model_settings.get("input_width", 208)),
        input_height=int(model_settings.get("input_height", 208)),
        intra_op_num_threads=int(runtime_settings.get("intra_op_num_threads", 2)),
        inter_op_num_threads=int(runtime_settings.get("inter_op_num_threads", 1)),
        normalize_embeddings=bool(runtime_settings.get("normalize_embeddings", True)),
    )
    model = VehicleReIDModel(reid_config)
    embedder = VehicleEmbedder(model)
    return TrackDescriptorCollector(embedder, CollectorConfig())


def persist_completed_run_reid(
    repository: Any,
    run_id: UUID,
    collector: TrackDescriptorCollector,
    *,
    candidate_limit: int = 5,
) -> dict[str, Any]:
    """Finalize descriptors and persist them using the run's database IDs."""
    descriptors = collector.finalize_all()
    if not descriptors:
        return {
            "persisted_descriptors": 0,
            "created_new_identities": 0,
            "candidate_count": 0,
            "skipped_missing_track": 0,
            "candidates": [],
        }

    track_id_map = get_track_id_map(repository.connection, run_id)

    # get_track_id_map performs a SELECT with autocommit disabled. Commit that
    # read-only transaction so the persistence function can own a top-level
    # transaction rather than creating a nested savepoint.
    repository.connection.commit()

    return persist_finalized_track_descriptors(
        connection=repository.connection,
        run_id=run_id,
        track_id_map=track_id_map,
        descriptors=descriptors,
        candidate_limit=candidate_limit,
    )
