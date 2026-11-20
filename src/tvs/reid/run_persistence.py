from __future__ import annotations

"""Persist finalized per-track ReID descriptors in one committed transaction."""

from typing import Any
from uuid import UUID

import numpy as np


def _pack_descriptor(embedding: np.ndarray) -> bytes:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (512,):
        raise ValueError(f"Expected a 512-D descriptor, received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Descriptor contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Descriptor has zero norm")
    return (vector / norm).astype("<f4", copy=False).tobytes()


def persist_finalized_track_descriptors(
    connection: Any,
    run_id: UUID,
    track_id_map: dict[int, int],
    descriptors: dict[int, np.ndarray],
    *,
    candidate_limit: int = 5,
    candidate_finder=None,
) -> dict[str, Any]:
    """Persist descriptors and identity records atomically.

    All SELECTs and INSERTs are inside one top-level transaction. This avoids
    psycopg treating transaction() blocks as savepoints after an earlier SELECT
    has already opened an implicit transaction.
    """
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1")

    if candidate_finder is None:
        from .postgres_store import find_cross_video_candidates
        active_candidate_finder = find_cross_video_candidates
    else:
        active_candidate_finder = candidate_finder

    persisted = 0
    created_new_identities = 0
    candidate_records: list[dict[str, Any]] = []
    skipped_missing_track = 0

    with connection.transaction():
        for raw_published_id, descriptor in descriptors.items():
            published_id = int(raw_published_id)
            track_id = track_id_map.get(published_id)
            if track_id is None:
                skipped_missing_track += 1
                continue

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT vehicle_type
                    FROM vehicle_tracks
                    WHERE track_id = %s AND run_id = %s
                    """,
                    (int(track_id), run_id),
                )
                row = cursor.fetchone()

            if row is None or not row[0]:
                skipped_missing_track += 1
                continue

            vehicle_type = str(row[0])
            blob = _pack_descriptor(descriptor)
            candidates = active_candidate_finder(
                connection,
                descriptor,
                vehicle_type,
                exclude_run_id=run_id,
                limit=candidate_limit,
            )
            best = candidates[0] if candidates else None

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO vehicle_reid_descriptors
                        (track_id, vehicle_type, embedding, embedding_dimension)
                    VALUES (%s, %s, %s, 512)
                    ON CONFLICT (track_id) DO NOTHING
                    RETURNING descriptor_id
                    """,
                    (int(track_id), vehicle_type, blob),
                )
                inserted = cursor.fetchone()
                if inserted is None:
                    continue

                if best is None:
                    cursor.execute(
                        """
                        INSERT INTO persistent_vehicles DEFAULT VALUES
                        RETURNING persistent_vehicle_id
                        """
                    )
                    persistent_id = int(cursor.fetchone()[0])
                    cursor.execute(
                        """
                        INSERT INTO vehicle_track_identities
                            (track_id, persistent_vehicle_id, match_status)
                        VALUES (%s, %s, 'new')
                        """,
                        (int(track_id), persistent_id),
                    )
                    created_new_identities += 1
                else:
                    cursor.execute(
                        """
                        INSERT INTO vehicle_track_identities
                            (track_id, persistent_vehicle_id, match_status,
                             candidate_vehicle_id, candidate_similarity)
                        VALUES (%s, NULL, 'candidate', %s, %s)
                        """,
                        (
                            int(track_id),
                            int(best["persistent_vehicle_id"]),
                            float(best["similarity"]),
                        ),
                    )
                    candidate_records.append(
                        {
                            "published_track_id": published_id,
                            "track_id": int(track_id),
                            "candidate_vehicle_id": int(best["persistent_vehicle_id"]),
                            "similarity": float(best["similarity"]),
                            "source_run_id": best["source_run_id"],
                            "source_published_track_id": int(
                                best["source_published_track_id"]
                            ),
                        }
                    )

                persisted += 1

    # The top-level transaction has committed before these counts are returned.
    return {
        "persisted_descriptors": persisted,
        "created_new_identities": created_new_identities,
        "candidate_count": len(candidate_records),
        "skipped_missing_track": skipped_missing_track,
        "candidates": candidate_records,
    }
