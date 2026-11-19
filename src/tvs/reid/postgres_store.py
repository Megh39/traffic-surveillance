from __future__ import annotations

"""PostgreSQL persistence helpers for cross-video vehicle Re-ID.

This module stores one normalized 512-D descriptor per completed vehicle track.
It deliberately records candidate matches for review; it does not automatically
merge identities using an uncalibrated similarity threshold.
"""

import numpy as np
import psycopg

from .embedder import cosine_similarity


def _pack_descriptor(embedding: np.ndarray) -> bytes:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (512,):
        raise ValueError(f"Expected a 512-D descriptor, received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Descriptor contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Descriptor has zero norm")
    vector = vector / norm
    return vector.astype("<f4", copy=False).tobytes()


def _unpack_descriptor(blob: bytes) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.shape != (512,):
        raise ValueError("Stored descriptor is not 512-D")
    return vector.copy()


def find_cross_video_candidates(
    connection: psycopg.Connection,
    embedding: np.ndarray,
    vehicle_type: str,
    exclude_run_id=None,
    limit: int = 5,
) -> list[dict]:
    """Return top appearance candidates from other runs; no identity is merged."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query = _pack_descriptor(embedding)
    query_vector = _unpack_descriptor(query)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                p.persistent_vehicle_id,
                d.embedding,
                t.run_id,
                t.published_track_id
            FROM vehicle_reid_descriptors AS d
            JOIN vehicle_tracks AS t ON t.track_id = d.track_id
            JOIN vehicle_track_identities AS i ON i.track_id = d.track_id
            JOIN persistent_vehicles AS p
              ON p.persistent_vehicle_id = i.persistent_vehicle_id
            WHERE d.vehicle_type = %s
              AND i.match_status IN ('confirmed', 'new')
              AND (%s IS NULL OR t.run_id <> %s)
            """,
            (vehicle_type, exclude_run_id, exclude_run_id),
        )
        rows = cursor.fetchall()

    candidates = []
    for persistent_id, blob, run_id, published_track_id in rows:
        score = cosine_similarity(query_vector, _unpack_descriptor(blob))
        candidates.append({
            "persistent_vehicle_id": persistent_id,
            "similarity": score,
            "source_run_id": str(run_id),
            "source_published_track_id": published_track_id,
        })
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    return candidates[:limit]


def store_track_as_new_identity(
    connection: psycopg.Connection,
    track_id: int,
    vehicle_type: str,
    embedding: np.ndarray,
) -> int:
    """Create a new persistent ID and save its descriptor.

    This is intentionally conservative: it never auto-links to a candidate.
    Call find_cross_video_candidates first; a human/calibrated policy must
    confirm a match before reusing an existing persistent vehicle ID.
    """
    blob = _pack_descriptor(embedding)
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO persistent_vehicles DEFAULT VALUES "
                "RETURNING persistent_vehicle_id"
            )
            persistent_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO vehicle_reid_descriptors
                    (track_id, vehicle_type, embedding, embedding_dimension)
                VALUES (%s, %s, %s, 512)
                """,
                (track_id, vehicle_type, blob),
            )
            cursor.execute(
                """
                INSERT INTO vehicle_track_identities
                    (track_id, persistent_vehicle_id, match_status)
                VALUES (%s, %s, 'new')
                """,
                (track_id, persistent_id),
            )
    return persistent_id


def confirm_track_identity(
    connection: psycopg.Connection,
    track_id: int,
    persistent_vehicle_id: int,
) -> None:
    """Link a track to an existing persistent ID after an explicit decision."""
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO vehicle_track_identities
                    (track_id, persistent_vehicle_id, match_status)
                VALUES (%s, %s, 'confirmed')
                ON CONFLICT (track_id) DO UPDATE SET
                    persistent_vehicle_id = EXCLUDED.persistent_vehicle_id,
                    match_status = 'confirmed',
                    candidate_vehicle_id = NULL,
                    candidate_similarity = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (track_id, persistent_vehicle_id),
            )
            cursor.execute(
                """
                UPDATE persistent_vehicles
                SET updated_at = CURRENT_TIMESTAMP
                WHERE persistent_vehicle_id = %s
                """,
                (persistent_vehicle_id,),
            )
