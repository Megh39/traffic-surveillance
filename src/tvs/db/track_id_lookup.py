from __future__ import annotations

from typing import Any
from uuid import UUID


def get_track_id_map(connection: Any, run_id: UUID) -> dict[int, int]:
    """Return {published_track_id: database track_id} for one processing run."""
    sql = """
        SELECT published_track_id, track_id
        FROM vehicle_tracks
        WHERE run_id = %s
        ORDER BY published_track_id
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, (run_id,))
        rows = cursor.fetchall()
    return {
        int(published_track_id): int(track_id)
        for published_track_id, track_id in rows
    }
