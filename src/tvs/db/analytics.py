from psycopg.rows import dict_row

from src.tvs.db.connection import get_connection


class TrafficAnalytics:
    """Read-only analytics queries for the traffic surveillance database."""

    def __init__(self):
        self.connection = get_connection()

    def close(self):
        self.connection.close()

    def get_latest_completed_run(self):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT run_id, camera_id, started_at, completed_at,
                       source_fps, frames_processed, total_detections,
                       published_observations, unique_tracks
                FROM processing_runs
                WHERE status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
                """
            )
            return cursor.fetchone()

    def get_summary(self, run_id):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT vt.track_id) AS total_tracks,
                    COUNT(tp.track_point_id) AS total_observations,
                    COUNT(DISTINCT tp.frame_id) AS frames_with_observations,
                    COALESCE(MAX(pr.frames_processed), 0) AS frames_processed,
                    COALESCE(MAX(pr.total_detections), 0) AS total_detections,
                    COALESCE(MAX(pr.published_observations), 0)
                        AS published_observations,
                    COALESCE(MAX(pr.unique_tracks), 0) AS unique_tracks
                FROM processing_runs pr
                LEFT JOIN vehicle_tracks vt ON vt.run_id = pr.run_id
                LEFT JOIN track_points tp ON tp.track_id = vt.track_id
                WHERE pr.run_id = %s
                GROUP BY pr.run_id
                """,
                (run_id,),
            )
            return cursor.fetchone()

    def get_vehicle_types(self, run_id):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT vehicle_type, COUNT(*) AS track_count
                FROM vehicle_tracks
                WHERE run_id = %s
                GROUP BY vehicle_type
                ORDER BY track_count DESC, vehicle_type
                """,
                (run_id,),
            )
            return cursor.fetchall()

    def get_observations_over_time(self, run_id):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    FLOOR(tp.elapsed_seconds / 10) * 10
                        AS interval_start_seconds,
                    COUNT(*) AS observation_count
                FROM track_points tp
                JOIN vehicle_tracks vt ON vt.track_id = tp.track_id
                WHERE vt.run_id = %s
                GROUP BY interval_start_seconds
                ORDER BY interval_start_seconds
                """,
                (run_id,),
            )
            return cursor.fetchall()

    def get_tracks(self, run_id, limit=100):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    published_track_id,
                    vehicle_type,
                    observation_count,
                    first_frame_id,
                    last_frame_id,
                    first_elapsed_seconds,
                    last_elapsed_seconds,
                    ROUND(
                        (last_elapsed_seconds - first_elapsed_seconds)::numeric,
                        2
                    ) AS duration_seconds,
                    ROUND(max_confidence::numeric, 3) AS max_confidence
                FROM vehicle_tracks
                WHERE run_id = %s
                ORDER BY observation_count DESC, published_track_id
                LIMIT %s
                """,
                (run_id, limit),
            )
            return cursor.fetchall()

    def get_runs(self, limit=20):
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    run_id,
                    camera_id,
                    status,
                    started_at,
                    completed_at,
                    frames_processed,
                    total_detections,
                    published_observations,
                    unique_tracks
                FROM processing_runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()
