from __future__ import annotations

from uuid import UUID, uuid4

from src.tvs.db.connection import get_connection


class TrafficRepository:
    def __init__(self):
        self.connection = get_connection()

    def close(self):
        self.connection.close()

    def register_camera(self, camera_id: str):
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO camera_metadata (camera_id)
                    VALUES (%s)
                    ON CONFLICT (camera_id) DO NOTHING
                    """,
                    (camera_id,),
                )

    def create_processing_run(self, camera_id: str, source_fps: float) -> UUID:
        run_id = uuid4()
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO processing_runs (run_id, camera_id, source_fps)
                    VALUES (%s, %s, %s)
                    """,
                    (run_id, camera_id, source_fps),
                )
        return run_id

    @staticmethod
    def _values_sql(row_count: int, column_count: int) -> str:
        row = "(" + ", ".join(["%s"] * column_count) + ")"
        return ", ".join([row] * row_count)

    def save_observations_batch(self, run_id: UUID, frame_observations) -> int:
        """
        Persist observations from multiple frames in one transaction.

        Track summaries are aggregated once per published ID per batch.
        Each frame-level observation is retained as a track_points row.
        """
        aggregates = {}
        point_rows = []

        for frame, observations in frame_observations:
            for observation in observations:
                published_id = observation.published_track_id
                if published_id is None:
                    continue

                x, y, width, height = observation.bbox
                confidence = observation.confidence

                aggregate = aggregates.get(published_id)
                if aggregate is None:
                    aggregates[published_id] = {
                        "class_id": observation.class_id,
                        "source_label": observation.source_label,
                        "vehicle_type": observation.vehicle_type,
                        "first_frame_id": frame.frame_id,
                        "last_frame_id": frame.frame_id,
                        "first_elapsed_seconds": frame.elapsed_seconds,
                        "last_elapsed_seconds": frame.elapsed_seconds,
                        "observation_count": 1,
                        "max_confidence": confidence,
                    }
                else:
                    aggregate["last_frame_id"] = frame.frame_id
                    aggregate["last_elapsed_seconds"] = frame.elapsed_seconds
                    aggregate["observation_count"] += 1
                    aggregate["max_confidence"] = max(
                        aggregate["max_confidence"], confidence
                    )

                point_rows.append(
                    (
                        published_id,
                        frame.frame_id,
                        frame.elapsed_seconds,
                        x,
                        y,
                        width,
                        height,
                        observation.class_id,
                        observation.source_label,
                        observation.vehicle_type,
                        confidence,
                        observation.detection_index,
                    )
                )

        if not point_rows:
            return 0

        track_rows = [
            (
                run_id,
                published_id,
                values["class_id"],
                values["source_label"],
                values["vehicle_type"],
                values["first_frame_id"],
                values["last_frame_id"],
                values["first_elapsed_seconds"],
                values["last_elapsed_seconds"],
                values["observation_count"],
                values["max_confidence"],
            )
            for published_id, values in aggregates.items()
        ]

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                track_ids = {}

                # One row per (run_id, published_track_id) in each UPSERT.
                # 500 rows x 11 parameters stays below PostgreSQL's limit.
                for offset in range(0, len(track_rows), 500):
                    chunk = track_rows[offset:offset + 500]
                    sql = """
                        INSERT INTO vehicle_tracks (
                            run_id,
                            published_track_id,
                            class_id,
                            source_label,
                            vehicle_type,
                            first_frame_id,
                            last_frame_id,
                            first_elapsed_seconds,
                            last_elapsed_seconds,
                            observation_count,
                            max_confidence
                        )
                        VALUES
                    """ + self._values_sql(len(chunk), 11) + """
                        ON CONFLICT (run_id, published_track_id)
                        DO UPDATE SET
                            last_frame_id = EXCLUDED.last_frame_id,
                            last_elapsed_seconds = EXCLUDED.last_elapsed_seconds,
                            observation_count =
                                vehicle_tracks.observation_count
                                + EXCLUDED.observation_count,
                            max_confidence = GREATEST(
                                vehicle_tracks.max_confidence,
                                EXCLUDED.max_confidence
                            )
                        RETURNING published_track_id, track_id
                    """
                    params = [value for row in chunk for value in row]
                    cursor.execute(sql, params)
                    track_ids.update(dict(cursor.fetchall()))

                resolved_points = [
                    (
                        track_ids[published_id],
                        frame_id,
                        elapsed_seconds,
                        x,
                        y,
                        width,
                        height,
                        class_id,
                        source_label,
                        vehicle_type,
                        confidence,
                        detection_index,
                    )
                    for (
                        published_id,
                        frame_id,
                        elapsed_seconds,
                        x,
                        y,
                        width,
                        height,
                        class_id,
                        source_label,
                        vehicle_type,
                        confidence,
                        detection_index,
                    ) in point_rows
                ]

                # 1000 rows x 12 parameters; preserve each individual point.
                for offset in range(0, len(resolved_points), 1000):
                    chunk = resolved_points[offset:offset + 1000]
                    sql = """
                        INSERT INTO track_points (
                            track_id,
                            frame_id,
                            elapsed_seconds,
                            bbox_x,
                            bbox_y,
                            bbox_width,
                            bbox_height,
                            class_id,
                            source_label,
                            vehicle_type,
                            confidence,
                            detection_index
                        )
                        VALUES
                    """ + self._values_sql(len(chunk), 12)
                    params = [value for row in chunk for value in row]
                    cursor.execute(sql, params)

        return len(point_rows)

    def save_observations(self, run_id: UUID, frame, observations) -> int:
        # Retain the original method for callers outside the upload pipeline.
        return self.save_observations_batch(run_id, [(frame, observations)])

    def complete_processing_run(
        self,
        run_id: UUID,
        frames_processed: int,
        total_detections: int,
        published_observations: int,
        unique_tracks: int,
    ):
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE processing_runs
                    SET
                        status = 'completed',
                        completed_at = CURRENT_TIMESTAMP,
                        frames_processed = %s,
                        total_detections = %s,
                        published_observations = %s,
                        unique_tracks = %s
                    WHERE run_id = %s
                    """,
                    (
                        frames_processed,
                        total_detections,
                        published_observations,
                        unique_tracks,
                        run_id,
                    ),
                )

    def fail_processing_run(self, run_id: UUID):
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE processing_runs
                    SET status = 'failed', completed_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
