
from uuid import UUID, uuid4

from src.tvs.db.connection import get_connection


class TrafficRepository:
    """Persistence operations for traffic surveillance processing."""

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

    def create_processing_run(
        self,
        camera_id: str,
        source_fps: float,
    ) -> UUID:
        run_id = uuid4()

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO processing_runs (
                        run_id, camera_id, source_fps
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (run_id, camera_id, source_fps),
                )

        return run_id

    def save_observations(self, run_id: UUID, frame, observations):
        """Save a frame's observations in one transaction."""

        if not observations:
            return 0

        saved_count = 0

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                for observation in observations:
                    published_id = observation.published_track_id

                    if published_id is None:
                        continue

                    x, y, width, height = observation.bbox

                    cursor.execute(
                        """
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
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, 1, %s
                        )
                        ON CONFLICT (run_id, published_track_id)
                        DO UPDATE SET
                            last_frame_id = EXCLUDED.last_frame_id,
                            last_elapsed_seconds = EXCLUDED.last_elapsed_seconds,
                            observation_count =
                                vehicle_tracks.observation_count + 1,
                            max_confidence = GREATEST(
                                vehicle_tracks.max_confidence,
                                EXCLUDED.max_confidence
                            )
                        RETURNING track_id
                        """,
                        (
                            run_id,
                            published_id,
                            observation.class_id,
                            observation.source_label,
                            observation.vehicle_type,
                            observation.frame_id,
                            observation.frame_id,
                            frame.elapsed_seconds,
                            frame.elapsed_seconds,
                            observation.confidence,
                        ),
                    )

                    track_id = cursor.fetchone()[0]

                    cursor.execute(
                        """
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
                        VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            track_id,
                            observation.frame_id,
                            frame.elapsed_seconds,
                            x,
                            y,
                            width,
                            height,
                            observation.class_id,
                            observation.source_label,
                            observation.vehicle_type,
                            observation.confidence,
                            observation.detection_index,
                        ),
                    )

                    saved_count += 1

        return saved_count

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
                    SET
                        status = 'failed',
                        completed_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
