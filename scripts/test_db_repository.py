
from types import SimpleNamespace

from src.tvs.db.repository import TrafficRepository


def main():
    repository = TrafficRepository()
    run_id = None

    try:
        repository.register_camera("mvi20011")
        run_id = repository.create_processing_run(
            camera_id="mvi20011",
            source_fps=30.0,
        )

        frame = SimpleNamespace(
            frame_id=1,
            elapsed_seconds=1.0,
        )

        observation = SimpleNamespace(
            frame_id=1,
            published_track_id=999999,
            bbox=(100.0, 120.0, 80.0, 60.0),
            class_id=2,
            source_label="car",
            vehicle_type="car",
            confidence=0.95,
            detection_index=0,
        )

        saved = repository.save_observations(
            run_id,
            frame,
            [observation],
        )

        repository.complete_processing_run(
            run_id=run_id,
            frames_processed=1,
            total_detections=1,
            published_observations=saved,
            unique_tracks=1,
        )

        with repository.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    pr.status,
                    vt.observation_count,
                    COUNT(tp.track_point_id)
                FROM processing_runs pr
                JOIN vehicle_tracks vt
                    ON vt.run_id = pr.run_id
                JOIN track_points tp
                    ON tp.track_id = vt.track_id
                WHERE pr.run_id = %s
                GROUP BY pr.status, vt.observation_count
                """,
                (run_id,),
            )

            result = cursor.fetchone()

        assert result is not None
        assert result[0] == "completed"
        assert result[1] == 1
        assert result[2] == 1

        print("DATABASE REPOSITORY TEST PASSED")
        print(f"Processing run: {run_id}")
        print(f"Saved observations: {saved}")
        print(f"Database result: {result}")

    finally:
        if run_id is not None:
            with repository.connection.transaction():
                with repository.connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM processing_runs WHERE run_id = %s",
                        (run_id,),
                    )

        repository.close()


if __name__ == "__main__":
    main()
