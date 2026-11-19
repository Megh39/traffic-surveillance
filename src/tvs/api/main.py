from typing import Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from psycopg.rows import dict_row

from src.tvs.db.analytics import TrafficAnalytics

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from src.tvs.ingestion.video_file import VideoFileSource
from src.tvs.pipeline.video_processor import process_video

app = FastAPI(
    title="Traffic Surveillance API",
    description="Analytics API for the traffic video surveillance system.",
    version="1.0.0",
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
PROCESSED_VIDEO_DIR = PROJECT_ROOT / "outputs" / "uploaded_videos"

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}

logger = logging.getLogger(__name__)


def _resolve_run(analytics, run_id: Optional[UUID] = None):
    """Return the requested completed run or the latest completed run."""

    with analytics.connection.cursor(row_factory=dict_row) as cursor:
        if run_id is not None:
            cursor.execute(
                """
                SELECT run_id, camera_id, started_at, completed_at,
                       source_fps, frames_processed, total_detections,
                       published_observations, unique_tracks
                FROM processing_runs
                WHERE run_id = %s AND status = 'completed'
                """,
                (run_id,),
            )
        else:
            cursor.execute("""
                SELECT run_id, camera_id, started_at, completed_at,
                       source_fps, frames_processed, total_detections,
                       published_observations, unique_tracks
                FROM processing_runs
                WHERE status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
                """)

        run = cursor.fetchone()

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="No matching completed processing run was found.",
        )

    return run


@app.get("/health")
def health():
    analytics = None
    try:
        analytics = TrafficAnalytics()
        with analytics.connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection is unavailable.",
        ) from exc
    finally:
        if analytics is not None:
            analytics.close()


@app.get("/analytics/summary")
def summary(run_id: Optional[UUID] = None):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        metrics = analytics.get_summary(run["run_id"])

        return jsonable_encoder(
            {
                "run": run,
                "summary": metrics,
            }
        )
    finally:
        analytics.close()


@app.get("/analytics/vehicle-types")
def vehicle_types(run_id: Optional[UUID] = None):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        results = analytics.get_vehicle_types(run["run_id"])

        return jsonable_encoder(
            {
                "run_id": run["run_id"],
                "data": results,
            }
        )
    finally:
        analytics.close()


@app.get("/analytics/observations-over-time")
def observations_over_time(run_id: Optional[UUID] = None):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        results = analytics.get_observations_over_time(run["run_id"])

        return jsonable_encoder(
            {
                "run_id": run["run_id"],
                "interval_seconds": 10,
                "data": results,
            }
        )
    finally:
        analytics.close()


@app.get("/analytics/tracks")
def tracks(
    run_id: Optional[UUID] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        results = analytics.get_tracks(run["run_id"], limit)

        return jsonable_encoder(
            {
                "run_id": run["run_id"],
                "count": len(results),
                "data": results,
            }
        )
    finally:
        analytics.close()


@app.get("/analytics/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)):
    analytics = TrafficAnalytics()
    try:
        results = analytics.get_runs(limit)

        return jsonable_encoder(
            {
                "count": len(results),
                "data": results,
            }
        )
    finally:
        analytics.close()


@app.post("/videos/process")
def upload_and_process_video(file: UploadFile = File(...)):
    """
    Upload a video, process it through the detection and tracking
    pipeline, and persist its observations in PostgreSQL.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A video filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    upload_id = uuid4().hex
    camera_id = f"upload_{upload_id[:12]}"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    input_path = UPLOAD_DIR / f"{upload_id}{extension}"
    output_path = PROCESSED_VIDEO_DIR / f"{upload_id}.mp4"

    try:
        total_size = 0

        with input_path.open("wb") as destination:
            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 500 MB upload limit.",
                    )

                destination.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        source = VideoFileSource(video_path=input_path)

        result = process_video(
            source=source,
            output_path=output_path,
            camera_id=camera_id,
        )

        result.pop("output_path", None)

        return {
            "message": "Video processed successfully.",
            "upload_id": upload_id,
            "output_url": f"/videos/{upload_id}/output",
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Video processing failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {exc}",
        ) from exc

    finally:
        if input_path.exists():
            input_path.unlink()


@app.get("/videos/{upload_id}/output")
def download_processed_video(upload_id: str):
    """Return the annotated video generated from an uploaded video."""

    try:
        validated_id = UUID(upload_id).hex
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid upload ID.",
        )

    output_path = PROCESSED_VIDEO_DIR / f"{validated_id}.mp4"

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found.",
        )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{validated_id}.mp4",
        content_disposition_type="inline",
    )
