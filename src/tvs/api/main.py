from typing import Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from psycopg.rows import dict_row

from src.tvs.db.analytics import TrafficAnalytics


app = FastAPI(
    title="Traffic Surveillance API",
    description="Analytics API for the traffic video surveillance system.",
    version="1.0.0",
)


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

        return jsonable_encoder({
            "run": run,
            "summary": metrics,
        })
    finally:
        analytics.close()


@app.get("/analytics/vehicle-types")
def vehicle_types(run_id: Optional[UUID] = None):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        results = analytics.get_vehicle_types(run["run_id"])

        return jsonable_encoder({
            "run_id": run["run_id"],
            "data": results,
        })
    finally:
        analytics.close()


@app.get("/analytics/observations-over-time")
def observations_over_time(run_id: Optional[UUID] = None):
    analytics = TrafficAnalytics()
    try:
        run = _resolve_run(analytics, run_id)
        results = analytics.get_observations_over_time(run["run_id"])

        return jsonable_encoder({
            "run_id": run["run_id"],
            "interval_seconds": 10,
            "data": results,
        })
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

        return jsonable_encoder({
            "run_id": run["run_id"],
            "count": len(results),
            "data": results,
        })
    finally:
        analytics.close()


@app.get("/analytics/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)):
    analytics = TrafficAnalytics()
    try:
        results = analytics.get_runs(limit)

        return jsonable_encoder({
            "count": len(results),
            "data": results,
        })
    finally:
        analytics.close()
