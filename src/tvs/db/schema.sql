
CREATE TABLE IF NOT EXISTS camera_metadata (
    camera_id TEXT PRIMARY KEY,
    name TEXT,
    location_description TEXT,
    gps_coordinates JSONB,
    coverage_zone TEXT,
    rtsp_url TEXT,
    active_status BOOLEAN NOT NULL DEFAULT TRUE,
    ptz_support BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id UUID PRIMARY KEY,
    camera_id TEXT NOT NULL
        REFERENCES camera_metadata(camera_id),

    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,

    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),

    source_fps DOUBLE PRECISION NOT NULL
        CHECK (source_fps > 0),

    frames_processed BIGINT NOT NULL DEFAULT 0,
    total_detections BIGINT NOT NULL DEFAULT 0,
    published_observations BIGINT NOT NULL DEFAULT 0,
    unique_tracks BIGINT NOT NULL DEFAULT 0,

    CHECK (frames_processed >= 0),
    CHECK (total_detections >= 0),
    CHECK (published_observations >= 0),
    CHECK (unique_tracks >= 0)
);

CREATE TABLE IF NOT EXISTS vehicle_tracks (
    track_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id UUID NOT NULL
        REFERENCES processing_runs(run_id) ON DELETE CASCADE,

    published_track_id INTEGER NOT NULL,

    class_id INTEGER NOT NULL,
    source_label TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,

    first_frame_id BIGINT NOT NULL,
    last_frame_id BIGINT NOT NULL,

    first_elapsed_seconds DOUBLE PRECISION NOT NULL,
    last_elapsed_seconds DOUBLE PRECISION NOT NULL,

    observation_count BIGINT NOT NULL DEFAULT 0,
    max_confidence DOUBLE PRECISION,

    UNIQUE (run_id, published_track_id),

    CHECK (last_frame_id >= first_frame_id),
    CHECK (last_elapsed_seconds >= first_elapsed_seconds),
    CHECK (observation_count >= 0),
    CHECK (
        max_confidence IS NULL OR
        max_confidence BETWEEN 0 AND 1
    )
);

CREATE TABLE IF NOT EXISTS track_points (
    track_point_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    track_id BIGINT NOT NULL
        REFERENCES vehicle_tracks(track_id) ON DELETE CASCADE,

    frame_id BIGINT NOT NULL,
    elapsed_seconds DOUBLE PRECISION NOT NULL,

    bbox_x DOUBLE PRECISION NOT NULL,
    bbox_y DOUBLE PRECISION NOT NULL,
    bbox_width DOUBLE PRECISION NOT NULL,
    bbox_height DOUBLE PRECISION NOT NULL,

    class_id INTEGER NOT NULL,
    source_label TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,

    confidence DOUBLE PRECISION NOT NULL,
    detection_index INTEGER NOT NULL,

    UNIQUE (track_id, frame_id),

    CHECK (bbox_width >= 0),
    CHECK (bbox_height >= 0),
    CHECK (elapsed_seconds >= 0),
    CHECK (confidence BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_processing_runs_camera
    ON processing_runs(camera_id);

CREATE INDEX IF NOT EXISTS idx_vehicle_tracks_run
    ON vehicle_tracks(run_id);

CREATE INDEX IF NOT EXISTS idx_track_points_track
    ON track_points(track_id);

CREATE INDEX IF NOT EXISTS idx_track_points_frame
    ON track_points(frame_id);
