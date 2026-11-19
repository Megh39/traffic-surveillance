-- Additive Re-ID persistence migration.
-- Does not alter or replace existing Docker/Compose configuration or tables.

CREATE TABLE IF NOT EXISTS persistent_vehicles (
    persistent_vehicle_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehicle_reid_descriptors (
    descriptor_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL UNIQUE
        REFERENCES vehicle_tracks(track_id) ON DELETE CASCADE,
    vehicle_type TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_dimension SMALLINT NOT NULL DEFAULT 512,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (embedding_dimension = 512),
    CHECK (octet_length(embedding) = 2048)
);

CREATE TABLE IF NOT EXISTS vehicle_track_identities (
    track_id BIGINT PRIMARY KEY
        REFERENCES vehicle_tracks(track_id) ON DELETE CASCADE,
    persistent_vehicle_id BIGINT
        REFERENCES persistent_vehicles(persistent_vehicle_id),
    match_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (match_status IN ('pending', 'new', 'candidate', 'confirmed')),
    candidate_vehicle_id BIGINT
        REFERENCES persistent_vehicles(persistent_vehicle_id),
    candidate_similarity REAL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        candidate_similarity IS NULL OR
        candidate_similarity BETWEEN -1 AND 1
    )
);

CREATE INDEX IF NOT EXISTS idx_vehicle_reid_descriptors_type
    ON vehicle_reid_descriptors(vehicle_type);

CREATE INDEX IF NOT EXISTS idx_vehicle_track_identities_persistent
    ON vehicle_track_identities(persistent_vehicle_id);
