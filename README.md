# Traffic Video Surveillance and Vehicle Re-Identification

## 1. Project Overview

The Traffic Video Surveillance and Vehicle Re-Identification system processes traffic videos, detects vehicles, tracks them through frames, extracts visual descriptors, and stores processing and identity information in PostgreSQL.

The application has four main parts:

1. **Video-processing pipeline** — reads video frames and runs vehicle detection, tracking, annotation, and ReID descriptor extraction.
2. **YOLO11n and ByteTrack** — YOLO11n detects vehicles; ByteTrack associates detections across consecutive frames and produces application-published track IDs.
3. **Vehicle Re-Identification (ReID)** — a pretrained OSNet-AIN x1.0 model runs through ONNX Runtime on the CPU and generates 512-dimensional vehicle descriptors.
4. **FastAPI, PostgreSQL, and Streamlit** — FastAPI exposes processing and analytics endpoints, PostgreSQL stores results, and Streamlit displays the results.

The Python application runs from the Windows `.venv`. PostgreSQL runs separately in the existing Docker container named `traffic-postgres`.

## 2. Important ID Definitions

These IDs represent different things and must not be treated as interchangeable:

- **ByteTrack internal ID:** the track identifier used internally by the tracker.
- **Published Track ID:** the application-level track identifier assigned to a confirmed track within one processing run.
- **Persistent Vehicle ID:** an application-level identity intended to represent the same physical vehicle across different processing runs.

A published track is not automatically a persistent vehicle identity. ReID compares visual descriptors and can produce a candidate match. A candidate is not a confirmed identity.

## 3. ReID Model and Configuration

The project uses the pretrained Open Model Zoo `vehicle-reid-0001` model:

| Property                 | Value                                            |
| ------------------------ | ------------------------------------------------ |
| Architecture             | OSNet-AIN x1.0                                   |
| Model format             | ONNX                                             |
| Input size               | 208 × 208                                       |
| Descriptor dimension     | 512                                              |
| Runtime provider         | CPUExecutionProvider                             |
| Intra-op threads         | 2                                                |
| Inter-op threads         | 1                                                |
| Descriptor normalization | L2 normalization enabled                         |
| Model path               | `models/reid/osnet_ain_x1_0_vehicle_reid.onnx` |

The model is pretrained. No training or fine-tuning is performed as part of the current project workflow. CPU inference is used to avoid requiring additional GPU memory alongside YOLO.

The matching threshold in `config/reid.yaml` is deliberately unset (`null`). A cosine similarity is a comparison score, not a calibrated probability. It must not be interpreted as a percentage chance that two images show the same vehicle.

## 4. ReID Processing Workflow

For each published track, the descriptor collector samples suitable vehicle crops from frames. The current collector applies these default conditions:

- Detection confidence must be at least `0.50`.
- Bounding-box area must be at least `1,024` pixels.
- At least 3 valid descriptor samples are required for a track.
- The observation must have a published track ID.
- The descriptor must be finite, non-zero, and have 512 dimensions.

Valid samples are aggregated into one normalized descriptor for the track. At the end of a video run, the descriptor is stored in PostgreSQL and the persistence layer checks for possible matches against descriptors from earlier runs.

Current identity statuses:

| Status            | Meaning                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `new`           | A new persistent vehicle identity was created.                                                                       |
| `candidate`     | A possible match to an existing persistent vehicle was found. It is not confirmed.                                   |
| `confirmed`     | A candidate identity was explicitly confirmed.                                                                       |
| `pending`       | Identity processing is pending.                                                                                      |
| `not_processed` | The dashboard/API display label used when no identity record exists for a track. It is not a stored database status. |

The current implementation does not automatically merge candidate identities. This is intentional until matching thresholds and decisions are validated on suitable held-out footage.

## 5. Database and Docker

PostgreSQL runs in the existing Docker container:

- Container: `traffic-postgres`
- Image: `postgres:16`
- Database: `traffic_db`
- User: `traffic_app`
- Host port: `5432`
- Persistent Docker volume: `traffic_postgres_data`

The Python application connects using the settings in the project-root `.env` file. The application connects to `localhost:5432`; it does not start Docker itself.

**Do not remove the existing Docker volume.** Do not run `docker compose down -v` for normal shutdown or restart. The named volume stores PostgreSQL data across container restarts.

### Start/check PostgreSQL

Run these commands in PowerShell from the project directory:

```powershell
docker start traffic-postgres
docker ps
```

If the container is already running, Docker may report that it is already started. `docker ps` verifies that it is running.

### Open the PostgreSQL shell

```powershell
docker exec -it traffic-postgres psql -U traffic_app -d traffic_db
```

Inside `psql`, list tables:

```sql
\dt
```

Exit the PostgreSQL shell:

```sql
\q
```

### Apply the ReID migration (only if it has not already been applied)

From the project root in PowerShell:

```powershell
Get-Content .\migrations\001_add_reid_identity_tables.sql | docker exec -i traffic-postgres psql -U traffic_app -d traffic_db
```

The migration adds the ReID identity and descriptor tables. It is additive and does not require replacing the existing Compose configuration or PostgreSQL volume. If the tables already exist, inspect the output rather than deleting or recreating the database.

## 6. Running the Application on Windows

Use the existing project `.venv` and the same local commands used during development. Open separate PowerShell terminals for the API and dashboard.

### Terminal 1 — FastAPI backend

From the project root, with the project virtual environment active:

```powershell
uvicorn src.tvs.api.main:app --reload
```

API base address:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### Terminal 2 — Streamlit dashboard

From the project root, with the project virtual environment active:

```powershell
streamlit run dashboard/app.py
```

The dashboard normally opens at:

```text
http://localhost:8501
```

Keep the PostgreSQL container, FastAPI process, and Streamlit process running while using the application. Stop the API and dashboard with `Ctrl + C` in their respective terminals.

## 7. Upload and Process a Video

1. Open the Streamlit dashboard at `http://localhost:8501`.
2. Open the video-processing section.
3. Select a supported traffic video.
4. Start processing and wait for the API request to finish.
5. Review the processing statistics and annotated output video.
6. Open the **Vehicle Re-ID** tab and select the completed processing run.
7. Review descriptor counts, identity statuses, and candidate similarities.

The ReID tab reads persisted descriptor and identity records for the selected run. It distinguishes newly assigned persistent IDs from unconfirmed cross-run candidates.

## 8. Results Obtained So Far

### 8.1 VeRi cross-camera descriptor evaluation

The local evaluation used the existing ONNX model and the VeRi test annotations. All 11,579 annotated images were resolved and embedded.

| Metric                                     |                  Result |
| ------------------------------------------ | ----------------------: |
| XML records                                |                  11,579 |
| Resolved images                            |                  11,579 |
| Missing images                             |                       0 |
| Embedded images                            |                  11,579 |
| Vehicle identities                         |                     200 |
| Camera IDs                                 | 19 (`c001`–`c019`) |
| Identities represented in multiple cameras |                     200 |
| Valid queries                              |                  11,579 |
| Queries without cross-camera positives     |                       0 |
| Rank-1 accuracy                            |                  98.16% |
| mAP                                        |                  85.82% |
| Evaluation time reported by script         |          23.286 seconds |

**Evaluation protocol:** every annotated image was used as a query; images from the query camera were excluded from that query's gallery; gallery images with the same `vehicleID` from another camera were treated as positives.

**Limitations:** this is the explicitly implemented all-annotated-images cross-camera retrieval protocol. It is not necessarily identical to the official VeRi-776 query/gallery protocol. Model training-data provenance must be checked before treating the result as an independent benchmark. Retrieval similarity is not a calibrated probability and does not by itself justify automatic identity merging.

Evaluation outputs created locally:

- `data\veri_cross_camera_results.json`
- `data\veri_cross_camera_embeddings.npz`
- `data\veri_cross_camera_results_manifest.csv`

### 8.2 Video processing and ReID persistence run

A previously completed video run produced the following processing results:

| Metric                                    |        Result |
| ----------------------------------------- | ------------: |
| Frames processed                          |           456 |
| Total detections                          |        12,592 |
| Published observations                    |         8,866 |
| Unique published tracks                   |           110 |
| Descriptors persisted                     |            82 |
| Persistent identities created in that run |            82 |
| Candidate matches in that run             |             0 |
| Tracks without stored descriptors         |            28 |
| ReID persistence time                     | 0.866 seconds |

The database was subsequently checked and contained 110 tracks, 82 descriptor rows, 82 identity rows, and 82 persistent vehicle records for that run.

### 8.3 Later dashboard run showing candidate matches

A later dashboard screenshot for a different selected processing run showed:

| Dashboard metric    | Result |
| ------------------- | -----: |
| Tracks in run       |    110 |
| Descriptors stored  |     82 |
| New persistent IDs  |      0 |
| Candidate matches   |     82 |
| Confirmed matches   |      0 |
| Pending             |      0 |
| Not processed       |     28 |
| Missing descriptors |     28 |

This dashboard result indicates that 82 tracks had candidate identity records and 28 tracks had no descriptor/identity record. It does **not** establish that all 82 candidates are correct physical-vehicle matches. Candidate matches remain unconfirmed.

### 8.4 Descriptor collector unit tests

The following tests passed:

- Descriptor conversion and aggregation.
- Insufficient sample handling.
- Rejection of low-confidence observations and duplicate frames.
- Unpublished track handling.

### 8.5 Database lookup, persistence, and pipeline helper tests

The following test groups passed during development:

- Track ID lookup: 3 tests.
- Run persistence: 4 tests.
- ReID pipeline lookup/integration helper: 2 tests.
- Descriptor collector: 4 tests.

Combined test run at that point:

```text
Ran 13 tests
OK
```

Python compilation of the updated video processor, ReID integration, track lookup, descriptor collector, and run persistence modules also completed successfully at that point.

These unit tests and compilation checks do not replace an end-to-end test using the current local code, database, API, and dashboard.

## 9. Useful Development and Verification Commands

Run these from PowerShell at the project root, using the project's active `.venv`.

### Run all unit tests

```powershell
python -m unittest discover -s tests -v
```

### Run the descriptor collector tests

```powershell
python -m unittest discover -s tests -p "test_track_descriptor_collector.py" -v
```

### Run the track ID lookup tests

```powershell
python -m unittest discover -s tests -p "test_track_id_lookup.py" -v
```

### Run the persistence tests

```powershell
python -m unittest discover -s tests -p "test_run_persistence.py" -v
```

### Run the ReID pipeline lookup tests

```powershell
python -m unittest discover -s tests -p "test_reid_pipeline_lookup.py" -v
```

### Compile the ReID-related Python modules

```powershell
python -m py_compile src/tvs/pipeline/video_processor.py src/tvs/pipeline/reid_integration.py src/tvs/db/track_id_lookup.py src/tvs/reid/track_descriptor_collector.py src/tvs/reid/run_persistence.py
```

Check the exit code immediately after compilation:

```powershell
$LASTEXITCODE
```

A value of `0` indicates that compilation succeeded.

### Verify the ReID database counts for a run

Open the PostgreSQL shell using the command in Section 5, then run the following SQL after replacing the run ID with the run being inspected:

```sql
SELECT
    COUNT(*) AS total_tracks,
    COUNT(d.track_id) AS descriptors_stored,
    COUNT(i.track_id) AS identity_records,
    COUNT(DISTINCT i.persistent_vehicle_id) AS persistent_identities
FROM vehicle_tracks t
LEFT JOIN vehicle_reid_descriptors d
    ON t.track_id = d.track_id
LEFT JOIN vehicle_track_identities i
    ON t.track_id = i.track_id
WHERE t.run_id = 'PASTE-RUN-UUID-HERE';
```

### List tracks that do not have stored descriptors

Replace the run UUID with the run being investigated:

```sql
SELECT
    t.published_track_id,
    t.vehicle_type,
    t.observation_count,
    t.first_frame_id,
    t.last_frame_id
FROM vehicle_tracks t
LEFT JOIN vehicle_reid_descriptors d
    ON t.track_id = d.track_id
WHERE t.run_id = 'PASTE-RUN-UUID-HERE'
  AND d.track_id IS NULL
ORDER BY t.published_track_id;
```

### Inspect identity statuses and candidate similarities

```sql
SELECT
    t.run_id,
    t.published_track_id,
    t.vehicle_type,
    i.match_status,
    i.persistent_vehicle_id,
    i.candidate_vehicle_id,
    i.candidate_similarity
FROM vehicle_tracks t
LEFT JOIN vehicle_track_identities i
    ON t.track_id = i.track_id
ORDER BY t.run_id, t.published_track_id;
```

### Check Docker and PostgreSQL port

```powershell
docker ps
Test-NetConnection localhost -Port 5432
```

## 10. Interpreting Missing Descriptors

A track without a descriptor is still a tracked vehicle. It means that the ReID collector did not produce a descriptor that passed its current collection and validation rules, or no descriptor was persisted for that track.

The current collector's minimum sample count, confidence threshold, and bounding-box area threshold are possible reasons. The current database does not record a detailed rejection reason for each missing descriptor, so the exact cause must be investigated from the track's observations and processing logs. Do not lower all thresholds without checking the cause; low-quality descriptors can increase false candidate matches.

## 11. Current Limitations and Next Validation Steps

- The ReID system produces candidate matches; candidate records are not automatically confirmed or merged.
- The cosine similarity threshold remains unset pending calibration on held-out footage.
- A missing descriptor does not mean ByteTrack failed to track the vehicle.
- The reason for each missing descriptor is not yet stored as a database diagnostic.
- The VeRi evaluation is a custom retrieval protocol and must be described as such.
- A convincing end-to-end cross-video demonstration requires two videos containing the same physical vehicles, with ground-truth identity annotations or independently verified matching vehicles.
- Similar-looking vehicles can produce false candidates. Candidate similarity alone is insufficient to establish identity.

## 12. Project Objective

The project aims to combine vehicle detection, within-video multi-object tracking, vehicle appearance descriptors, cross-run ReID candidate retrieval, PostgreSQL persistence, API services, and an interactive dashboard into one traffic surveillance application.

The current implementation demonstrates vehicle detection and tracking, descriptor extraction and persistence, and candidate generation. Confirmed cross-camera identity performance still requires carefully controlled validation and calibration.
