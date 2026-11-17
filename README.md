
# Traffic Video Surveillance and Vehicle Analytics System

## Overview

The **Traffic Video Surveillance and Vehicle Analytics System** is a computer vision-based application designed to process traffic videos, detect vehicles, track their movement, and generate useful traffic analytics.

The system uses YOLO-based object detection to identify vehicles in video frames and ByteTrack for multi-object tracking. Detected vehicles are assigned tracking IDs, allowing the system to maintain their identities across consecutive frames.

The application supports two processing workflows:

- **Video upload:** Upload a traffic video through the Streamlit dashboard and process it directly using the detection and tracking pipeline.
- **Dataset processing:** Process the configured UA-DETRAC image sequence through the existing pipeline for traffic surveillance experiments and evaluation.

Processed observations and tracking information are stored in PostgreSQL. A FastAPI backend exposes the stored information through REST APIs, while a Streamlit dashboard provides video playback, processing statistics, visualizations, and historical analytics.

## Key Features

- Vehicle detection using YOLO.
- Multi-object vehicle tracking using ByteTrack.
- Unique tracking IDs for detected vehicles.
- Bounding-box and trajectory visualization.
- Video ingestion from uploaded video files.
- Image-sequence ingestion for dataset processing.
- Annotated video generation and download.
- PostgreSQL-based storage of tracking observations.
- REST API integration using FastAPI.
- Interactive dashboard using Streamlit.
- Vehicle-type distribution and observation timeline visualizations.
- Vehicle track records and CSV export.
- Processing history across multiple video runs.

## How the System Works

The application follows a sequential video analytics pipeline.

### 1. Video Ingestion

The system accepts video input through the `VideoFileSource` ingestion component.

For dashboard uploads, the user selects a video file, which is temporarily saved by the API before processing begins.

The ingestion component reads the video frame by frame and extracts relevant information, including:

- Frame image.
- Frame ID.
- Effective video FPS.
- Elapsed time.
- Source video information.

The system also supports image-sequence ingestion for dataset-based processing.

### 2. Vehicle Detection

Each incoming frame is passed to the YOLO-based vehicle detector.

The detector identifies vehicle objects and produces bounding boxes, class information, and confidence scores.

A configurable confidence threshold determines which detections are suitable for publication and subsequent processing.

### 3. Multi-Object Tracking

The detections are passed to the ByteTrack tracker.

The tracker associates detections across consecutive frames to maintain vehicle identities over time.

Each confirmed vehicle track receives a published tracking ID. The tracker also maintains information about vehicle positions and movement across frames.

Tracking observations that satisfy the configured publication criteria are passed to the persistence and visualization stages.

### 4. Visualization and Video Generation

The tracking renderer processes the original frames along with the tracking observations.

It generates annotated frames containing:

- Vehicle bounding boxes.
- Published vehicle tracking IDs.
- Vehicle movement trajectories.

The annotated frames are written into an output MP4 video.

The resulting video can be viewed or downloaded through the dashboard.

### 5. Database Persistence

The system uses PostgreSQL to store processing and tracking information.

The database maintains information about processing runs, cameras, vehicle tracks, and individual tracking observations.

Important processing information includes:

- Processing run ID.
- Camera ID.
- Source video FPS.
- Number of processed frames.
- Total detections.
- Published observations.
- Unique vehicle tracks.
- Tracking and observation details.

This allows the system to retrieve historical results without processing the same video again.

### 6. Analytics and API Integration

FastAPI provides the backend interface between the processing system, PostgreSQL database, and dashboard.

The API exposes endpoints for:

- Database health checks.
- Processing summaries.
- Vehicle-type statistics.
- Observations over time.
- Vehicle tracking records.
- Processing history.
- Video uploads and processing.
- Downloading processed videos.

### 7. Dashboard

The Streamlit dashboard provides an interactive interface for using the application.

Users can upload traffic videos, initiate processing, view the generated annotated video, and download the output.

The dashboard also displays processing statistics, vehicle-type distribution, observation timelines, vehicle track records, and historical processing runs.

## Pipeline Flow

The overall processing workflow can be represented as:

**Video Input → Frame Ingestion → YOLO Vehicle Detection → ByteTrack Tracking → Observation Persistence → Frame Annotation → Output Video**

The supporting application workflow is:

**Processing Pipeline → PostgreSQL Database → FastAPI Backend → Streamlit Dashboard**

For uploaded videos, the API receives the video, invokes the processing pipeline, stores the resulting observations, and makes the annotated video available for playback and download.

## Technology Stack

| Component             | Technology         |
| --------------------- | ------------------ |
| Programming Language  | Python             |
| Object Detection      | YOLO (Ultralytics) |
| Multi-Object Tracking | ByteTrack          |
| Computer Vision       | OpenCV             |
| Database              | PostgreSQL         |
| Backend API           | FastAPI            |
| Dashboard             | Streamlit          |
| Data Processing       | NumPy, Pandas      |
| Dependency Management | uv                 |
| Database Deployment   | Docker             |

## Requirements

Before running the application, ensure that the following are installed:

- Python 3.11 or later, but below Python 3.14.
- uv package manager.
- Docker Desktop.
- PostgreSQL database container configured for the project.
- NVIDIA GPU with compatible drivers and CUDA support for GPU acceleration (CPU fallback is supported by the detector configuration).
- Git.

## Installation and Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/Megh39/traffic-surveillance.git
cd traffic-surveillance
```

### 2. Install Dependencies

Use uv to install the project dependencies:

```powershell
uv sync
```

### 3. Start PostgreSQL

Ensure Docker Desktop is running.

If the project's PostgreSQL container already exists but is stopped, start it using:

```powershell
docker start traffic-postgres
```

Verify that the container is running:

```powershell
docker ps
```

The PostgreSQL container should expose port `5432`.

Ensure the database has been initialized and the project's database connection settings are configured before starting the application.

## Running the Application

The application uses separate processes for the database, FastAPI backend, and Streamlit dashboard.

Keep each service running in its own terminal.

### 1. Start the FastAPI Backend

Open a PowerShell terminal in the project directory:

```powershell
uv run uvicorn src.tvs.api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 2. Verify the Backend

Open the health endpoint:

```text
http://127.0.0.1:8000/health
```

A successful response should look like:

```json
{
  "status": "healthy",
  "database": "connected"
}
```

This confirms that the backend can communicate with PostgreSQL.

### 3. Start the Streamlit Dashboard

Open another PowerShell terminal in the project directory:

```powershell
uv run streamlit run dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

The dashboard communicates with the FastAPI backend, which retrieves and processes the required information from PostgreSQL.

## Uploading and Processing a Video

The dashboard supports direct video uploads.

1. Open `http://localhost:8501`.
2. Locate the **Upload and process a video** section.
3. Select a video file.
4. Click **Process uploaded video**.
5. Wait for the processing operation to complete.
6. Review the generated processing statistics.
7. Play or download the annotated video.

Supported video formats:

- MP4
- AVI
- MOV
- MKV
- M4V

The current API upload limit is **500 MB**.

The processing result includes information such as:

- Frames processed.
- Total vehicle detections.
- Publishable detections.
- Published tracking observations.
- Unique vehicle tracks.
- Source video resolution and FPS.
- Processing FPS.
- Average inference latency.
- Tracking association statistics.

The original uploaded file is temporarily stored during processing and removed after the request completes. The generated annotated video is retained in the output directory.

## Running the Dataset Processing Pipeline

The project also supports processing the configured UA-DETRAC image sequence through the existing command-line pipeline.

To run it:

```powershell
uv run python scripts/run_pipeline.py
```

The pipeline uses the configured camera, detector, and tracker settings to process the image sequence.

The resulting processing run and tracking observations are stored in PostgreSQL, and the generated annotated video is written to the configured output location.

## Useful Development Commands

### Run the Full Test Suite

```powershell
uv run pytest
```

### Run Detector Tests

```powershell
uv run pytest tests/perception/test_vehicle_detector.py
```

### Run Ingestion Tests

```powershell
uv run pytest tests/ingestion/
```

### Check Python Syntax

For example:

```powershell
uv run python -m py_compile src/tvs/api/main.py
```

### Check Docker Containers

```powershell
docker ps
```

### Stop the PostgreSQL Container

```powershell
docker stop traffic-postgres
```

To stop the FastAPI and Streamlit applications, press `Ctrl + C` in their respective terminals.

## Important Notes

- PostgreSQL must be running before using the backend and dashboard.
- FastAPI and Streamlit must both be running for the complete dashboard experience.
- Video processing currently runs synchronously through the API request.
- Processing time depends on video resolution, frame count, and available hardware.
- The detector supports automatic device selection and CPU fallback according to its configuration.
- Tracking quality can vary depending on vehicle overlap, occlusion, camera angle, video quality, and movement.
- The application is currently designed for local execution and demonstration.

## Project Objective

The primary objective of this project is to develop an integrated traffic video surveillance and analytics system that combines computer vision, multi-object tracking, database persistence, backend API services, and interactive visualization.

By connecting these components into a single workflow, the system transforms raw traffic videos into annotated video outputs and structured tracking information that can be used for traffic analysis and monitoring.
