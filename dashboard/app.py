
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Traffic Surveillance Dashboard",
    page_icon=None,
    layout="wide",
)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024

st.title("Traffic Surveillance Dashboard")
st.caption("Vehicle detection, tracking, and traffic analytics")

with st.sidebar:
    st.header("Dashboard settings")

    api_url = st.text_input(
        "API base URL",
        value="http://127.0.0.1:8000",
    ).rstrip("/")

    if st.button("Refresh dashboard"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=15)
def fetch_api(endpoint, base_url, params=None):
    response = requests.get(
        f"{base_url}{endpoint}",
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------
# API AND DATABASE CONNECTION
# ---------------------------------------------------------

try:
    health = fetch_api("/health", api_url)

    if health.get("status") != "healthy":
        st.error("The API or database is not healthy.")
        st.stop()

except requests.RequestException as exc:
    st.error(f"Cannot connect to the API at {api_url}.")
    st.code(str(exc))
    st.info(
        "Start the API in a separate PowerShell window with: "
        "uv run uvicorn src.tvs.api.main:app "
        "--host 127.0.0.1 --port 8000 --reload"
    )
    st.stop()

st.sidebar.success("API and database connected")


# ---------------------------------------------------------
# VIDEO UPLOAD AND PROCESSING
# ---------------------------------------------------------

st.header("Upload and process a video")

st.write(
    "Upload a traffic video to run vehicle detection, "
    "tracking, annotation, and PostgreSQL persistence."
)

uploaded_file = st.file_uploader(
    "Choose a traffic video",
    type=["mp4", "avi", "mov", "mkv", "m4v"],
    max_upload_size=500,
    help="Maximum upload size: 500 MB.",
)

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.write(
        f"Selected file: **{uploaded_file.name}** "
        f"({file_size_mb:.2f} MB)"
    )

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        st.error("The selected video exceeds the 500 MB limit.")

    else:
        if st.button(
            "Process uploaded video",
            type="primary",
            disabled=False,
        ):
            try:
                progress = st.progress(
                    0,
                    text="Uploading video to the processing API...",
                )

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                response = requests.post(
                    f"{api_url}/videos/process",
                    files=files,
                    timeout=(15, 1800),
                )

                response.raise_for_status()
                upload_result = response.json()

                progress.progress(
                    100,
                    text="Video processing completed.",
                )

                st.session_state["last_upload_result"] = upload_result

                st.cache_data.clear()

                st.success("Video processed successfully.")

            except requests.HTTPError as exc:
                st.error("The API rejected or failed to process the video.")

                if exc.response is not None:
                    try:
                        st.json(exc.response.json())
                    except ValueError:
                        st.code(exc.response.text)
                else:
                    st.code(str(exc))

            except requests.Timeout:
                st.error(
                    "The processing request timed out. "
                    "Check the API terminal before retrying."
                )

            except requests.RequestException as exc:
                st.error("Could not communicate with the processing API.")
                st.code(str(exc))

            except ValueError as exc:
                st.error("The API returned an invalid response.")
                st.code(str(exc))


# ---------------------------------------------------------
# DISPLAY THE LATEST UPLOAD RESULT
# ---------------------------------------------------------

last_upload = st.session_state.get("last_upload_result")

if last_upload:
    st.divider()
    st.subheader("Latest uploaded video result")

    result = last_upload.get("result", {})
    upload_id = last_upload.get("upload_id")

    if upload_id:
        st.caption(f"Upload ID: {upload_id}")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Frames processed",
        f"{result.get('frames_processed', 0):,}",
    )

    metric2.metric(
        "Total detections",
        f"{result.get('total_detections', 0):,}",
    )

    metric3.metric(
        "Unique vehicle tracks",
        f"{result.get('unique_tracks', 0):,}",
    )

    metric4.metric(
        "Published observations",
        f"{result.get('published_observations', 0):,}",
    )

    metric5, metric6, metric7, metric8 = st.columns(4)

    metric5.metric(
        "Video resolution",
        f"{result.get('width', 0)} × {result.get('height', 0)}",
    )

    metric6.metric(
        "Source FPS",
        f"{result.get('fps', 0):.2f}",
    )

    metric7.metric(
        "Processing FPS",
        f"{result.get('processing_fps', 0):.2f}",
    )

    metric8.metric(
        "Average inference latency",
        f"{result.get('average_inference_ms', 0):.2f} ms",
    )

    st.write(
        f"**Processing time:** "
        f"{result.get('elapsed_seconds', 0):.2f} seconds"
    )

    st.write(
        f"**Publishable detections:** "
        f"{result.get('publishable_detections', 0):,}"
    )

    association = result.get("association_stats", {})

    with st.expander("Tracking association statistics"):
        st.json(association)

    if upload_id:
        output_url = f"{api_url}/videos/{upload_id}/output"

        try:
            video_response = requests.get(
                output_url,
                timeout=(15, 1800),
            )
            video_response.raise_for_status()

            video_bytes = video_response.content

            st.subheader("Processed video playback")

            st.video(video_bytes)

            st.download_button(
                label="Download processed video",
                data=video_bytes,
                file_name=f"processed_{upload_id}.mp4",
                mime="video/mp4",
            )

        except requests.RequestException as exc:
            st.error("Could not retrieve the processed video.")
            st.code(str(exc))

    st.divider()


# ---------------------------------------------------------
# PROCESSING HISTORY AND ANALYTICS
# ---------------------------------------------------------

st.header("Traffic analytics")

try:
    runs_response = fetch_api(
        "/analytics/runs",
        api_url,
        {"limit": 100},
    )

    runs = runs_response.get("data", [])

    completed_runs = [
        run for run in runs
        if run.get("status") == "completed"
    ]

except requests.RequestException as exc:
    st.error("Failed to load processing history.")
    st.code(str(exc))
    st.stop()

if not completed_runs:
    st.warning(
        "No completed processing runs are available yet. "
        "Upload a video above to create your first run."
    )

else:
    run_options = {
        (
            f"{run['camera_id']} | "
            f"{run['started_at']} | "
            f"{str(run['run_id'])[:8]}"
        ): run["run_id"]
        for run in completed_runs
    }

    selected_label = st.sidebar.selectbox(
        "Select processing run",
        list(run_options.keys()),
    )

    selected_run_id = run_options[selected_label]
    params = {"run_id": selected_run_id}

    try:
        summary_response = fetch_api(
            "/analytics/summary",
            api_url,
            params,
        )

        summary = summary_response["summary"]
        run = summary_response["run"]

        vehicle_response = fetch_api(
            "/analytics/vehicle-types",
            api_url,
            params,
        )

        vehicle_data = vehicle_response.get("data", [])

        timeline_response = fetch_api(
            "/analytics/observations-over-time",
            api_url,
            params,
        )

        timeline_data = timeline_response.get("data", [])

        tracks_response = fetch_api(
            "/analytics/tracks",
            api_url,
            {
                "run_id": selected_run_id,
                "limit": 1000,
            },
        )

        track_data = tracks_response.get("data", [])

    except (requests.RequestException, KeyError, ValueError) as exc:
        st.error("Failed to load analytics from the API.")
        st.code(str(exc))
        st.stop()

    # -----------------------------------------------------
    # PROCESSING OVERVIEW
    # -----------------------------------------------------

    st.subheader("Processing overview")

    st.caption(
        f"Camera: {run['camera_id']} | "
        f"Run ID: {run['run_id']} | "
        f"Status: Completed"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frames processed",
        f"{summary['frames_processed']:,}",
    )

    col2.metric(
        "Total detections",
        f"{summary['total_detections']:,}",
    )

    col3.metric(
        "Vehicle tracks",
        f"{summary['total_tracks']:,}",
    )

    col4.metric(
        "Track observations",
        f"{summary['total_observations']:,}",
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Frames with observations",
        f"{summary['frames_with_observations']:,}",
    )

    col6.metric(
        "Source video FPS",
        f"{run['source_fps']:.2f}",
    )

    col7.metric(
        "Published observations",
        f"{summary['published_observations']:,}",
    )

    st.divider()

    # -----------------------------------------------------
    # VEHICLE TYPE DISTRIBUTION AND TIMELINE
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.subheader("Vehicle type distribution")

        if vehicle_data:
            vehicle_df = pd.DataFrame(vehicle_data)
            vehicle_df = vehicle_df.set_index("vehicle_type")

            st.bar_chart(vehicle_df[["track_count"]])

            st.dataframe(
                vehicle_df.reset_index(),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No vehicle-type data is available.")

    with right:
        st.subheader("Observations over time")

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)

            timeline_df = timeline_df.rename(
                columns={
                    "interval_start_seconds": "Time (seconds)",
                    "observation_count": "Observations",
                }
            )

            timeline_df = timeline_df.set_index("Time (seconds)")

            st.line_chart(timeline_df[["Observations"]])

        else:
            st.info("No observation timeline is available.")

    st.divider()

    # -----------------------------------------------------
    # VEHICLE TRACK RECORDS
    # -----------------------------------------------------

    st.subheader("Vehicle track records")

    st.caption(
        "Showing up to 1,000 tracks, "
        "ordered by observation count."
    )

    if track_data:
        tracks_df = pd.DataFrame(track_data)

        st.dataframe(
            tracks_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download track records as CSV",
            data=tracks_df.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_track_records.csv",
            mime="text/csv",
        )

    else:
        st.info("No track records are available.")

    st.divider()

    # -----------------------------------------------------
    # PROCESSING HISTORY
    # -----------------------------------------------------

    st.subheader("Processing history")

    if runs:
        history_df = pd.DataFrame(runs)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No processing history is available.")


st.caption(
    "Data source: PostgreSQL through the Traffic Surveillance API."
)
