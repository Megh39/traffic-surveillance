import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Traffic Surveillance Dashboard",
    page_icon=None,
    layout="wide",
)

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
        "python -m uvicorn src.tvs.api.main:app --host 127.0.0.1 --port 8000"
    )
    st.stop()

st.sidebar.success("API and database connected")

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

    if not completed_runs:
        st.warning("No completed processing runs are available.")
        st.stop()

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

    summary_response = fetch_api(
        "/analytics/summary", api_url, params
    )
    summary = summary_response["summary"]
    run = summary_response["run"]

    vehicle_response = fetch_api(
        "/analytics/vehicle-types", api_url, params
    )
    vehicle_data = vehicle_response.get("data", [])

    timeline_response = fetch_api(
        "/analytics/observations-over-time", api_url, params
    )
    timeline_data = timeline_response.get("data", [])

    tracks_response = fetch_api(
        "/analytics/tracks",
        api_url,
        {"run_id": selected_run_id, "limit": 1000},
    )
    track_data = tracks_response.get("data", [])

except (requests.RequestException, KeyError, ValueError) as exc:
    st.error("Failed to load analytics from the API.")
    st.code(str(exc))
    st.stop()

st.subheader("Processing overview")
st.caption(
    f"Camera: {run['camera_id']} | "
    f"Run ID: {run['run_id']} | "
    f"Status: Completed"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Frames processed", f"{summary['frames_processed']:,}")
col2.metric("Total detections", f"{summary['total_detections']:,}")
col3.metric("Vehicle tracks", f"{summary['total_tracks']:,}")
col4.metric("Track observations", f"{summary['total_observations']:,}")

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

st.subheader("Vehicle track records")
st.caption(
    "Showing up to 1,000 tracks, ordered by observation count."
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
