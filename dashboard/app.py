import altair as alt
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
video_path = PROJECT_ROOT / "processed_h264.mp4"
st.set_page_config(
    page_title="Traffic Surveillance Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 500
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024

# Palette: asphalt, lane-marking yellow, road-sign green
ASPHALT = "#1E2328"
LANE = "#F2B705"
SIGN = "#0B6E4F"
PAPER = "#F6F7F5"
INK = "#1E2328"
MUTED = "#6B7280"
LINE = "#DDE0DB"

# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Barlow', system-ui, sans-serif;
    }}
    .stApp {{ background: {PAPER}; }}
    .block-container {{ padding-top: 2rem; max-width: 1280px; }}

    h1, h2, h3, h4 {{
        font-family: 'Barlow Semi Condensed', 'Barlow', sans-serif !important;
        color: {INK};
        letter-spacing: 0.1px;
    }}
    h1 {{ font-weight: 700; font-size: 2.3rem !important; margin-bottom: 0; }}
    h2 {{ font-weight: 600; }}
    h3 {{ font-weight: 600; }}

    /* Force light colors in the main area (works even when the browser is in dark mode) */
    :root {{ color-scheme: light; }}
    [data-testid="stHeader"] {{ background: {PAPER}; }}
    [data-testid="stHeader"] *,
    [data-testid="stToolbar"] * {{ color: {INK} !important; }}
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] label,
    [data-testid="stMain"] label *,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"]:not(button *),
    [data-testid="stMain"] [data-testid="stMarkdownContainer"]:not(button *) * {{
        color: {INK} !important;
    }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] *,
    [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] * {{
        color: {MUTED} !important;
    }}
    [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: {MUTED} !important;
    }}
    [data-testid="stMain"] [data-testid="stFileUploader"] small {{ color: {MUTED} !important; }}
    [data-testid="stMain"] [data-testid="stExpander"] summary,
    [data-testid="stMain"] [data-testid="stExpander"] summary * {{ color: {INK} !important; }}
    [data-testid="stMain"] [data-baseweb="tab"] {{ color: {MUTED} !important; }}
    [data-testid="stMain"] [data-baseweb="tab"][aria-selected="true"] {{ color: {INK} !important; }}
    [data-testid="stMain"] [data-baseweb="tab-border"] {{ background-color: {LINE} !important; }}
    [data-testid="stMain"] [data-baseweb="tab-highlight"] {{
        background-color: {LANE} !important;
        height: 3px !important;
    }}
    [data-testid="stMain"] hr {{ border-color: {LINE} !important; opacity: 1; }}
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="input"] > div {{
        background: #FFFFFF !important;
        color: {INK} !important;
        border-color: {LINE} !important;
    }}
    [data-testid="stMain"] .stButton > button[kind="primary"] p {{ color: {ASPHALT} !important; }}
    [data-testid="stMain"] .stDownloadButton > button,
    [data-testid="stMain"] [data-testid="stFileUploader"] button {{
        background: {ASPHALT};
        border: none;
    }}
    [data-testid="stMain"] .stDownloadButton > button *,
    [data-testid="stMain"] [data-testid="stFileUploader"] button * {{ color: #FFFFFF !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {ASPHALT}; }}
    section[data-testid="stSidebar"] * {{ color: #E8EAE6; }}
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: #2B3138 !important;
        border-color: #3A424B !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: transparent;
        border: 1px solid #4A535D;
        color: #FFFFFF;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        border-color: {LANE};
        color: {LANE};
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        border-left: 4px solid {LANE};
        border-radius: 6px;
        padding: 0.85rem 1rem;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {MUTED};
        font-size: 0.85rem;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: {INK};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Barlow Semi Condensed', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.6rem 0;
    }}
    .stTabs [aria-selected="true"] {{ color: {INK}; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {LANE}; height: 3px; }}

    /* Buttons */
    .stButton > button[kind="primary"],
    .stDownloadButton > button {{
        border-radius: 6px;
        font-weight: 600;
    }}
    .stButton > button[kind="primary"] {{
        background: {LANE};
        border: none;
        color: {ASPHALT};
    }}
    .stButton > button[kind="primary"]:hover {{
        background: #FFC71F;
        color: {ASPHALT};
    }}

    /* Status pill */
    .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #E3F1EB;
        color: {SIGN};
    }}
    .run-meta {{ color: {MUTED}; font-size: 0.92rem; margin: 0.25rem 0 1rem; }}
    .run-meta b {{ color: {INK}; font-weight: 600; }}

    [data-testid="stFileUploader"] section {{
        border: 2px dashed #C3C8BF;
        background: #FFFFFF;
        border-radius: 8px;
    }}
    hr {{ border-color: {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------


@st.cache_data(ttl=15)
def fetch_api(endpoint, base_url, params=None):
    response = requests.get(
        f"{base_url}{endpoint}",
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def fmt_int(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "–"


def fmt_float(value, suffix=""):
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return "–"


def fmt_time(value):
    try:
        return pd.to_datetime(value).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(value)


def chart_theme(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED,
        titleColor=MUTED,
        gridColor="#E6E8E4",
        domainColor=LINE,
        tickColor=LINE,
        labelFont="Barlow",
        titleFont="Barlow",
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Dashboard")

    api_url = st.text_input(
        "API base URL",
        value="http://127.0.0.1:8000",
    ).rstrip("/")

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("Traffic Surveillance")
st.caption("Vehicle detection, tracking, and traffic analytics")

# ---------------------------------------------------------
# API AND DATABASE CONNECTION
# ---------------------------------------------------------

try:
    health = fetch_api("/health", api_url)

    if health.get("status") != "healthy":
        st.error(
            "The API or database is not healthy. Check the API terminal for errors."
        )
        st.stop()

except requests.RequestException as exc:
    st.error(f"Cannot connect to the API at {api_url}.")
    st.code(str(exc))
    st.info(
        "Start the API in a separate PowerShell window with: "
        "`uv run uvicorn src.tvs.api.main:app --host 127.0.0.1 --port 8000 --reload`"
    )
    st.stop()

st.sidebar.success("API and database connected")

# ---------------------------------------------------------
# LOAD RUNS (needed for the sidebar picker)
# ---------------------------------------------------------

try:
    runs_response = fetch_api("/analytics/runs", api_url, {"limit": 100})
    runs = runs_response.get("data", [])
    completed_runs = [r for r in runs if r.get("status") == "completed"]
except requests.RequestException as exc:
    st.error("Could not load processing history.")
    st.code(str(exc))
    st.stop()

selected_run_id = None

if completed_runs:
    run_options = {
        f"{run['camera_id']} · {fmt_time(run['started_at'])} · {str(run['run_id'])[:8]}": run[
            "run_id"
        ]
        for run in completed_runs
    }

    with st.sidebar:
        st.divider()
        st.subheader("Processing run")
        selected_label = st.selectbox(
            "Choose a run to analyze",
            list(run_options.keys()),
        )
        selected_run_id = run_options[selected_label]

tab_process, tab_analytics, tab_tracks, tab_reid, tab_history = st.tabs(
    ["Process video", "Analytics", "Vehicle tracks", "Vehicle Re-ID", "History"]
)

# =========================================================
# TAB 1: UPLOAD AND PROCESS
# =========================================================

with tab_process:
    st.subheader("Upload a traffic video")
    st.write(
        "The video is analyzed for vehicles, tracked frame to frame, "
        "annotated, and saved to the database."
    )

    uploaded_file = st.file_uploader(
        "Choose a traffic video",
        type=["mp4", "avi", "mov", "mkv", "m4v"],
        max_upload_size=MAX_UPLOAD_MB,
        help=f"Maximum upload size: {MAX_UPLOAD_MB} MB.",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        info_col, action_col = st.columns([3, 1], vertical_alignment="center")
        info_col.markdown(
            f"**{uploaded_file.name}**  \n"
            f"<span style='color:{MUTED}'>{file_size_mb:.2f} MB</span>",
            unsafe_allow_html=True,
        )

        if uploaded_file.size > MAX_UPLOAD_SIZE:
            st.error(
                f"This video is over the {MAX_UPLOAD_MB} MB limit. Choose a smaller file."
            )
        else:
            start = action_col.button(
                "Process video",
                type="primary",
                use_container_width=True,
            )

            if start:
                try:
                    with st.spinner(
                        "Processing video. Detection and tracking can take several minutes."
                    ):
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

                    st.session_state["last_upload_result"] = upload_result
                    st.session_state.pop("processed_video", None)
                    st.cache_data.clear()
                    st.toast("Video processed", icon="✅")

                except requests.HTTPError as exc:
                    st.error("The API rejected the video or failed to process it.")
                    if exc.response is not None:
                        try:
                            st.json(exc.response.json())
                        except ValueError:
                            st.code(exc.response.text)
                    else:
                        st.code(str(exc))

                except requests.Timeout:
                    st.error(
                        "The request timed out. Check the API terminal "
                        "to see if processing is still running before you retry."
                    )

                except requests.RequestException as exc:
                    st.error("Could not reach the processing API.")
                    st.code(str(exc))

                except ValueError as exc:
                    st.error("The API returned a response that could not be read.")
                    st.code(str(exc))

    # ----- Latest result -----
    last_upload = st.session_state.get("last_upload_result")

    if last_upload:
        st.divider()
        st.subheader("Latest result")

        result = last_upload.get("result", {})
        upload_id = last_upload.get("upload_id")

        if upload_id:
            st.markdown(
                f"<div class='run-meta'><span class='pill'>Completed</span> "
                f"&nbsp; Upload ID <b>{upload_id}</b></div>",
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames processed", fmt_int(result.get("frames_processed", 0)))
        m2.metric("Total detections", fmt_int(result.get("total_detections", 0)))
        m3.metric("Unique vehicle tracks", fmt_int(result.get("unique_tracks", 0)))
        m4.metric(
            "Published observations", fmt_int(result.get("published_observations", 0))
        )

        m5, m6, m7, m8 = st.columns(4)
        m5.metric(
            "Video resolution",
            f"{result.get('width', 0)} × {result.get('height', 0)}",
        )
        m6.metric("Source FPS", fmt_float(result.get("fps", 0)))
        m7.metric("Processing FPS", fmt_float(result.get("processing_fps", 0)))
        m8.metric(
            "Avg inference latency",
            fmt_float(result.get("average_inference_ms", 0), " ms"),
        )

        st.caption(
            f"Processed in {fmt_float(result.get('elapsed_seconds', 0))} seconds · "
            f"{fmt_int(result.get('publishable_detections', 0))} publishable detections"
        )

        with st.expander("Tracking association statistics"):
            st.json(result.get("association_stats", {}))

        reid_result = result.get("reid", {})
        st.subheader("Re-Identification processing")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Descriptors stored", fmt_int(reid_result.get("persisted_descriptors", 0)))
        r2.metric("New persistent IDs", fmt_int(reid_result.get("created_new_identities", 0)))
        r3.metric("Cross-run candidates", fmt_int(reid_result.get("candidate_count", 0)))
        r4.metric("Tracks skipped", fmt_int(reid_result.get("skipped_missing_track", 0)))
        st.caption(
            "Descriptors are visual feature vectors. A candidate is a possible "
            "cross-run match, not an automatically confirmed vehicle identity."
        )
        if reid_result.get("candidates"):
            st.dataframe(
                pd.DataFrame(reid_result["candidates"]),
                use_container_width=True,
                hide_index=True,
            )

        if upload_id:
            st.subheader("Processed video")

            # Let the browser stream the MP4 directly from the API.
            # This avoids transferring the entire video through Streamlit's
            # Python session and embedding it as an in-memory byte payload.
            video_url = f"{api_url}/videos/{upload_id}/output"

            video_col, _ = st.columns([3, 1])
            with video_col:
                st.video(video_url)

            st.markdown(f"[Open or download the processed video]({video_url})")

# =========================================================
# ANALYTICS TABS (need a selected run)
# =========================================================

summary = run = None
reid_response = {}
vehicle_data, timeline_data, track_data, reid_summary, reid_data = [], [], [], {}, []

if selected_run_id is not None:
    params = {"run_id": selected_run_id}
    try:
        summary_response = fetch_api("/analytics/summary", api_url, params)
        summary = summary_response["summary"]
        run = summary_response["run"]

        vehicle_data = fetch_api("/analytics/vehicle-types", api_url, params).get(
            "data", []
        )
        timeline_data = fetch_api(
            "/analytics/observations-over-time", api_url, params
        ).get("data", [])
        track_data = fetch_api(
            "/analytics/tracks",
            api_url,
            {"run_id": selected_run_id, "limit": 1000},
        ).get("data", [])
        reid_response = fetch_api("/analytics/reid", api_url, params)
        reid_summary = reid_response.get("summary", {})
        reid_data = reid_response.get("data", [])
    except (requests.RequestException, KeyError, ValueError) as exc:
        summary = run = None
        with tab_analytics:
            st.error("Could not load analytics for this run.")
            st.code(str(exc))

NO_RUN_MESSAGE = (
    "No completed runs yet. Upload a video in the **Process video** tab "
    "to create the first one."
)

# ---------------------------------------------------------
# TAB 2: ANALYTICS
# ---------------------------------------------------------

with tab_analytics:
    if not completed_runs:
        st.info(NO_RUN_MESSAGE)

    elif summary and run:
        st.markdown(
            f"<div class='run-meta'><span class='pill'>Completed</span> &nbsp; "
            f"Camera <b>{run['camera_id']}</b> &nbsp;·&nbsp; "
            f"Run <b>{str(run['run_id'])[:8]}</b> &nbsp;·&nbsp; "
            f"Started <b>{fmt_time(run.get('started_at'))}</b></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames processed", fmt_int(summary["frames_processed"]))
        c2.metric("Total detections", fmt_int(summary["total_detections"]))
        c3.metric("Vehicle tracks", fmt_int(summary["total_tracks"]))
        c4.metric("Track observations", fmt_int(summary["total_observations"]))

        c5, c6, c7, _ = st.columns(4)
        c5.metric(
            "Frames with observations", fmt_int(summary["frames_with_observations"])
        )
        c6.metric("Source video FPS", fmt_float(run["source_fps"]))
        c7.metric("Published observations", fmt_int(summary["published_observations"]))

        st.write("")
        left, right = st.columns(2)

        with left:
            st.subheader("Vehicle types")

            if vehicle_data:
                vehicle_df = pd.DataFrame(vehicle_data).sort_values(
                    "track_count", ascending=False
                )

                bars = (
                    alt.Chart(vehicle_df)
                    .mark_bar(color=ASPHALT, cornerRadiusEnd=3)
                    .encode(
                        x=alt.X("track_count:Q", title="Tracks"),
                        y=alt.Y("vehicle_type:N", sort="-x", title=None),
                        tooltip=[
                            alt.Tooltip("vehicle_type:N", title="Type"),
                            alt.Tooltip("track_count:Q", title="Tracks", format=","),
                        ],
                    )
                    .properties(height=max(180, 38 * len(vehicle_df)))
                )
                st.altair_chart(chart_theme(bars), use_container_width=True)

                with st.expander("View as table"):
                    st.dataframe(vehicle_df, use_container_width=True, hide_index=True)
            else:
                st.info("No vehicle-type data for this run.")

        with right:
            st.subheader("Observations over time")

            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data).rename(
                    columns={
                        "interval_start_seconds": "seconds",
                        "observation_count": "observations",
                    }
                )

                area = (
                    alt.Chart(timeline_df)
                    .mark_area(
                        line={"color": SIGN, "strokeWidth": 2},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="#B9DDCE", offset=0),
                                alt.GradientStop(color="#EAF4EF", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=1,
                            y2=0,
                        ),
                        interpolate="monotone",
                    )
                    .encode(
                        x=alt.X("seconds:Q", title="Time (seconds)"),
                        y=alt.Y("observations:Q", title="Observations"),
                        tooltip=[
                            alt.Tooltip("seconds:Q", title="Second"),
                            alt.Tooltip(
                                "observations:Q", title="Observations", format=","
                            ),
                        ],
                    )
                    .properties(height=max(180, 38 * max(len(vehicle_data), 5)))
                )
                st.altair_chart(chart_theme(area), use_container_width=True)
            else:
                st.info("No observation timeline for this run.")

# ---------------------------------------------------------
# TAB 3: TRACKS
# ---------------------------------------------------------

with tab_tracks:
    if not completed_runs:
        st.info(NO_RUN_MESSAGE)

    elif track_data:
        tracks_df = pd.DataFrame(track_data)

        st.subheader("Vehicle track records")

        filtered_df = tracks_df
        if "vehicle_type" in tracks_df.columns:
            types = sorted(tracks_df["vehicle_type"].dropna().unique().tolist())
            chosen = st.multiselect(
                "Filter by vehicle type",
                types,
                placeholder="All vehicle types",
            )
            if chosen:
                filtered_df = tracks_df[tracks_df["vehicle_type"].isin(chosen)]

        st.caption(
            f"Showing {len(filtered_df):,} of {len(tracks_df):,} tracks "
            "(up to 1,000, ordered by observation count)."
        )

        st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=480)

        st.download_button(
            "Download as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="vehicle_track_records.csv",
            mime="text/csv",
        )
    else:
        st.info("No track records for this run.")

# ---------------------------------------------------------
# TAB 4: VEHICLE RE-IDENTIFICATION
# ---------------------------------------------------------

with tab_reid:
    st.subheader("Vehicle Re-Identification")
    st.write(
        "This view reads the persisted descriptor and identity records for the "
        "selected processing run. It distinguishes newly assigned persistent "
        "IDs from unconfirmed cross-run candidates."
    )

    if not completed_runs:
        st.info(NO_RUN_MESSAGE)
    elif selected_run_id is None:
        st.info("Select a completed processing run in the sidebar.")
    elif reid_data or reid_summary:
        st.caption(
            f"Run {str(selected_run_id)[:8]} · "
            f"{fmt_time(run.get('started_at')) if run else ''}"
        )

        a, b, c, d = st.columns(4)
        a.metric("Tracks in run", fmt_int(reid_summary.get("total_tracks", 0)))
        b.metric("Descriptors stored", fmt_int(reid_summary.get("descriptors_stored", 0)))
        c.metric("New persistent IDs", fmt_int(reid_summary.get("new_identities", 0)))
        d.metric("Candidate matches", fmt_int(reid_summary.get("candidate_matches", 0)))

        e, f, g, h = st.columns(4)
        e.metric("Confirmed matches", fmt_int(reid_summary.get("confirmed_matches", 0)))
        f.metric("Pending", fmt_int(reid_summary.get("pending_identities", 0)))
        g.metric("Not processed", fmt_int(reid_summary.get("not_processed", 0)))
        h.metric("Missing descriptors", fmt_int(reid_summary.get("descriptors_missing", 0)))

        st.markdown("**Identity status distribution**")
        status_counts = reid_response.get("status_counts", {})
        if status_counts:
            status_df = pd.DataFrame(
                [{"Status": key.replace("_", " ").title(), "Tracks": value}
                 for key, value in status_counts.items()]
            )
            st.dataframe(status_df, use_container_width=True, hide_index=True)

        reid_df = pd.DataFrame(reid_data)
        if not reid_df.empty:
            display_columns = [
                "published_track_id",
                "track_id",
                "vehicle_type",
                "observation_count",
                "descriptor_stored",
                "embedding_dimension",
                "match_status",
                "persistent_vehicle_id",
                "candidate_vehicle_id",
                "candidate_similarity",
            ]
            display_columns = [column for column in display_columns if column in reid_df.columns]
            view_df = reid_df[display_columns].copy()

            status_filter = st.multiselect(
                "Filter identity status",
                sorted(view_df["match_status"].dropna().unique().tolist()),
                default=sorted(view_df["match_status"].dropna().unique().tolist()),
            )
            view_df = view_df[view_df["match_status"].isin(status_filter)]

            vehicle_types_filter = sorted(view_df["vehicle_type"].dropna().unique().tolist())
            chosen_vehicle_types = st.multiselect(
                "Filter vehicle type",
                vehicle_types_filter,
                placeholder="All vehicle types",
            )
            if chosen_vehicle_types:
                view_df = view_df[view_df["vehicle_type"].isin(chosen_vehicle_types)]

            st.caption(f"Showing {len(view_df):,} ReID track records.")
            st.dataframe(view_df, use_container_width=True, hide_index=True, height=480)

            st.download_button(
                "Download ReID records as CSV",
                data=view_df.to_csv(index=False).encode("utf-8"),
                file_name="vehicle_reid_records.csv",
                mime="text/csv",
            )

            with st.expander("How to interpret these fields"):
                st.markdown(
                    "- **Published Track ID:** ID assigned by the tracking pipeline within this run.\n"
                    "- **Database Track ID:** Internal PostgreSQL row ID for that track.\n"
                    "- **Persistent Vehicle ID:** Identity record created for a vehicle track; "
                    "it is not proof of a cross-camera match unless the status is Confirmed.\n"
                    "- **Candidate Vehicle ID:** Existing persistent identity suggested by visual "
                    "similarity. It remains unconfirmed.\n"
                    "- **Descriptor:** A 512-dimensional numerical representation of the vehicle crop."
                )
        else:
            st.info("No vehicle tracks were returned for this run.")
    else:
        st.info("No ReID records are available for this run.")

# ---------------------------------------------------------
# TAB 5: HISTORY
# ---------------------------------------------------------

with tab_history:
    st.subheader("Processing history")

    if runs:
        history_df = pd.DataFrame(runs)

        for col in ("started_at", "completed_at", "finished_at"):
            if col in history_df.columns:
                history_df[col] = history_df[col].map(
                    lambda v: fmt_time(v) if v else "–"
                )

        st.caption(f"{len(history_df):,} most recent runs.")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No processing history yet.")

st.divider()
st.caption("Data source: PostgreSQL through the Traffic Surveillance API.")
