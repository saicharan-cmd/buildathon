"""Buildathon — Streamlit-only app (local only).

Run it — nothing else needed first:
    streamlit run app.py

On startup this app takes care of everything by itself:
- creates buildathon.db (SQLite) if missing, with the 11 cameras
  and a few clearly-labelled demo detections
- downloads the plate model (models/yolov8-plate.pt, ~6 MB) on
  first use of the camera page, if it is not there yet

No Postgres, no FastAPI / uvicorn, no second terminal, no .env.
Local SQLite only — it does NOT sync to the live Render database.
"""

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plate_core  # noqa: E402
from plate_core import (  # noqa: E402
    CAMERA_BY_ID,
    MIN_CONFIDENCE,
    WATCHLIST,
    acknowledge_alert,
    detect_and_save,
    ensure_model,
    init_db,
    list_alerts,
    list_cameras,
    list_detections,
    model_status,
    overview_stats,
    track_plate,
)

st.set_page_config(
    page_title="Buildathon — ANPR Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ------------------------------------------------------------------ setup
# Runs on every page load, but init_db() is idempotent: a second run
# never duplicates cameras, demo detections or alerts.
try:
    _seed_info = init_db()
except Exception as exc:  # noqa: BLE001 - a broken DB must not blank the page
    st.error(f"Could not open the local database (buildathon.db): {exc}")
    st.stop()

st.sidebar.title("🚦 Buildathon ANPR")
st.sidebar.caption("Streamlit-only · local SQLite · no backend needed")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Detections", "Live Capture", "Track any plate",
     "Cameras", "Alerts"],
)
region = st.sidebar.radio(
    "Region", ["All", "Odisha", "Telangana"], horizontal=True,
)
region_filter = {"All": None, "Odisha": "odisha",
                 "Telangana": "telangana"}[region]
st.sidebar.divider()
st.sidebar.info(
    "Demo plates — try these:\n\n"
    "• **TS09EX1234** — Hyderabad route\n"
    "• **TS08AB7777** — stolen / watchlist\n"
    "• **OD02AB1234** — Bhubaneswar"
)
st.sidebar.caption("Made with ❤️ by Saicharan Peddapelli")
st.sidebar.caption(
    "Local demo only — detections here stay on this laptop and "
    "do not sync to the live Render database."
)

# Model status in the sidebar, so a missing model is visible before
# anyone opens the camera page.
_model_exists, _model_size_ok, _model_sha_ok = model_status()
if _model_exists and _model_sha_ok:
    st.sidebar.success("Plate model ready.")
elif _model_exists:
    st.sidebar.warning(
        "Plate model file looks wrong — it will be re-downloaded "
        "when you use Live Capture."
    )
else:
    st.sidebar.info(
        "Plate model not downloaded yet (~6 MB). It downloads "
        "automatically the first time you use Live Capture."
    )


# -------------------------------------------------------------- helpers
def _detections_df(region=None, camera_id=None, plate_query=None,
                   limit=500):
    """Detections as a DataFrame, empty-safe (always has columns)."""
    rows = list_detections(
        region=region, camera_id=camera_id,
        plate_query=plate_query, limit=limit,
    )
    columns = ["timestamp", "plate_text", "camera_id", "camera_name",
               "city", "region", "confidence", "source", "event_id"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def _camera_label(camera_id):
    cam = CAMERA_BY_ID.get(camera_id)
    if cam is None:
        return camera_id
    return f"{camera_id} — {cam['name']} ({cam['city']})"


def _show_detections_table(df):
    """Table with a DEMO / CAPTURED badge column. Empty-safe."""
    if df.empty:
        st.info("No detections yet. Use Live Capture, or check the "
                "region filter above.")
        return
    show = df.copy()
    show["source"] = show["source"].str.upper()
    show["confidence"] = show["confidence"].round(3)
    st.dataframe(
        show[["timestamp", "plate_text", "camera_id", "camera_name",
              "confidence", "source"]],
        width="stretch", hide_index=True,
    )


def _route_map(stops):
    """Plotly map of a plate's track. Empty-safe: returns False if
    there is nothing to draw (never crashes on empty data)."""
    pts = [s for s in stops if s.get("latitude") is not None
           and s.get("longitude") is not None]
    if not pts:
        return False
    lats = [p["latitude"] for p in pts]
    lons = [p["longitude"] for p in pts]
    labels = [
        f"{i}. {p.get('camera_name', p['camera_id'])} — {p['timestamp']}"
        for i, p in enumerate(pts, 1)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lons, mode="markers+lines+text",
        text=[str(i) for i in range(1, len(pts) + 1)],
        textposition="top center",
        marker=dict(size=14, color="red"),
        line=dict(color="blue", width=4),
        hovertext=labels, hoverinfo="text", name="Route",
    ))
    fig.update_layout(
        map_style="open-street-map",
        map_center=dict(lat=lats[0], lon=lons[0]),
        map_zoom=11, height=450, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    return True


# ============================================================== OVERVIEW
if page == "Overview":
    st.title("Overview")
    stats = overview_stats(region=region_filter)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", stats["total_detections"])
    c2.metric("Unique vehicles", stats["unique_vehicles"])
    c3.metric("Cameras online",
              f"{stats['cameras_online']} / {stats['cameras_total']}")
    c4.metric("Active alerts", stats["active_alerts"])

    df = _detections_df(region=region_filter)
    if df.empty:
        st.info("No detections yet — go to Live Capture and take a "
                "photo of a plate.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Detections per camera")
            per_cam = (
                df.groupby("camera_id").size().reset_index(name="count")
            )
            if per_cam.empty:
                st.info("Nothing to chart yet.")
            else:
                st.plotly_chart(
                    px.bar(per_cam, x="camera_id", y="count"),
                    width="stretch",
                )
        with right:
            st.subheader("Detections by hour")
            per_hour = stats["per_hour"]
            if sum(per_hour.values()) == 0:
                st.info("Nothing to chart yet.")
            else:
                hourly = pd.DataFrame(
                    {"hour": list(per_hour.keys()),
                     "count": list(per_hour.values())}
                )
                st.plotly_chart(
                    px.bar(hourly, x="hour", y="count"),
                    width="stretch",
                )
        st.subheader("Latest detections")
        _show_detections_table(df.head(10))

# ============================================================ DETECTIONS
elif page == "Detections":
    st.title("Detections")
    cameras = list_cameras(region=region_filter)
    cam_options = ["All cameras"] + [c["camera_id"] for c in cameras]
    col_a, col_b = st.columns(2)
    with col_a:
        chosen_cam = st.selectbox("Camera", cam_options)
    with col_b:
        plate_search = st.text_input(
            "Search plate", placeholder="e.g. TS09EX1234")
    camera_id = None if chosen_cam == "All cameras" else chosen_cam
    df = _detections_df(region=region_filter, camera_id=camera_id,
                        plate_query=plate_search or None)
    st.caption(
        f"{len(df)} detection(s) shown. "
        "DEMO = seeded demo sightings · CAPTURED = plates this app "
        "read from a photo."
    )
    _show_detections_table(df)
    if not df.empty:
        st.download_button(
            "Download these detections as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="buildathon_detections.csv",
            mime="text/csv",
        )

# ========================================================== LIVE CAPTURE
elif page == "Live Capture":
    st.title("Live Capture — laptop webcam")
    st.write(
        "Take a photo of a number plate with your laptop webcam, or "
        "upload a photo instead. The app runs YOLOv8 + EasyOCR on it, "
        "checks it is a valid Indian plate, and saves it. "
        "Only plates at confidence 0.5 or higher are saved."
    )
    cameras = list_cameras()
    cam_ids = [c["camera_id"] for c in cameras]
    default_idx = cam_ids.index("CAM01") if "CAM01" in cam_ids else 0
    chosen = st.selectbox(
        "Which camera is this photo from?",
        cam_ids,
        index=default_idx,
        format_func=_camera_label,
    )
    chosen_cam = CAMERA_BY_ID.get(chosen, {})
    if chosen_cam.get("status") == "offline":
        st.warning(
            f"{chosen} ({chosen_cam.get('name')}) is marked offline, "
            "but the photo will still be saved under that camera."
        )

    photo = st.camera_input("Take a photo of a plate")
    upload = st.file_uploader(
        "…or upload a photo of a plate instead",
        type=["jpg", "jpeg", "png"],
    )
    # A fresh photo/upload replaces the previous result; re-running
    # the detection on every Streamlit rerun would re-run YOLO each
    # time, so the result is stored in session state keyed by the
    # photo's hash.
    image_bytes = None
    image_source = None
    if photo is not None:
        image_bytes = photo.getvalue()
        image_source = "webcam photo"
    elif upload is not None:
        image_bytes = upload.getvalue()
        image_source = "uploaded photo"

    if image_bytes:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("capture_hash") != image_hash:
            st.session_state["capture_hash"] = image_hash
            st.session_state["capture_result"] = None
        if st.button("🔍 Detect plate in this photo", type="primary"):
            # Clear any previous result first, so a failed run never
            # leaves a stale success on screen.
            st.session_state["capture_result"] = None
            # NOTE: no st.stop() inside this try block — Streamlit's
            # stop exception must never be caught as a detection error.
            error_msg = None
            found, saved = [], 0
            with st.spinner(
                "Loading the plate model (first time: downloads ~6 MB, "
                "then EasyOCR starts — about 20–30 seconds)…"
            ):
                path, message = ensure_model()
                if path is None:
                    error_msg = message
                else:
                    try:
                        import numpy as np
                        from PIL import Image
                        import cv2

                        try:
                            pil_image = Image.open(
                                io.BytesIO(image_bytes)).convert("RGB")
                        except Exception as exc:  # noqa: BLE001
                            error_msg = (
                                "That photo could not be read. Take "
                                f"the photo again. ({exc})"
                            )
                            pil_image = None
                        if pil_image is not None:
                            frame = cv2.cvtColor(
                                np.array(pil_image), cv2.COLOR_RGB2BGR)
                            found, saved = detect_and_save(frame, chosen)
                    except Exception as exc:  # noqa: BLE001
                        error_msg = f"Detection failed: {exc}"
            if error_msg is not None:
                st.error(error_msg)
            else:
                st.session_state["capture_result"] = {
                    "found": found, "saved": saved,
                    "camera_id": chosen, "source": image_source,
                }
                st.rerun()

        result = st.session_state.get("capture_result")
        if result is not None:
            found = result["found"]
            if not found:
                st.warning(
                    "No number plate found in this photo. Try again "
                    "closer to the plate, straight on, in good light. "
                    "Small, angled or blurry plates are often missed — "
                    "that is a model limit, not a crash."
                )
            for det in found:
                st.divider()
                st.subheader(det["plate_text"] or "(unreadable plate)")
                st.write(
                    f"YOLO confidence: {det['yolo_conf']} · "
                    f"OCR read: `{det['raw_text'] or '(empty)'}` "
                    f"(confidence {det['ocr_conf']}) · "
                    f"Combined confidence: **{det['confidence']}**"
                )
                if det["corrected"]:
                    st.info(
                        "OCR misread a character and it was corrected "
                        f"by position ({det['reason']}). Check the "
                        "plate before trusting it."
                    )
                if det["saved"]:
                    st.success(
                        f"Saved to {result['camera_id']} as a CAPTURED "
                        "detection."
                    )
                    if det["save_info"].get("alert_created"):
                        st.error(
                            "🚨 Watchlist alert! "
                            f"{det['plate_text']} is on the watchlist "
                            "(stolen vehicle) — an alert was created. "
                            "See the Alerts page."
                        )
                elif det["valid"] and det["confidence"] < MIN_CONFIDENCE:
                    st.warning(
                        "Plate read, but confidence is below 0.5, so "
                        "it was not saved."
                    )
                elif not det["valid"]:
                    st.warning(
                        "Plate could not be read as a valid Indian "
                        f"plate ({det['reason']}). It was not saved."
                    )
                else:
                    st.info(
                        "Not saved: "
                        f"{det['save_info'].get('reason', 'unknown reason')}."
                    )
                for line in det.get("skip_log", []):
                    st.caption(f"Detection log: {line}")
            if found and result["saved"] == 0:
                st.caption(
                    "Nothing was saved from this photo. The log lines "
                    "above say why — usually a misread plate or low "
                    "confidence."
                )
    else:
        st.info(
            "No photo yet. Allow camera access in your browser, take "
            "a photo, then tap “Detect plate in this photo”."
        )
        st.session_state["capture_result"] = None
        st.session_state["capture_hash"] = None

# ======================================================= TRACK ANY PLATE
elif page == "Track any plate":
    st.title("Track any plate")
    st.write(
        "Enter any plate to see its track (every camera it was seen "
        "at) and its alerts together — across Odisha and Hyderabad."
    )
    plate_input = st.text_input(
        "Plate number", placeholder="e.g. TS09EX1234",
        value=st.session_state.get("track_plate", ""),
    )
    if st.button("Track plate", type="primary") or plate_input:
        st.session_state["track_plate"] = plate_input
        plate = (plate_input or "").strip().upper()
        if not plate:
            st.info("Type a plate number first.")
        else:
            stops, alerts = track_plate(plate)
            if not stops and not alerts:
                st.info(
                    f"No sightings or alerts for {plate} yet. Check "
                    "the spelling, or capture it on the Live Capture "
                    "page."
                )
            else:
                st.subheader(f"{plate} — {len(stops)} sighting(s)")
                if stops:
                    track_df = pd.DataFrame(stops)
                    st.dataframe(
                        track_df[["timestamp", "camera_id",
                                  "camera_name", "city", "confidence",
                                  "source"]],
                        width="stretch", hide_index=True,
                    )
                    if not _route_map(stops):
                        st.info("No map coordinates for this track.")
                else:
                    st.info("No sightings for this plate yet.")
                st.subheader("Alerts for this plate")
                if alerts:
                    for alert in alerts:
                        st.warning(
                            f"{alert['message']} — "
                            f"camera {alert.get('camera_id') or '—'}, "
                            f"{alert['created_at']}, "
                            f"status: {alert['status']}"
                        )
                else:
                    st.info(
                        f"No alerts for {plate}."
                        + (" This plate is not on the watchlist."
                           if plate not in WATCHLIST else "")
                    )

# =============================================================== CAMERAS
elif page == "Cameras":
    st.title("Cameras — 11 total")
    cameras = list_cameras(region=region_filter)
    if not cameras:
        st.info("No cameras for this region filter.")
    else:
        cam_df = pd.DataFrame(cameras)
        st.dataframe(
            cam_df[["camera_id", "name", "city", "region", "status",
                    "latitude", "longitude"]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "Offline cameras (CAM03 KIIT Square, HYD03 Kukatpally) "
            "are still known cameras — a photo can be saved under "
            "them, they just have no live feed right now."
        )
        pts = [c for c in cameras
               if c.get("latitude") is not None
               and c.get("longitude") is not None]
        if pts:
            fig = go.Figure(go.Scattermap(
                lat=[c["latitude"] for c in pts],
                lon=[c["longitude"] for c in pts],
                mode="markers+text",
                text=[c["camera_id"] for c in pts],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=["green" if c["status"] == "online" else "red"
                           for c in pts]),
                hovertext=[
                    f"{c['camera_id']} — {c['name']}<br>{c['city']} · "
                    f"{c['status']}" for c in pts],
                hoverinfo="text",
            ))
            fig.update_layout(
                map_style="open-street-map",
                map_center=dict(lat=18.5, lon=81.5),
                map_zoom=5, height=500,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, width="stretch")

# ================================================================ ALERTS
elif page == "Alerts":
    st.title("Alerts")
    st.write(
        "Watchlist: **TS08AB7777** (stolen vehicle). Any detection of "
        "a watchlist plate creates an alert here automatically."
    )
    alerts = list_alerts()
    active = [a for a in alerts if a["status"] == "active"]
    st.metric("Active alerts", len(active))
    if not alerts:
        st.info("No alerts yet.")
    else:
        for alert in alerts:
            with st.container(border=True):
                st.write(
                    f"**{alert.get('plate_text') or '—'}** — "
                    f"{alert['message']}"
                )
                st.caption(
                    f"Camera: {alert.get('camera_id') or '—'} · "
                    f"{alert['created_at']} · "
                    f"Status: {alert['status']}"
                )
                if alert["status"] == "active":
                    if st.button(
                        "Acknowledge",
                        key=f"ack-{alert['id']}",
                    ):
                        acknowledge_alert(alert["id"])
                        st.rerun()
