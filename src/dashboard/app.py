"""
M14 — Dashboard (Sec 14 layout).

Run with:  streamlit run src/dashboard/app.py

Uploads or selects a video, runs the full ADAS pipeline
(Detector + ByteTrack + Lane Detection + Depth Estimator + Traffic Detector + Hazard Detector + TTC + Risk + Decision)
frame-by-frame, and renders live annotated overlays with dynamic telemetry panels.
"""
import csv
import glob
import os
import sys
import tempfile
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.pipeline import load_config, draw_overlay
from src.perception.detector import Detector
from src.dashboard.live_camera import draw_live_detections
from src.lane.lane_detector import LaneDetector
from src.traffic.traffic_detector import TrafficDetector
from src.hazard.hazard_detector import HazardDetector
from src.depth.depth_estimator import DepthEstimator
from src.motion.motion import MotionTracker, estimate_distance_proxy
from src.ttc.ttc_calc import compute_ttc
from src.risk.risk_engine import assess_risk
from src.decision.decision_engine import decide

st.set_page_config(page_title="Astra ADAS", layout="wide", page_icon="🚗")

st.title("🚗 Astra ADAS")
st.caption("Research prototype for Indian Driving Conditions (IDD-D). "
           "Real-time perception, tracking, TTC estimation, hazard assessment, and decision support.")

# Load master config
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
cfg = load_config(CONFIG_PATH)

# Sidebar configuration
st.sidebar.header("⚙️ Pipeline Controls")
page = st.sidebar.radio("Dashboard", ["Video Analysis", "Live Camera"])
conf_threshold = st.sidebar.slider("Detection Confidence", min_value=0.15, max_value=0.80,
                                   value=float(cfg.get("detection", {}).get("conf_threshold", 0.35)), step=0.05)
if st.sidebar.button("↻ Refresh data"):
    st.sidebar.success("Dashboard refreshed.")

st.sidebar.subheader("Detection classes")
st.sidebar.markdown(
    "\n".join(f"- {name}" for name in cfg["detection"]["relevant_classes"])
)
st.sidebar.subheader("IDD-D test benchmark")
st.sidebar.metric("mAP@50", "37.1%")
st.sidebar.metric("Test images", "4,197")
st.sidebar.metric("Detector throughput", "34.5 FPS")
st.sidebar.caption("Saved detector benchmark; not end-to-end video FPS.")
with st.sidebar.expander("System modules", expanded=False):
    st.markdown(
        "- Decision engine\n"
        "- Depth\n"
        "- Evaluation\n"
        "- Hazard\n"
        "- Lane\n"
        "- Motion\n"
        "- Perception\n"
        "- Pipeline\n"
        "- Risk\n"
        "- Tracking\n"
        "- Traffic\n"
        "- TTC"
    )

if page == "Live Camera":
    st.subheader("Live camera object detection")
    st.info(
        "Start the stream and allow camera access in your browser. "
        "Detected objects are outlined and labeled with their class and confidence."
    )

    if (
        "live_camera_detector" not in st.session_state
        or st.session_state.get("live_camera_conf_threshold") != conf_threshold
    ):
        camera_cfg = load_config(CONFIG_PATH)
        camera_cfg["detection"]["conf_threshold"] = conf_threshold
        st.session_state.live_camera_detector = Detector(camera_cfg)
        st.session_state.live_camera_conf_threshold = conf_threshold

    camera_detector = st.session_state.live_camera_detector

    def process_camera_frame(video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        detections = camera_detector.track_frame(frame)
        annotated = draw_live_detections(frame, detections)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="astra-live-camera",
        video_frame_callback=process_camera_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.stop()

source_mode = st.radio(
    "Choose input source:",
    ["Use Existing Project Video", "Upload New Video"],
    horizontal=True,
)
enable_depth = st.sidebar.checkbox("Enable MiDaS Depth (PIP Map)", value=False,
                                   help="Enables monocular neural depth estimation. Note: adds latency on CPU.")
max_frames_to_run = st.sidebar.number_input("Max Frames to Process (0 = All)", min_value=0, max_value=10000, value=300, step=50,
                                            help="Set a limit to quickly test short snippets on CPU.")

cfg["detection"]["conf_threshold"] = conf_threshold
cfg["depth"]["use_midas"] = enable_depth

# Discover available videos
available_videos = sorted(glob.glob(os.path.join(ROOT_DIR, "outputs", "videos", "*.mp4")))
video_options = {os.path.basename(p): p for p in available_videos}

video_path_to_run = None
is_temp_file = False
video_name = None

if source_mode == "Use Existing Project Video":
    if video_options:
        selected_name = st.selectbox("Select a video from outputs/videos/:", list(video_options.keys()))
        video_path_to_run = video_options[selected_name]
        video_name = selected_name
        st.info(f"Selected: `{video_path_to_run}`")
    else:
        st.warning("No existing .mp4 videos found in `outputs/videos/`. Please upload a video.")
else:
    uploaded = st.file_uploader("Upload a road video (mp4, avi, mov):", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path_to_run = tmp.name
        video_name = uploaded.name
        is_temp_file = True

col_video, col_metrics = st.columns([2, 1])
video_slot = col_video.empty()
metrics_slot = col_metrics.empty()
status_slot = st.empty()

start_btn = st.button("🚀 Run ADAS Pipeline", type="primary", disabled=(video_path_to_run is None))

if start_btn and video_path_to_run is not None:
    status_slot.info("Initializing models and starting video inference...")

    detector = Detector(cfg)
    lane_detector = LaneDetector(cfg)
    traffic_detector = TrafficDetector(cfg)
    hazard_detector = HazardDetector(cfg)
    depth_estimator = DepthEstimator(cfg)
    motion = MotionTracker(
        direction_deadband_px_s=cfg.get("motion", {}).get(
            "direction_deadband_px_s", 5.0
        )
    )

    cap = cv2.VideoCapture(video_path_to_run)
    if not cap.isOpened():
        status_slot.error(f"Could not open video file: {video_path_to_run}")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if not fps_in or not 0 < fps_in < float("inf"):
            fps_in = 30

        source_name = os.path.splitext(os.path.basename(video_name))[0]
        run_name = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        output_dir = os.path.join(ROOT_DIR, "outputs", "dashboard_runs", run_name)
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{source_name}_annotated.mp4")
        metrics_csv_path = os.path.join(output_dir, f"{source_name}_metrics.csv")
        output_container = av.open(output_video_path, mode="w")
        output_stream = output_container.add_stream(
            "libx264", rate=Fraction(fps_in).limit_denominator(100_000)
        )
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "ultrafast", "crf": "23"}

        frame_idx = 0
        t_prev = time.time()
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        metric_rows = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames_to_run > 0 and frame_idx >= max_frames_to_run:
                break
            frame_idx += 1

            t0 = time.time()
            video_time_s = (frame_idx - 1) / fps_in
            source_frame = frame.copy()

            # 1. Lane detection (M5)
            lane_result = lane_detector.detect(source_frame)
            frame = lane_detector.draw(frame, lane_result)
            lane_pos = lane_result.position

            # 2. Object detection + ByteTrack tracking (M3/M4/M8)
            detections = detector.track_frame(source_frame)
            active_ids = {d["id"] for d in detections if d["id"] is not None}
            motion.prune(active_ids)

            # 3. Depth Map (M9) - if enabled
            depth_map = depth_estimator.estimate_depth(source_frame) if enable_depth else None

            # 4. Traffic light & sign recognition (M6)
            traffic_result = traffic_detector.process(source_frame, detections)
            frame = traffic_detector.draw(frame, traffic_result)
            tl_state = traffic_result.dominant_light
            tl_for_engine = tl_state if tl_state in ("RED", "YELLOW", "GREEN") else None

            panel_rows = []
            worst_risk, worst_action = "LOW", "MAINTAIN SPEED"

            for det in detections:
                distance_m = None
                if depth_map is not None:
                    distance_m = depth_estimator.get_object_distance(depth_map, det["xyxy"])

                if distance_m is None:
                    distance_m = estimate_distance_proxy(cfg, det["cls_name"], det["xyxy"])

                x1, y1, x2, y2 = det["xyxy"]
                motion_estimate = motion.observe(
                    det["id"], distance_m,
                    center_xy=((x1 + x2) / 2, (y1 + y2) / 2),
                    timestamp=video_time_s,
                )
                closing_speed = motion_estimate.closing_speed_mps
                ttc = compute_ttc(cfg, distance_m, closing_speed)

                is_in_path = hazard_detector.check_in_path(det, lane_result, width, height)
                det["in_path"] = is_in_path

                risk, risk_reasons = assess_risk(
                    cfg, ttc=ttc, distance_m=distance_m, cls_name=det["cls_name"],
                    lane_position=lane_pos, traffic_light=tl_for_engine, in_path=is_in_path,
                )
                action, reasons = decide(cfg, risk=risk, ttc=ttc, traffic_light=tl_for_engine,
                                         obstacle_in_path=is_in_path, risk_reasons=risk_reasons)

                draw_overlay(frame, det, distance_m, ttc, risk, action)
                metric_rows.append({
                    "frame": frame_idx,
                    "track_id": det["id"],
                    "class": det["cls_name"],
                    "conf": det["conf"],
                    "distance_proxy_m": distance_m,
                    "closing_speed_mps": closing_speed,
                    "image_velocity_x_px_s": motion_estimate.image_velocity_x_px_s,
                    "image_velocity_y_px_s": motion_estimate.image_velocity_y_px_s,
                    "image_direction": motion_estimate.image_direction,
                    "ttc_s": None if ttc in (None, float("inf")) else ttc,
                    "risk": risk,
                    "action": action,
                    "reasons": "; ".join(reasons),
                    "lane_position": lane_pos,
                    "lane_deviation_px": (
                        round(lane_result.lateral_deviation, 1)
                        if lane_result.lateral_deviation is not None else None
                    ),
                    "traffic_light": tl_state,
                    "traffic_light_count": len(traffic_result.lights),
                    "traffic_light_states": "; ".join(
                        f"ID {light.track_id}: {light.state} "
                        f"(color {light.confidence:.0%}, "
                        f"det {light.detection_confidence:.0%})"
                        for light in traffic_result.lights
                    ),
                    "signs_detected": traffic_result.signs_detected,
                })

                if risk_order[risk] >= risk_order[worst_risk]:
                    worst_risk, worst_action = risk, action

                in_path_badge = "⚠️ **[IN PATH]** " if is_in_path else ""
                dist_str = f"{distance_m:.1f}m*" if distance_m else "N/A"
                ttc_str = f"{ttc:.1f}s" if (ttc and ttc != float("inf")) else "N/A"
                speed_str = (
                    f"{closing_speed:+.1f}m/s*"
                    if closing_speed is not None else "N/A"
                )

                panel_rows.append(
                    f"- **{det['cls_name']}** (ID {det['id']}): {dist_str} | "
                    f"Range-rate proxy: {speed_str} | Direction: "
                    f"`{motion_estimate.image_direction}` (image) | TTC: {ttc_str} | "
                    f"{in_path_badge}Risk: `{risk}` | Action: `{action}`"
                )

            # Picture-in-picture depth map
            if depth_map is not None:
                depth_color = depth_estimator.apply_colormap(depth_map)
                pip_w = int(width * 0.25)
                pip_h = int(height * 0.25)
                depth_resized = cv2.resize(depth_color, (pip_w, pip_h))
                frame[height-pip_h:height, width-pip_w:width] = depth_resized
                cv2.rectangle(frame, (width-pip_w, height-pip_h), (width, height), (255, 255, 255), 2)
                cv2.putText(frame, "MiDaS Depth", (width-pip_w+5, height-pip_h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in output_stream.encode(video_frame):
                output_container.mux(packet)
            fps_display = 1.0 / max(time.time() - t_prev, 1e-6)
            t_prev = time.time()

            # Render live frame
            video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            lane_dev_str = f"{lane_result.lateral_deviation:+.1f} px" if lane_result.lateral_deviation is not None else "N/A"
            total_str = f" / {total_video_frames}" if total_video_frames > 0 else ""

            # Risk color styling
            risk_color_badge = "🔴" if worst_risk == "HIGH" else ("🟡" if worst_risk == "MEDIUM" else "🟢")
            action_badge = "🛑" if worst_action in ("BRAKE", "STOP") else ("⚠️" if worst_action in ("SLOW DOWN", "AVOID OBSTACLE") else "✅")

            metrics_slot.markdown(
                f"### Frame {frame_idx}{total_str}\n"
                f"⚡ **FPS:** `{fps_display:.1f}`\n\n"
                f"🛡️ **Overall Risk:** {risk_color_badge} **{worst_risk}**\n\n"
                f"📢 **Action Command:** {action_badge} **{worst_action}**\n\n"
                f"---\n"
                f"🛣️ **Lane Position:** `{lane_pos}` (Dev: `{lane_dev_str}`)\n\n"
                f"🚦 **Traffic Light:** `{tl_state}`\n\n"
                f"🚦 **Signal Count:** `{len(traffic_result.lights)}`"
                + (
                    " — " + "; ".join(
                        f"ID {light.track_id}: {light.state} "
                        f"(color {light.confidence:.0%}, "
                        f"det {light.detection_confidence:.0%})"
                        for light in traffic_result.lights
                    )
                    if traffic_result.lights else ""
                )
                + "\n\n"
                f"🚸 **Signs Count:** `{traffic_result.signs_detected}`\n\n"
                f"---\n"
                f"👁️ **Tracked Objects ({len(detections)}):**\n\n" +
                ("\n".join(panel_rows[:10]) if panel_rows else "*No objects in field of view*") +
                "\n\n*Speed is relative closing-speed estimated from uncalibrated "
                "distance; direction is image-plane motion (not compass/world direction).*"
            )

        cap.release()
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        if is_temp_file and os.path.exists(video_path_to_run):
            os.unlink(video_path_to_run)
        fieldnames = [
            "frame", "track_id", "class", "conf", "distance_proxy_m",
            "closing_speed_mps", "image_velocity_x_px_s", "image_velocity_y_px_s",
            "image_direction", "ttc_s", "risk", "action", "reasons",
            "lane_position", "lane_deviation_px", "traffic_light",
            "traffic_light_count", "traffic_light_states", "signs_detected",
        ]
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as metrics_file:
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            metrics_writer.writeheader()
            metrics_writer.writerows(metric_rows)

        status_slot.success(
            f"Processing finished. Evaluated {frame_idx} frames and saved "
            f"{len(metric_rows)} object records."
        )
        st.subheader("Annotated video")
        st.video(Path(output_video_path).read_bytes(), format="video/mp4")
        st.download_button(
            "Download annotated video",
            data=Path(output_video_path).read_bytes(),
            file_name=os.path.basename(output_video_path),
            mime="video/mp4",
        )
        st.download_button(
            "Download per-object metrics (CSV)",
            data=Path(metrics_csv_path).read_bytes(),
            file_name=os.path.basename(metrics_csv_path),
            mime="text/csv",
        )
        st.caption(f"Saved outputs: `{output_video_path}` and `{metrics_csv_path}`")
