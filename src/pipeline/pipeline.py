"""
M13 — End-to-end integration (Sec 27, Sec 13-16).

Wires: detector.track_frame() -> lane_detector.detect() ->
distance proxy -> MotionTracker -> TTC ->
risk_engine -> decision_engine, draws overlays, and logs a per-frame,
per-object row to CSV so M15 (experimental validation) has raw data to
compute FPS, latency, and scenario outcomes without re-running video.

Run directly:  python -m src.pipeline.pipeline --config config.yaml
"""
import argparse
import os
import sys
import time

import csv
import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.perception.detector import Detector
from src.lane.lane_detector import LaneDetector
from src.motion.motion import MotionTracker, estimate_distance_proxy
from src.ttc.ttc_calc import compute_ttc
from src.traffic.traffic_detector import TrafficDetector
from src.hazard.hazard_detector import HazardDetector
from src.depth.depth_estimator import DepthEstimator
from src.risk.risk_engine import assess_risk
from src.decision.decision_engine import decide

RISK_COLORS = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_overlay(frame, det, distance_m, ttc, risk, action):
    x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
    color = RISK_COLORS.get(risk, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_parts = [f"{det['cls_name']}"]
    if det["id"] is not None:
        label_parts.append(f"ID{det['id']}")
    if distance_m is not None:
        label_parts.append(f"{distance_m:.1f}m*")  # * = unvalidated proxy
    if ttc is not None and ttc != float("inf"):
        label_parts.append(f"TTC{ttc:.1f}s")
    if getattr(det, "in_path", False) or det.get("in_path", False):
        label_parts.append("[IN PATH]")
    label = " | ".join(label_parts)

    cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, action, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run(cfg_path, video_path=None, output_path=None, headless=False, max_frames=0, log_interval=25):
    cfg = load_config(cfg_path)
    detector     = Detector(cfg)
    lane_detector    = LaneDetector(cfg)        # M5
    traffic_detector = TrafficDetector(cfg)    # M6
    hazard_detector  = HazardDetector(cfg)     # M7
    depth_estimator  = DepthEstimator(cfg)     # M9
    motion           = MotionTracker(
        direction_deadband_px_s=cfg.get("motion", {}).get(
            "direction_deadband_px_s", 5.0
        )
    )

    video_path  = video_path  or cfg["pipeline"]["input_video"]
    output_path = output_path or cfg["pipeline"]["output_video"]
    metrics_csv = cfg["pipeline"]["metrics_csv"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or not 0 < fps_in < float("inf"):
        fps_in = 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_csv),  exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps_in, (width, height))

    print(f"============================================================")
    print(f"   ASTRA ADAS — LIVE PIPELINE MONITORING")
    print(f"============================================================")
    print(f"Input video:       {video_path} ({width}x{height} @ {fps_in:.1f} FPS, {total_frames} frames)")
    print(f"Output video:      {output_path}")
    print(f"Metrics CSV:       {metrics_csv}")
    print(f"Frames to process: {max_frames if max_frames > 0 else total_frames}")
    print(f"------------------------------------------------------------")

    rows = []
    frame_idx = 0
    t_start = time.time()
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    action_counts = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break
        frame_idx += 1
        t0 = time.time()
        video_time_s = (frame_idx - 1) / fps_in
        source_frame = frame

        # ── M5: Lane detection ────────────────────────────────────────
        lane_result  = lane_detector.detect(source_frame)
        frame        = lane_detector.draw(frame, lane_result)
        lane_pos     = lane_result.position   # "LEFT"/"CENTER"/"RIGHT"/"UNKNOWN"

        # ── M8: Object detection + tracking ──────────────────────────
        detections  = detector.track_frame(source_frame)
        active_ids  = {d["id"] for d in detections if d["id"] is not None}
        motion.prune(active_ids)

        # ── M9: Depth Map Estimation ─────────────────────────────────
        depth_map = depth_estimator.estimate_depth(source_frame)

        # ── M6: Traffic light / sign recognition ─────────────────────
        traffic_result = traffic_detector.process(source_frame, detections)
        frame          = traffic_detector.draw(frame, traffic_result)
        tl_state       = traffic_result.dominant_light   # RED/YELLOW/GREEN/UNKNOWN
        tl_for_engine  = tl_state if tl_state in ("RED", "YELLOW", "GREEN") else None

        frame_worst_risk = "LOW"
        frame_worst_action = "MAINTAIN SPEED"
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

        for det in detections:
            # Try M9 Depth Map distance first
            distance_m = None
            if depth_map is not None:
                distance_m = depth_estimator.get_object_distance(depth_map, det["xyxy"])
                
            # Fallback to M12 Pinhole Proxy if MiDaS is disabled or fails
            if distance_m is None:
                distance_m = estimate_distance_proxy(cfg, det["cls_name"], det["xyxy"])
                
            x1, y1, x2, y2 = det["xyxy"]
            motion_estimate = motion.observe(
                det["id"], distance_m,
                center_xy=((x1 + x2) / 2, (y1 + y2) / 2),
                timestamp=video_time_s,
            )
            closing_speed = motion_estimate.closing_speed_mps
            ttc           = compute_ttc(cfg, distance_m, closing_speed)
            
            # M7: Determine if object is in our path
            is_in_path = hazard_detector.check_in_path(det, lane_result, width, height)
            det["in_path"] = is_in_path

            risk, risk_reasons = assess_risk(
                cfg, ttc=ttc, distance_m=distance_m, cls_name=det["cls_name"],
                lane_position=lane_pos,
                traffic_light=tl_for_engine,   # ← M6
                in_path=is_in_path,            # ← M7
            )
            action, reasons = decide(
                cfg, risk=risk, ttc=ttc,
                traffic_light=tl_for_engine,   # ← M6
                obstacle_in_path=is_in_path,   # ← M7
                risk_reasons=risk_reasons,
            )

            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1

            if risk_order.get(risk, 0) >= risk_order.get(frame_worst_risk, 0):
                frame_worst_risk = risk
                frame_worst_action = action

            draw_overlay(frame, det, distance_m, ttc, risk, action)

            rows.append({
                "frame": frame_idx, "track_id": det["id"], "class": det["cls_name"],
                "conf": det["conf"], "distance_proxy_m": distance_m,
                "closing_speed_mps": closing_speed,
                "image_velocity_x_px_s": motion_estimate.image_velocity_x_px_s,
                "image_velocity_y_px_s": motion_estimate.image_velocity_y_px_s,
                "image_direction": motion_estimate.image_direction,
                "ttc_s": None if ttc in (None, float("inf")) else ttc,
                "risk": risk, "action": action, "reasons": "; ".join(reasons),
                "lane_position": lane_pos,
                "lane_deviation_px": round(lane_result.lateral_deviation, 1)
                    if lane_result.lateral_deviation is not None else None,
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

        # Draw M9 Depth Map in bottom right corner (picture-in-picture)
        if depth_map is not None:
            depth_color = depth_estimator.apply_colormap(depth_map)
            pip_w = int(width * 0.25)
            pip_h = int(height * 0.25)
            depth_resized = cv2.resize(depth_color, (pip_w, pip_h))
            frame[height-pip_h:height, width-pip_w:width] = depth_resized
            cv2.rectangle(frame, (width-pip_w, height-pip_h), (width, height), (255,255,255), 2)
            cv2.putText(frame, "MiDaS Depth", (width-pip_w+5, height-pip_h+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        latency_ms = (time.time() - t0) * 1000
        cv2.putText(frame, f"Frame {frame_idx} | {latency_ms:.0f} ms/frame",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

        # Real-time console monitoring log
        if frame_idx % log_interval == 0 or frame_idx == 1:
            fps_instant = 1.0 / max(time.time() - t0, 1e-6)
            progress_pct = (frame_idx / (max_frames if max_frames > 0 else total_frames)) * 100
            lane_str = f"Lane:{lane_pos}"
            if lane_result.lateral_deviation is not None:
                lane_str += f"({lane_result.lateral_deviation:+.0f}px)"
            print(f"[MONITOR] Frame {frame_idx:4d} ({progress_pct:4.1f}%) | "
                  f"Latency: {latency_ms:4.0f}ms ({fps_instant:4.1f} FPS) | "
                  f"Objects: {len(detections):2d} | "
                  f"Risk: {frame_worst_risk:6s} | "
                  f"Action: {frame_worst_action:14s} | "
                  f"{lane_str}")

        if not headless:
            cv2.imshow("Astra ADAS - preview (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"------------------------------------------------------------")
    print(f"Processed {frame_idx} frames in {total_time:.1f}s "
          f"({avg_fps:.1f} FPS average) -> {output_path}")

    fieldnames = ["frame", "track_id", "class", "conf", "distance_proxy_m",
                  "closing_speed_mps", "image_velocity_x_px_s",
                  "image_velocity_y_px_s", "image_direction",
                  "ttc_s", "risk", "action", "reasons",
                  "lane_position", "lane_deviation_px",
                  "traffic_light", "traffic_light_count",
                  "traffic_light_states", "signs_detected"]
    with open(metrics_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Per-object metrics ({len(rows)} entries) written to {metrics_csv}")

    print(f"\n================ MONITORING SUMMARY ================")
    print(f"Frames processed:   {frame_idx}")
    print(f"Total detections:   {len(rows)}")
    print(f"Risk breakdown:     {risk_counts}")
    print(f"Actions issued:     {action_counts}")
    print(f"Average throughput: {avg_fps:.1f} FPS")
    print(f"====================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--video",        default=None)
    parser.add_argument("--output",       default=None)
    parser.add_argument("--max-frames",   type=int, default=0,
                        help="Limit processing to N frames (0 for full video).")
    parser.add_argument("--log-interval", type=int, default=25,
                        help="Console monitoring output interval in frames.")
    parser.add_argument("--headless",     action="store_true",
                        help="Disable live cv2 preview window (use for servers).")
    args = parser.parse_args()
    run(args.config, args.video, args.output, args.headless,
        max_frames=args.max_frames, log_interval=args.log_interval)
