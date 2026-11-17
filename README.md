# Astra ADAS

This repository contains an end-to-end research prototype for video-based
road-scene perception and risk estimation. The pipeline is runnable, but its
distance estimates and recommendations are not validated for real-vehicle use.

## What currently runs

```
video -> 15-class YOLO detection -> ByteTrack IDs
       -> lane / traffic / hazard heuristics
       -> MiDaS relative depth (or a bounding-box proxy)
       -> relative motion -> TTC -> rule-based risk and decision
       -> annotated video + CSV metrics + Streamlit dashboard
```

The current detector loads `models/detection/best.pt`. The lane detector is a
classical OpenCV Canny/Hough implementation; traffic lights are classified
with HSV color thresholds, and traffic signs are counted but not
subclassified. A detected light is not associated with the ego lane, so a
signal-state estimate is context telemetry rather than a reliable stop/go
decision. Hazard assessment uses a geometric corridor. MiDaS depth is relative,
and its configured scale factor is not camera-calibrated. These are prototype
heuristics, not trained or validated ADAS capabilities.

The dashboard and video pipeline report traffic-light count, each detected
light's color estimate/confidence, sign count, per-track relative closing
speed, and image-plane direction. Speed/TTC still depend on the uncalibrated
distance estimate; image-plane direction is left/right/up/down in the video,
not compass direction or the object's real-world heading. These telemetry
fields are diagnostic outputs, not validated measurements.

The existing checkpoint was evaluated against the COCO-pretrained baseline
on the full IDD-D test split. See the latest result in
`outputs/evaluation/m4_benchmark_test/m4_benchmark_report.md`. A completed
YOLOv8s M5 training run is still required; the presence of `best.pt` alone
does not mean that training run finished.

## Block-diagram requirement checklist

Status is against the attached end-to-end architecture diagram. "Partial"
means a prototype path exists but the behavior is not trained, calibrated,
or validated to the diagram's implied operational level.

| Diagram block | Status | Current implementation / remaining work |
|---|---|---|
| 1. Input source | Partial | Uploaded and project video files work, with browser webcam input available in the dashboard. CARLA and other live-stream input are not wired. |
| 2. Image acquisition | Partial | OpenCV reads video frames; the dashboard receives live browser camera frames over WebRTC. No explicit capture/stream diagnostics. |
| 3. Image preprocessing | Partial | YOLO performs its internal resize/normalization. Inference defaults to 640px to improve small traffic-signal/sign recall; 416px is available for CPU throughput. Explicit brightness, denoise, contrast, and normalization stages are not implemented. |
| 4.1 Lane detection | Partial | OpenCV Canny/Hough lane lines and lateral offset are heuristic; no labeled lane benchmark, departure warning validation, or lane-keeping control. |
| 4.2 Vehicle/object detection | Partial | IDD 15-class YOLO checkpoint; see the benchmark results below. Small/rare-class recall remains weak. |
| 4.3 Pedestrian detection | Partial | `person` and `rider` classes use the same YOLO detector; no separate pedestrian/cyclist-specific validation. |
| 4.4 Traffic-light detection | Partial | Lower class-specific confidence threshold, raw-box recovery, short-term box association, and HSV state estimate; ego-lane signal association and labeled state accuracy are missing. |
| 4.5 Traffic-sign recognition | Partial | Lower class-specific confidence threshold and presence/count output; stop/yield/speed-limit/type recognition is not implemented. |
| 4.6 Road-hazard detection | Partial | Object boxes and a geometric in-path corridor. Potholes, debris, construction and hazard classes need data/model support and validation. |
| Models: YOLOv8 / OpenCV | Partial | YOLOv8 IDD checkpoint and OpenCV heuristics run. No completed M5 training run is present in `runs/detect/astra_m5_yolov8s/`. |
| Models: DeepSORT | Not implemented | Tracking is Ultralytics ByteTrack; DeepSORT is only a placeholder. |
| Models: MiDaS | Partial | Optional relative-depth map; configured scale is not calibrated to meters. |
| 5. Object tracking | Partial | ByteTrack IDs and per-track image motion work; weak signal/sign boxes use short-term IoU association. No labeled ID-switch, MOTA, or IDF1 validation. |
| 6. Distance/risk assessment | Partial | Distance proxy, range-rate proxy, TTC, and rule-based risk run. Metric calibration and ground-truth validation are outstanding. |
| 7. Road-scene understanding | Partial | Lane, object, signal, and hazard outputs are combined as rule inputs; no learned scene/context model. |
| 8. Decision-making engine | Partial | Explainable rule-based recommendations exist. No validated driving policy or vehicle-control connection. |
| 9. Driving assistance/recommendations | Partial | Text/video actions and risk labels are shown. Turn signals, stay-in-lane logic, collision audio alarms, and safe-driving validation are absent. |
| 10. Dashboard/visualization | Partial | Streamlit video telemetry, annotated output, downloads, CSV logs, and live webcam object overlays are available. No validated real-time alert/stream reliability guarantees. |
| Feedback loop | Not implemented | No operator corrections, event feedback, or retraining loop is collected from dashboard results. |

### Detection-resolution experiment

To improve distant traffic-signal/sign recall without claiming that the model
itself was retrained, inference defaults to 640px, with lower class-specific
confidence cutoffs for those two classes. Other classes keep the configured
0.22 cutoff. The tracker preserves raw signal/sign boxes that do not receive a ByteTrack
ID and assigns them short-term box-association IDs. Lane detection resizes
large frames to 640px, smooths brief detection dropouts, and measures lateral
offset at the bottom of the lane boundaries.
On the same evenly sampled 200-image IDD-D
**validation** subset, the existing IDD checkpoint measured:

| Inference size | mAP@50 | Indian-specific mAP@50 | Mean latency | Throughput |
|---:|---:|---:|---:|---:|
| 640px | 0.5045 | 0.4870 | 34.3 ms/image | 29.2 FPS |
| 960px | 0.5602 | 0.5194 | 65.4 ms/image | 15.3 FPS |

The 640px setting measured 29.2 FPS for detection on CPU; the full pipeline
adds processing overhead, and MiDaS depth is therefore off by default
(enable `depth.use_midas` to trade throughput for relative-depth estimates).
Use `detection.imgsz: 416` for lower CPU latency or `960` for higher
small-object recall. These are validation-set resolution comparisons, not
held-out test results. The 960px setting improved mAP@50 by 0.0557 on this
sample but roughly halved CPU detector throughput. Detailed paired reports are in
`outputs/evaluation/m4_resolution_640/` and
`outputs/evaluation/m4_resolution_960/`. Reproduce the full paired benchmark
at 960px with:

```bash
python -m src.evaluation.benchmark_m4 --split val --num-images all --imgsz 960 --out-dir outputs/evaluation/m4_val_960
```

Compare against an identical run at `--imgsz 640`. Preserve the test split for
final model selection. Current weak areas in the existing full-test report
include animal recall (0.113), traffic-light recall (0.230), traffic-sign
recall (0.290), and vehicle-fallback recall (0.086); rare classes have very
few examples. Further model improvement needs completed training, per-class
sampling/augmentation experiments on train/validation data, and final
held-out evaluation.

## 1. Install

```bash
python -m venv venv && source venv/bin/activate   # or your preferred env
pip install -r requirements.txt
```

Resources needed:
- Python 3.10+
- A GPU is optional for M3 baseline inference (CPU works, just slower);
  a GPU (Colab or local) becomes necessary for M3/M4 **training**.
- Model checkpoints are not included in this code-only repository. Provide
  `models/detection/best.pt` for the default detector; supply `yolov8n.pt` if
  you use a configuration or script that references it.
- ~2 GB disk for model checkpoints and working video files.
- The IDD dataset (Sec 17) only when you get to M2/M3 training — not needed
  to run the prototype on COCO classes today.
- Select **Live Camera** in the dashboard and allow browser camera access to
  run object detection on your webcam.

## 2. Run the pipeline on any road video

```bash
python -m src.pipeline.pipeline --video path/to/your_road_video.mp4 --headless
```

Outputs:
- `outputs/videos/annotated_output.mp4` — boxes, IDs, distance*, TTC, action
- `outputs/metrics/run_metrics.csv` — one row per object per frame (this is
  your raw data for M15 experiments: FPS, per-scenario outcome tables, etc.)
- The CSV also records per-frame signal counts/states and image-plane motion;
  distance and closing-speed values remain uncalibrated proxies.

`*` = distance is the temporary unvalidated proxy (Sec 12) — displayed with
a `*` on purpose so it's never mistaken for a calibrated measurement.

## 3. Run the Streamlit dashboard (M14)

From the project root, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run src/dashboard/app.py --server.port 8501
```

Open `http://localhost:8501/`. Use the sidebar to switch between **Video
Analysis** and **Live Camera**, adjust detection confidence, review the
configured detection classes and benchmark metrics, or refresh the project
video list. Live Camera uses the browser webcam and displays YOLO detections
and track IDs. Video analysis shows annotated frames and telemetry while it
processes, then offers an H.264 MP4 player and CSV/video downloads.

The React/FastAPI version remains in `web/` as an optional alternative.

## 4. Dataset inspection

```bash
python scripts/inspect_idd.py --root data/idd/raw
```

For raw IDD data, reports image/annotation counts, pairing gaps,
duplicate/corrupt images, and a sample annotation structure. The supplied
YOLO-format dataset is already configured in
`IDDDetectionsYOLODataset/data.yaml`.

## 5. Where each spec section lives in code

| Spec section | File |
|---|---|
| Sec 7 Object detection (M3) | `src/perception/detector.py` |
| Sec 11 Tracking (M8) | `Detector.track_frame()` (ByteTrack) + `src/tracking/tracker.py` (DeepSORT stub) |
| Sec 8 Lane heuristic (M5) | `src/lane/lane_detector.py` |
| Sec 9 Traffic light / sign presence (M6) | `src/traffic/traffic_detector.py` |
| Sec 10 Hazard corridor (M7) | `src/hazard/hazard_detector.py` |
| Sec 12 Distance proxy (M9) | `src/motion/motion.py::estimate_distance_proxy` |
| MiDaS relative depth (not calibrated) | `src/depth/depth_estimator.py` |
| Sec 13 Relative velocity | `src/motion/motion.py::MotionTracker` |
| Sec 14 TTC (M10) | `src/ttc/ttc_calc.py` |
| Sec 15 Risk engine (M11) | `src/risk/risk_engine.py` |
| Sec 16 Decision engine (M12) | `src/decision/decision_engine.py` |
| Sec 13/27 End-to-end + logging (M13) | `src/pipeline/pipeline.py` |
| Sec 26 Streamlit dashboard (M14) | `src/dashboard/app.py` |
| Optional dashboard API | `src/dashboard/api.py` |
| Optional React dashboard UI | `web/src/App.jsx` |
| Sec 18/35 Dataset integrity (M2) | `scripts/inspect_idd.py` |

## 6. Honest next steps, in order (matches your Sec 41)

1. **Finish M5 training on a GPU runtime** using
   `scripts/astra_m5_colab_training.ipynb` or `scripts/train_improved.py`.
   The current local Python environment is CPU-only, so a full 50-epoch,
   960px run should be done in Colab/Kaggle rather than locally.
2. **Compare checkpoints on validation data**, then evaluate the selected
   checkpoint on the held-out test split with:
   `python -m src.evaluation.benchmark_m4 --num-images all --split test`.
   The current checkpoint's report is in
   `outputs/evaluation/m4_benchmark_test/`.
3. **Improve M5–M7 beyond the current heuristics** only when suitable
   labeled lane and traffic-light/sign data is available. Sign-type
   classification and ego-lane signal association are not implemented.
   Validate light-state classification against labeled day/night/weather
   footage before treating its color estimates as reliable.
4. **Complete M9/M10 validation** with camera calibration or metric-depth
   ground truth and labeled motion scenarios. Do not treat the current
   MiDaS-scaled values as measured meters or the resulting TTC as validated.
5. **Complete M15 experiments** across multiple videos and labeled scenarios.
   Tracking MOTA/IDF1 and TTC/risk accuracy require corresponding ground truth;
   pipeline CSVs alone are not sufficient.

## Safety note (carried from the spec, Sec 1 / Sec 16 / Sec 38)

This is a research/ADAS prototype. It does not control a real vehicle, its
TTC/distance are not certified measurements until M9/M10 validation is
complete, and its recommendations are prototype outputs, not control
commands.
