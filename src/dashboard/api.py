"""FastAPI service for the Astra ADAS web dashboard."""
import base64
import csv
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.pipeline import load_config, run as run_pipeline
from src.perception.detector import Detector
from src.dashboard.live_camera import draw_live_detections

LOGGER = logging.getLogger("astra_adas.api")
CONFIG_PATH = ROOT_DIR / "config.yaml"
VIDEO_DIR = ROOT_DIR / "outputs" / "videos"
RUNS_DIR = ROOT_DIR / "outputs" / "dashboard_runs"
UPLOADS_DIR = ROOT_DIR / "outputs" / "dashboard_uploads"
WEB_DIST_DIR = ROOT_DIR / "web" / "dist"
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_LIVE_FRAME_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Astra ADAS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_jobs = {}
_jobs_lock = threading.Lock()
_detector = None
_detector_lock = threading.Lock()


def _video_path(video_name: str) -> Path:
    if Path(video_name).name != video_name:
        raise HTTPException(status_code=400, detail="Invalid video name.")
    candidate = (VIDEO_DIR / video_name).resolve()
    if candidate.parent != VIDEO_DIR.resolve() or candidate.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported video source.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Project video not found.")
    return candidate


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload an MP4, AVI, or MOV video.")

    written = 0
    try:
        with destination.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    target.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="Video upload exceeds the 500 MB limit.",
                    )
                target.write(chunk)
    finally:
        await upload.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")


def _process_run(run_id: str, video_path: Path, source_name: str, options: dict) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem).strip("._")
    safe_stem = safe_stem or "road_video"
    output_path = run_dir / f"{safe_stem}_annotated.mp4"
    metrics_path = run_dir / f"{safe_stem}_metrics.csv"
    config_path = run_dir / "run_config.yaml"

    try:
        cfg = load_config(str(CONFIG_PATH))
        cfg["detection"]["conf_threshold"] = options["confidence"]
        cfg["depth"]["use_midas"] = options["enable_depth"]
        cfg["pipeline"]["output_video"] = str(output_path)
        cfg["pipeline"]["metrics_csv"] = str(metrics_path)
        with config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(cfg, config_file, sort_keys=False)

        with _jobs_lock:
            _jobs[run_id]["status"] = "running"
            _jobs[run_id]["started_at"] = datetime.now().isoformat(timespec="seconds")

        run_pipeline(
            str(config_path),
            video_path=str(video_path),
            output_path=str(output_path),
            headless=True,
            max_frames=options["max_frames"],
            log_interval=50,
        )

        with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
            record_count = max(0, sum(1 for _ in csv.DictReader(metrics_file)))
        with _jobs_lock:
            _jobs[run_id].update({
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "video_url": f"/api/runs/{run_id}/video",
                "metrics_url": f"/api/runs/{run_id}/metrics",
                "record_count": record_count,
            })
    except Exception as exc:
        LOGGER.exception("Video run %s failed", run_id)
        with _jobs_lock:
            _jobs[run_id].update({
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
            })
    finally:
        if video_path.parent == UPLOADS_DIR.resolve():
            video_path.unlink(missing_ok=True)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "Astra ADAS",
        "checkpoint_available": (ROOT_DIR / "models" / "detection" / "best.pt").is_file(),
    }


@app.get("/api/videos")
def list_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for path in sorted(VIDEO_DIR.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
            videos.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "url": f"/api/videos/{path.name}/file",
            })
    return {"videos": videos}


@app.get("/api/videos/{video_name}/file")
def get_video(video_name: str):
    path = _video_path(video_name)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/api/runs", status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
    video_name: Optional[str] = Form(default=None),
    upload: Optional[UploadFile] = File(default=None),
    confidence: float = Form(default=0.22, ge=0.15, le=0.80),
    enable_depth: bool = Form(default=False),
    max_frames: int = Form(default=300, ge=0, le=10000),
):
    if (video_name is None) == (upload is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one project video name or uploaded video.",
        )

    run_id = uuid.uuid4().hex
    if upload is not None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "").suffix.lower()
        video_path = (UPLOADS_DIR / f"{run_id}{suffix}").resolve()
        await _save_upload(upload, video_path)
        source_name = Path(upload.filename or "road_video.mp4").name
    else:
        video_path = _video_path(video_name)
        source_name = video_path.name

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": run_id,
        "source_name": source_name,
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "record_count": None,
        "video_url": None,
        "metrics_url": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[run_id] = job
    background_tasks.add_task(
        _process_run,
        run_id,
        video_path,
        source_name,
        {
            "confidence": confidence,
            "enable_depth": enable_depth,
            "max_frames": max_frames,
        },
    )
    return job


@app.get("/api/runs")
def list_runs():
    with _jobs_lock:
        runs = sorted(
            (dict(job) for job in _jobs.values()),
            key=lambda job: job["created_at"],
            reverse=True,
        )
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return dict(job)


def _run_output(run_id: str, suffix: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID.")
    run_dir = (RUNS_DIR / run_id).resolve()
    if run_dir.parent != RUNS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid run path.")
    matches = list(run_dir.glob(f"*_{suffix}"))
    if len(matches) != 1 or not matches[0].is_file():
        raise HTTPException(status_code=404, detail="Run output is not available.")
    return matches[0]


@app.get("/api/runs/{run_id}/video")
def download_run_video(run_id: str):
    path = _run_output(run_id, "annotated.mp4")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/runs/{run_id}/metrics")
def download_run_metrics(run_id: str):
    path = _run_output(run_id, "metrics.csv")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.post("/api/live/detect")
async def detect_live_frame(request: Request, confidence: float = 0.22):
    if not 0.15 <= confidence <= 0.80:
        raise HTTPException(status_code=422, detail="Confidence must be between 0.15 and 0.80.")
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_LIVE_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Camera frame exceeds the 8 MB limit.")
    frame_bytes = await request.body()
    if not frame_bytes or len(frame_bytes) > MAX_LIVE_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Camera frame is empty or exceeds the 8 MB limit.")
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Camera frame is not a valid image.")

    global _detector
    with _detector_lock:
        if _detector is None:
            cfg = load_config(str(CONFIG_PATH))
            _detector = Detector(cfg)
        _detector.conf = confidence
        _detector.track_conf = min(
            confidence,
            min(_detector.class_conf_thresholds.values(), default=confidence),
        )
        detections = _detector.track_frame(frame)
        annotated = draw_live_detections(frame.copy(), detections)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode annotated camera frame.")

    return {
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "detections": [
            {
                "class_name": detection["cls_name"],
                "confidence": float(detection["conf"]),
                "track_id": detection["id"],
            }
            for detection in detections
        ],
    }


if WEB_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST_DIR, html=True), name="web")
