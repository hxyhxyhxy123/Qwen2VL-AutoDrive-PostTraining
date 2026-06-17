"""FastAPI service for lightweight cloud video event analysis."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.common import REPO_ROOT, ensure_dir
from src.demo.vlm_frame_analyzer import analyze_frame
from src.demo.yolo_video_event_demo import process_video_with_yolo

try:
    from fastapi import FastAPI, File, UploadFile
except Exception as exc:  # pragma: no cover - depends on local env
    raise RuntimeError("fastapi is required. Install it with: pip install fastapi uvicorn") from exc


app = FastAPI(title="AutoDrive VLM Cloud Event Service", version="0.1.0")
LAST_RESULT = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "autodrive-cloud-yolo-event-service"}


@app.post("/v1/video/analyze")
def analyze_video(
    file: UploadFile = File(...),
    conf: float = 0.28,
    imgsz: int = 640,
    max_seconds: float = 30.0,
    detect_stride: int = 1,
    model: str = "yolov8n.pt",
    device: Optional[str] = None,
):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = ensure_dir(REPO_ROOT / "outputs/demo/service_uploads")
    out_dir = ensure_dir(REPO_ROOT / "outputs/demo/service_runs" / stamp)
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    video_path = upload_dir / f"{stamp}{suffix}"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = process_video_with_yolo(
        video_path=video_path,
        output_dir=out_dir,
        model_name=model,
        device=device,
        conf=conf,
        imgsz=imgsz,
        max_seconds=max_seconds,
        detect_stride=detect_stride,
    )
    global LAST_RESULT
    LAST_RESULT = result
    return result


@app.get("/v1/events/latest")
def latest_events():
    return LAST_RESULT or {"events": [], "summary": None}


@app.post("/v1/vlm/analyze_frame")
def analyze_vlm_frame(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    max_new_tokens: int = 96,
):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = ensure_dir(REPO_ROOT / "outputs/demo/vlm_uploads")
    suffix = Path(file.filename or "frame.jpg").suffix or ".jpg"
    image_path = upload_dir / f"{stamp}{suffix}"
    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return analyze_frame(image_path, prompt=prompt, max_new_tokens=max_new_tokens)
