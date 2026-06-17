"""YOLO-based real video event demo.

This is the practical deployment demo path:

High-frequency lightweight CV detector (YOLO) draws boxes and triggers events;
low-frequency VLM/ORPO analysis can later consume selected key frames. The VLM
is not used as a detector.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common import REPO_ROOT, ensure_dir


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
TRAFFIC_CLASSES = {"traffic light", "stop sign"}
KEEP_CLASSES = VEHICLE_CLASSES | TRAFFIC_CLASSES | {"person"}


@dataclass
class Detection:
    cls: str
    conf: float
    box: Tuple[int, int, int, int]


@dataclass
class Event:
    timestamp: float
    frame_id: int
    trigger: str
    event_type: str
    risk_level: str
    risk_object: str
    evidence: str
    suggested_action: str
    latency_ms: int
    model: str
    detections: List[Dict[str, Any]]


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("opencv-python is required. Install it with: pip install opencv-python") from exc


def _load_yolo(model_name: str):
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO(model_name)
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("ultralytics is required. Install it with: pip install ultralytics") from exc


def _default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _color_for(cls: str, risk: str = "low") -> Tuple[int, int, int]:
    if risk == "high":
        return (40, 40, 230)
    if risk == "medium":
        return (35, 140, 240)
    if cls == "person":
        return (80, 220, 120)
    if cls in VEHICLE_CLASSES:
        return (255, 180, 50)
    if cls in TRAFFIC_CLASSES:
        return (230, 120, 255)
    return (200, 200, 200)


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    cv2 = _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, frame.shape[1] - tw - 8))
    y = max(th + 8, min(y, frame.shape[0] - 4))
    cv2.rectangle(frame, (x, y - th - baseline - 5), (x + tw + 8, y + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _detections_to_event(
    detections: List[Detection],
    width: int,
    height: int,
) -> Tuple[str, str, str, str, str, str]:
    front_vehicle = []
    pedestrians = []
    traffic = []

    for det in detections:
        x1, y1, x2, y2 = det.box
        cx = (x1 + x2) / 2 / max(width, 1)
        cy = (y1 + y2) / 2 / max(height, 1)
        area = (x2 - x1) * (y2 - y1) / max(width * height, 1)
        in_attention_zone = 0.22 <= cx <= 0.78 and cy >= 0.42
        if det.cls in VEHICLE_CLASSES and in_attention_zone and area >= 0.018:
            front_vehicle.append(det)
        if det.cls == "person" and in_attention_zone and area >= 0.004:
            pedestrians.append(det)
        if det.cls in TRAFFIC_CLASSES:
            traffic.append(det)

    if pedestrians:
        return (
            "pedestrian_attention",
            "pedestrian_risk",
            "medium",
            "person",
            f"{len(pedestrians)} pedestrian(s) detected in the forward attention zone.",
            "slow_down",
        )
    if front_vehicle:
        return (
            "front_vehicle_attention",
            "front_vehicle_close",
            "medium",
            "vehicle",
            f"{len(front_vehicle)} vehicle-like object(s) detected near the ego-lane attention zone.",
            "slow_down",
        )
    if len(detections) >= 6:
        return (
            "dense_traffic",
            "traffic_attention",
            "low",
            "multiple_objects",
            f"{len(detections)} relevant objects detected in the scene.",
            "keep",
        )
    if traffic:
        return (
            "traffic_control",
            "traffic_signal_attention",
            "low",
            "traffic_light_or_sign",
            "Traffic control object detected.",
            "keep",
        )
    return ("safe", "safe", "low", "none", "No high-confidence traffic risk trigger.", "keep")


def _run_detector(model, frame: np.ndarray, conf: float, imgsz: int, device: str) -> List[Detection]:
    result = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    names = result.names
    detections: List[Detection] = []
    if result.boxes is None:
        return detections
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls_ids = result.boxes.cls.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    for box, cls_id, score in zip(xyxy, cls_ids, confs):
        cls_name = str(names[int(cls_id)])
        if cls_name not in KEEP_CLASSES:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        detections.append(Detection(cls=cls_name, conf=float(score), box=(x1, y1, x2, y2)))
    return detections


def process_video_with_yolo(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = "yolov8n.pt",
    device: Optional[str] = None,
    conf: float = 0.28,
    imgsz: int = 640,
    max_seconds: float = 30.0,
    detect_stride: int = 1,
    cooldown_seconds: float = 1.2,
) -> Dict[str, Any]:
    cv2 = _require_cv2()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    device = device or _default_device()
    model = _load_yolo(model_name)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) if output_dir else REPO_ROOT / "outputs" / "demo" / f"yolo_video_event_demo_{stamp}"
    ensure_dir(output_dir)
    output_video = output_dir / "annotated_yolo_video.mp4"
    events_json = output_dir / "events.json"
    summary_json = output_dir / "summary.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid dimensions.")

    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    max_frames = int(max_seconds * fps) if max_seconds > 0 else 10**12
    frame_id = 0
    last_event_t = -10**9
    last_dets: List[Detection] = []
    last_status = "safe | low | keep"
    events: List[Event] = []
    detector_times: List[float] = []
    start_wall = time.perf_counter()

    while frame_id < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_id / fps

        if frame_id % max(detect_stride, 1) == 0:
            t0 = time.perf_counter()
            last_dets = _run_detector(model, frame, conf=conf, imgsz=imgsz, device=device)
            det_ms = (time.perf_counter() - t0) * 1000
            detector_times.append(det_ms)
            trigger, event_type, risk, risk_object, evidence, action = _detections_to_event(last_dets, width, height)
            if trigger != "safe" and timestamp - last_event_t >= cooldown_seconds:
                event = Event(
                    timestamp=round(timestamp, 3),
                    frame_id=frame_id,
                    trigger=trigger,
                    event_type=event_type,
                    risk_level=risk,
                    risk_object=risk_object,
                    evidence=evidence,
                    suggested_action=action,
                    latency_ms=int(round(det_ms)),
                    model=f"{model_name}+event-trigger",
                    detections=[asdict(d) for d in last_dets],
                )
                events.append(event)
                last_event_t = timestamp
            last_status = f"{event_type} | {risk} | {action}"

        annotated = frame.copy()
        _, _, risk, _, _, _ = _detections_to_event(last_dets, width, height)
        for det in last_dets:
            x1, y1, x2, y2 = det.box
            color = _color_for(det.cls, risk)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            _draw_label(annotated, f"{det.cls} {det.conf:.2f}", x1, y1, color)

        cv2.rectangle(annotated, (0, 0), (width, 40), (18, 18, 18), -1)
        cv2.putText(
            annotated,
            f"YOLO detector | t={timestamp:05.2f}s | objects={len(last_dets)} | events={len(events)}",
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        _draw_label(annotated, last_status, 12, height - 16, _color_for("", risk))
        writer.write(annotated)
        frame_id += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start_wall
    event_rows = [asdict(e) for e in events]
    avg_det_ms = float(np.mean(detector_times)) if detector_times else None
    summary = {
        "input_video": str(video_path),
        "output_video": str(output_video),
        "events_json": str(events_json),
        "frames_processed": frame_id,
        "video_fps": fps,
        "duration_seconds": round(frame_id / fps, 3) if fps else 0,
        "events": len(events),
        "detector_model": model_name,
        "device": device,
        "confidence": conf,
        "imgsz": imgsz,
        "detect_stride": detect_stride,
        "avg_detector_latency_ms": round(avg_det_ms, 2) if avg_det_ms is not None else None,
        "wall_time_seconds": round(elapsed, 3),
        "processing_fps": round(frame_id / elapsed, 2) if elapsed > 0 else None,
        "mode": "cloud-yolo-event-trigger",
    }
    events_json.write_text(json.dumps(event_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "events": event_rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run YOLO boxes + event trigger on a driving clip.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.28)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--detect-stride", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=1.2)
    args = parser.parse_args(argv)

    result = process_video_with_yolo(
        video_path=args.video,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        max_seconds=args.max_seconds,
        detect_stride=args.detect_stride,
        cooldown_seconds=args.cooldown_seconds,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
