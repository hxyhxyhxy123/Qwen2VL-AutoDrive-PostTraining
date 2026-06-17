#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CLIP_PATH="${CLIP_PATH:-outputs/demo/nuscenes_scene_cam_front.mp4}"
OUT_DIR="${OUT_DIR:-outputs/demo/nuscenes_yolo_event_demo}"

"${PYTHON_BIN}" -m src.demo.build_nuscenes_scene_clip \
  --output-video "${CLIP_PATH}" \
  --scene-name "${SCENE_NAME:-scene-0061}" \
  --camera "${CAMERA:-CAM_FRONT}" \
  --fps "${FPS:-12}" \
  --max-frames "${MAX_FRAMES:-180}" \
  --resize-width "${RESIZE_WIDTH:-1280}"

"${PYTHON_BIN}" -m src.demo.yolo_video_event_demo \
  --video "${CLIP_PATH}" \
  --output-dir "${OUT_DIR}" \
  --model "${YOLO_MODEL:-yolov8n.pt}" \
  --conf "${CONF:-0.28}" \
  --imgsz "${IMGSZ:-640}" \
  --max-seconds "${MAX_SECONDS:-15}" \
  --detect-stride "${DETECT_STRIDE:-1}"
