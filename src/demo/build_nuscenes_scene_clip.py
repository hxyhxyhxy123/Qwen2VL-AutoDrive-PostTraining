"""Build a smoother real nuScenes scene clip from metadata.

The DriveLM/nuScenes release stores camera frames as images instead of mp4
files. This helper follows nuScenes metadata for one scene and one camera, then
reconstructs a short video clip for demos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.common import REPO_ROOT, ensure_dir


def _require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("opencv-python is required. Install it with: pip install opencv-python") from exc


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_scene_clip(
    nuscenes_root: str | Path,
    output_video: str | Path,
    scene_name: Optional[str] = "scene-0061",
    camera: str = "CAM_FRONT",
    fps: float = 12.0,
    max_frames: int = 180,
    start_index: int = 0,
    stride: int = 1,
    include_sweeps: bool = True,
    resize_width: int = 1280,
    output_meta: str | Path | None = None,
) -> dict:
    cv2 = _require_cv2()
    nuscenes_root = Path(nuscenes_root)
    meta_root = nuscenes_root / "v1.0-mini"
    if not meta_root.exists():
        raise FileNotFoundError(meta_root)

    scenes = _load_json(meta_root / "scene.json")
    samples = _load_json(meta_root / "sample.json")
    sample_data = _load_json(meta_root / "sample_data.json")
    sensors = _load_json(meta_root / "sensor.json")
    calibrated = _load_json(meta_root / "calibrated_sensor.json")

    scene = None
    if scene_name:
        scene = next((s for s in scenes if s["name"] == scene_name), None)
        if scene is None:
            raise ValueError(f"Scene not found: {scene_name}")
    else:
        scene = max(scenes, key=lambda s: int(s.get("nbr_samples", 0)))

    sample_to_scene = {s["token"]: s["scene_token"] for s in samples}
    sensor_channel = {s["token"]: s["channel"] for s in sensors}
    calib_channel = {c["token"]: sensor_channel[c["sensor_token"]] for c in calibrated}

    frame_rows = []
    for row in sample_data:
        if row.get("fileformat") != "jpg":
            continue
        if sample_to_scene.get(row["sample_token"]) != scene["token"]:
            continue
        if calib_channel.get(row["calibrated_sensor_token"]) != camera:
            continue
        if not include_sweeps and not row.get("is_key_frame", False):
            continue
        path = nuscenes_root / row["filename"]
        if path.exists():
            frame_rows.append({**row, "path": path})

    frame_rows = sorted(frame_rows, key=lambda r: int(r["timestamp"]))
    frame_rows = frame_rows[start_index:: max(stride, 1)]
    if max_frames > 0:
        frame_rows = frame_rows[:max_frames]
    if not frame_rows:
        raise RuntimeError(f"No frames found for scene={scene['name']} camera={camera}")

    first = cv2.imread(str(frame_rows[0]["path"]))
    if first is None:
        raise RuntimeError(f"Could not read image: {frame_rows[0]['path']}")
    h, w = first.shape[:2]
    if resize_width > 0 and w > resize_width:
        scale = resize_width / w
        out_w = resize_width
        out_h = int(round(h * scale))
    else:
        out_w, out_h = w, h

    output_video = Path(output_video)
    ensure_dir(output_video.parent)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")

    used: List[dict] = []
    for row in frame_rows:
        frame = cv2.imread(str(row["path"]))
        if frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        used.append(
            {
                "timestamp": row["timestamp"],
                "is_key_frame": row["is_key_frame"],
                "filename": row["filename"],
                "path": str(row["path"]),
            }
        )
    writer.release()

    meta = {
        "source": "nuScenes v1.0-mini",
        "nuscenes_root": str(nuscenes_root),
        "scene": scene["name"],
        "scene_description": scene.get("description", ""),
        "camera": camera,
        "include_sweeps": include_sweeps,
        "frames": len(used),
        "fps": fps,
        "duration_seconds": round(len(used) / fps, 3) if fps else 0,
        "output_video": str(output_video),
        "source_frames": used,
    }
    output_meta = Path(output_meta) if output_meta else output_video.with_suffix(".json")
    output_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a real nuScenes scene video clip.")
    parser.add_argument("--nuscenes-root", default=str(REPO_ROOT / "data/raw/nuscenes"))
    parser.add_argument("--output-video", default=str(REPO_ROOT / "outputs/demo/nuscenes_scene_cam_front.mp4"))
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--scene-name", default="scene-0061")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--resize-width", type=int, default=1280)
    args = parser.parse_args(argv)

    meta = build_scene_clip(
        nuscenes_root=args.nuscenes_root,
        output_video=args.output_video,
        scene_name=args.scene_name,
        camera=args.camera,
        fps=args.fps,
        max_frames=args.max_frames,
        start_index=args.start_index,
        stride=args.stride,
        include_sweeps=not args.keyframes_only,
        resize_width=args.resize_width,
        output_meta=args.output_meta,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
