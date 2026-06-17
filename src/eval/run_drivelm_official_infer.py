"""Infer DriveLM v1.1 val questions and write official output.json.

DriveLM official submission expects a list of records similar to:
  {
    "id": "<scene_id>_<frame_id>_<question_index>",
    "question": "<image>\\n...",
    "gt_answer": "",
    "answer": "..."
  }

The official val file does not release GT answers, so this script only performs
format generation and inference. Use `prepare_drivelm_submission.py` afterwards
to wrap the output with team metadata.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


TASKS = ("perception", "prediction", "planning", "behavior")
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

SYSTEM = (
    "You are a careful driving-scene visual assistant. Answer only from the "
    "provided multi-view camera images. Be concise and avoid unsupported claims."
)


def resolve_image(raw_path: str, image_root: Path) -> str:
    p = raw_path.replace("\\", "/")
    marker = "nuscenes/"
    if marker in p:
        p = p.split(marker, 1)[1]
    resolved = (image_root / p).resolve()
    if resolved.exists():
        return str(resolved)

    # DriveLM val image zip is packaged as val_data/CAM_*/xxx.jpg instead of
    # nuscenes/samples/CAM_*/xxx.jpg.
    parts = Path(p).parts
    if "samples" in parts:
        idx = parts.index("samples")
        if len(parts) >= idx + 3:
            camera = parts[idx + 1]
            filename = parts[-1]
            val_resolved = (image_root.parent / "val_data" / camera / filename).resolve()
            if val_resolved.exists():
                return str(val_resolved)
    return str(resolved)


def iter_questions(val_json: Path, image_root: Path) -> Iterable[Dict[str, Any]]:
    with open(val_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for scene_id, scene in data.items():
        for frame_id, frame in (scene.get("key_frames") or {}).items():
            img_map = frame.get("image_paths") or {}
            images = [
                resolve_image(img_map[c], image_root)
                for c in CAMERA_ORDER
                if c in img_map
            ]
            question_index = 0
            for task in TASKS:
                for qa in (frame.get("QA") or {}).get(task, []) or []:
                    question = str(qa.get("Q", "")).strip()
                    yield {
                        "id": f"{scene_id}_{frame_id}_{question_index}",
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "task": task,
                        "question": question,
                        "question_with_image": f"<image>\n{question}",
                        "gt_answer": str(qa.get("A", "") or ""),
                        "tag": qa.get("tag"),
                        "images": images,
                        "camera_order": CAMERA_ORDER[: len(images)],
                    }
                    question_index += 1


def make_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image in row["images"]:
        content.append({"type": "image", "image": image, "max_pixels": 200704})
    views = ", ".join(row.get("camera_order") or [])
    text = (
        f"Camera views are provided in this order: {views}.\n"
        f"Task: {row['task']}.\n"
        f"Question: {row['question']}\n"
        "Answer concisely. For multiple-choice questions, answer with the option letter when possible."
    )
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def load_model(model_path: str, adapter_path: Optional[str]):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def generate_answer(model, processor, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()

    rows = list(iter_questions(Path(args.val_json), Path(args.image_root)))
    if args.limit > 0:
        rows = rows[: args.limit]

    missing = [p for row in rows[:5] for p in row["images"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(missing[0])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    outputs: List[Dict[str, Any]] = []
    done_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            outputs = json.load(f)
        done_ids = {str(x.get("id")) for x in outputs}
        rows = [row for row in rows if row["id"] not in done_ids]
        print(
            json.dumps(
                {"resume_from": str(out_path), "completed": len(done_ids), "remaining": len(rows)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    model, processor = load_model(args.model_path, args.adapter or None)
    start_all = time.perf_counter()
    try:
        for idx, row in enumerate(rows, 1):
            start = time.perf_counter()
            try:
                if row["question"]:
                    answer = generate_answer(model, processor, make_messages(row), args.max_new_tokens)
                else:
                    answer = ""
                error = ""
            except Exception as exc:
                answer = ""
                error = repr(exc)
            outputs.append(
                {
                    "id": row["id"],
                    "question": row["question_with_image"],
                    "gt_answer": row["gt_answer"],
                    "answer": answer,
                }
            )
            if idx % args.flush_every == 0 or idx == len(rows):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, out_path)
                elapsed = time.perf_counter() - start_all
                speed = elapsed / idx
                remaining = speed * (len(rows) - idx)
                print(
                    json.dumps(
                        {
                            "progress": f"{len(done_ids) + idx}/{len(done_ids) + len(rows)}",
                            "elapsed_s": round(elapsed, 1),
                            "sec_per_q": round(speed, 3),
                            "eta_s": round(remaining, 1),
                            "output": str(out_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
