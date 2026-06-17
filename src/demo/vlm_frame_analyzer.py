"""Lazy ORPO-VLM frame analyzer for the cloud demo service."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.common import REPO_ROOT


DEFAULT_MODEL_PATH = str(REPO_ROOT / "model_cache/Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_ADAPTER_PATH = str(
    REPO_ROOT / "checkpoints/qwen3b-lora-orpo/v0-20260614-200914/checkpoint-125"
)

SYSTEM = (
    "You are a careful driving-scene visual assistant. Analyze only the visible "
    "image evidence and avoid unsupported claims."
)

DEFAULT_PROMPT = (
    "Analyze this driving-scene frame. Output strict JSON with these keys: "
    "event_type, risk_level, risk_object, evidence, suggested_action. "
    "Use risk_level from low, medium, high. Be concise."
)

_STATE: Dict[str, Any] = {}


def _extract_json_or_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {
        "event_type": "unparsed",
        "risk_level": "unknown",
        "risk_object": "unknown",
        "evidence": raw,
        "suggested_action": "unknown",
    }


def _load_model(model_path: str, adapter_path: Optional[str]):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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


def get_model(model_path: Optional[str] = None, adapter_path: Optional[str] = None):
    model_path = model_path or os.environ.get("AUTODRIVE_VLM_MODEL_PATH", DEFAULT_MODEL_PATH)
    adapter_path = adapter_path or os.environ.get("AUTODRIVE_VLM_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
    key = (model_path, adapter_path)
    if _STATE.get("key") != key:
        _STATE.clear()
        _STATE["key"] = key
        _STATE["model"], _STATE["processor"] = _load_model(model_path, adapter_path)
    return _STATE["model"], _STATE["processor"], model_path, adapter_path


def analyze_frame(
    image_path: str | Path,
    prompt: Optional[str] = None,
    max_new_tokens: int = 96,
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor, resolved_model_path, resolved_adapter_path = get_model(model_path, adapter_path)
    image_path = str(Path(image_path).resolve())
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": 401408},
                {"type": "text", "text": prompt or DEFAULT_PROMPT},
            ],
        },
    ]
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
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    latency = time.perf_counter() - start
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return {
        "image_path": image_path,
        "model_path": resolved_model_path,
        "adapter_path": resolved_adapter_path,
        "latency_s": round(latency, 3),
        "raw_output": raw,
        "parsed": _extract_json_or_text(raw),
        "mode": "qwen2.5-vl-3b-orpo-keyframe",
    }
