"""Run Qwen2.5-VL held-out evaluation for base/SFT/ORPO adapters.

The metrics here are intentionally simple and transparent:
- exact_match: strict normalized text equality.
- token_f1: lexical overlap between prediction and DriveLM reference answer.
- match_rate_f1_0_5: share of examples with token_f1 >= 0.5.

These are not a substitute for human/LLM-judge evaluation, but they are enough
to decide whether the trained checkpoints are ready for a real leaderboard pass.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_IMAGE = (
    "You are a careful driving-scene visual assistant. Answer only from the "
    "provided multi-view camera images. Be concise and avoid unsupported claims."
)
SYSTEM_TEXT = (
    "You are a driving-scene QA assistant. No images are available. Answer from "
    "the question text only and be concise."
)

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(s: str) -> str:
    s = (s or "").lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gt: str) -> float:
    p = normalize(pred).split()
    g = normalize(gt).split()
    if not p or not g:
        return 0.0
    used = [False] * len(g)
    common = 0
    for tok in p:
        for i, gt_tok in enumerate(g):
            if not used[i] and tok == gt_tok:
                used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def prompt_for(row: Dict[str, Any], input_setting: str) -> List[Dict[str, Any]]:
    task = row["task_type"]
    question = row["question"]
    if input_setting == "image":
        content: List[Dict[str, Any]] = []
        for image in row.get("images", []):
            content.append({"type": "image", "image": image, "max_pixels": 200704})
        views = ", ".join(row.get("camera_order") or [])
        text = (
            f"Camera views are provided in this order: {views}.\n"
            f"Task: {task}.\n"
            f"Question: {question}\n"
            "Answer in one concise sentence."
        )
        content.append({"type": "text", "text": text})
        return [
            {"role": "system", "content": SYSTEM_IMAGE},
            {"role": "user", "content": content},
        ]
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {
            "role": "user",
            "content": (
                f"Task: {task}.\n"
                f"Question: {question}\n"
                "Answer in one concise sentence."
            ),
        },
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


def generate_one(model, processor, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
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


def run_model(
    rows: List[Dict[str, Any]],
    model_name: str,
    model_path: str,
    adapter_path: Optional[str],
    input_settings: Iterable[str],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    model, processor = load_model(model_path, adapter_path)
    records: List[Dict[str, Any]] = []
    try:
        for setting in input_settings:
            for idx, row in enumerate(rows, 1):
                messages = prompt_for(row, setting)
                start = time.perf_counter()
                try:
                    pred = generate_one(model, processor, messages, max_new_tokens)
                    error = ""
                except Exception as exc:  # keep the run going and report failures
                    pred = ""
                    error = repr(exc)
                latency = time.perf_counter() - start
                f1 = token_f1(pred, row["gt"])
                records.append(
                    {
                        "id": row["id"],
                        "model": model_name,
                        "input_setting": setting,
                        "task_type": row["task_type"],
                        "question": row["question"],
                        "gt": row["gt"],
                        "pred": pred,
                        "exact_match": int(normalize(pred) == normalize(row["gt"])),
                        "token_f1": f1,
                        "match_f1_0_5": int(f1 >= 0.5),
                        "latency_s": latency,
                        "error": error,
                    }
                )
                if idx % 10 == 0:
                    print(f"{model_name}/{setting}: {idx}/{len(rows)}", flush=True)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return vals[idx]


def add_bootstrap_ci(rows: List[Dict[str, Any]], records: List[Dict[str, Any]], samples: int, seed: int) -> None:
    if samples <= 0:
        return
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["model"], rec["input_setting"])].append(rec)

    ci_by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, vals in groups.items():
        n = len(vals)
        if n == 0:
            continue
        exacts = []
        f1s = []
        match_rates = []
        for _ in range(samples):
            draw = [vals[rng.randrange(n)] for _ in range(n)]
            exacts.append(sum(v["exact_match"] for v in draw) / n)
            f1s.append(sum(v["token_f1"] for v in draw) / n)
            match_rates.append(sum(v["match_f1_0_5"] for v in draw) / n)
        ci_by_key[key] = {
            "exact_ci95_low": percentile(exacts, 0.025),
            "exact_ci95_high": percentile(exacts, 0.975),
            "token_f1_ci95_low": percentile(f1s, 0.025),
            "token_f1_ci95_high": percentile(f1s, 0.975),
            "match_f1_0_5_ci95_low": percentile(match_rates, 0.025),
            "match_f1_0_5_ci95_high": percentile(match_rates, 0.975),
        }

    for row in rows:
        row.update(
            ci_by_key.get(
                (row["model"], row["input_setting"]),
                {
                    "exact_ci95_low": 0.0,
                    "exact_ci95_high": 0.0,
                    "token_f1_ci95_low": 0.0,
                    "token_f1_ci95_high": 0.0,
                    "match_f1_0_5_ci95_low": 0.0,
                    "match_f1_0_5_ci95_high": 0.0,
                },
            )
        )


def aggregate(records: List[Dict[str, Any]], bootstrap_samples: int = 0, seed: int = 123) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["model"], rec["input_setting"])].append(rec)
    rows = []
    for (model, setting), vals in sorted(groups.items()):
        n = len(vals)
        rows.append(
            {
                "model": model,
                "input_setting": setting,
                "n": n,
                "exact_match": sum(v["exact_match"] for v in vals) / n if n else 0.0,
                "token_f1": sum(v["token_f1"] for v in vals) / n if n else 0.0,
                "match_rate_f1_0_5": sum(v["match_f1_0_5"] for v in vals) / n if n else 0.0,
                "latency_s": sum(v["latency_s"] for v in vals) / n if n else 0.0,
                "errors": sum(1 for v in vals if v["error"]),
            }
        )
    image_scores = {r["model"]: r["token_f1"] for r in rows if r["input_setting"] == "image"}
    text_scores = {r["model"]: r["token_f1"] for r in rows if r["input_setting"] == "text_only"}
    for row in rows:
        if row["model"] in image_scores and row["model"] in text_scores:
            row["vision_gap_token_f1"] = image_scores[row["model"]] - text_scores[row["model"]]
        else:
            row["vision_gap_token_f1"] = None
    add_bootstrap_ci(rows, records, bootstrap_samples, seed)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_leaderboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    show_vision_gap = any(r.get("vision_gap_token_f1") is not None for r in rows)
    headers = [
        "Model",
        "Input",
        "N",
        "Exact",
        "Token-F1",
        "Token-F1 95% CI",
        "F1>=0.5",
        "Latency(s)",
    ]
    if show_vision_gap:
        headers.insert(-1, "Vision Gap")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in sorted(rows, key=lambda x: (x["input_setting"] != "image", -x["token_f1"])):
        values = [
            r["model"],
            r["input_setting"],
            str(r["n"]),
            f"{r['exact_match']:.3f}",
            f"{r['token_f1']:.3f}",
            (
                f"[{r.get('token_f1_ci95_low', 0.0):.3f}, "
                f"{r.get('token_f1_ci95_high', 0.0):.3f}]"
            ),
            f"{r['match_rate_f1_0_5']:.3f}",
            f"{r['latency_s']:.2f}",
        ]
        if show_vision_gap:
            gap = r.get("vision_gap_token_f1")
            values.insert(-1, "" if gap is None else f"{gap:.3f}")
        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_adapter_spec(spec: str) -> Tuple[str, Optional[str]]:
    if "=" not in spec:
        raise ValueError(f"Adapter spec must be NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Adapter name is empty in spec: {spec}")
    if path.lower() in {"", "none", "base", "null"}:
        return name, None
    return name, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sft-adapter", default="")
    ap.add_argument("--orpo-adapter", default="")
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Additional model spec as NAME=ADAPTER_PATH. Use NAME=none for a base-model alias.",
    )
    ap.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--input-settings", nargs="+", default=["image", "text_only"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--bootstrap-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.eval_jsonl))[: args.limit]
    specs: List[Tuple[str, Optional[str]]] = []
    if args.include_base:
        specs.append(("base", None))
    if args.sft_adapter:
        specs.append(("sft", args.sft_adapter))
    if args.orpo_adapter:
        specs.append(("orpo", args.orpo_adapter))
    specs.extend(parse_adapter_spec(spec) for spec in args.adapter)
    deduped: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for name, adapter in specs:
        if name in seen:
            raise ValueError(f"Duplicate model name: {name}")
        seen.add(name)
        deduped.append((name, adapter))
    specs = deduped
    if not specs:
        raise ValueError("No model specs were provided")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    for name, adapter in specs:
        all_records.extend(
            run_model(
                rows,
                name,
                args.model_path,
                adapter,
                input_settings=args.input_settings,
                max_new_tokens=args.max_new_tokens,
            )
        )
        pred_path = out_dir / f"predictions_{name}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                if rec["model"] == name:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = aggregate(all_records, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_csv(out_dir / "heldout_metrics.csv", metrics)
    write_leaderboard(out_dir / "heldout_leaderboard.md", metrics)
    with open(out_dir / "heldout_predictions_all.jsonl", "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"metrics": metrics, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
