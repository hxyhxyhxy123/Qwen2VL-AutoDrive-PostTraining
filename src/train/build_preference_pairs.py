"""Build synthetic preference pairs from DriveLM SFT jsonl files.

The preferred response is the DriveLM ground-truth answer already present in
the SFT row. The rejected response is either a same-task answer from a different
sample or a conservative generic bad answer. This is suitable for a small P2
ORPO/DPO engineering experiment, but it should be reported as synthetic
preference data rather than human preference labels.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BAD_BY_TASK = {
    "perception": [
        "There are no relevant road users or objects visible in the provided views.",
        "The scene cannot be interpreted from these camera images.",
    ],
    "prediction": [
        "No visible agent is likely to change motion, so no prediction is needed.",
        "The future motion cannot be inferred from the provided camera views.",
    ],
    "planning": [
        "The ego vehicle should continue without considering nearby agents.",
        "The correct action cannot be determined from these images.",
    ],
    "behavior": [
        "The high-level driving behavior is unrelated to the visual scene.",
        "There is not enough information to describe the vehicle behavior.",
    ],
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def answer_of(row: Dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected the last message to be an assistant answer")
    return str(messages[-1].get("content", "")).strip()


def task_of(row: Dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("task") or "unknown")


def sample_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def grouped_answers(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        ans = answer_of(row)
        if ans:
            out[task_of(row)].append((ans, row))
    return out


def make_rejected(
    row: Dict[str, Any],
    answers_by_task: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    rng: random.Random,
) -> Tuple[str, str]:
    task = task_of(row)
    chosen = answer_of(row)
    candidates = answers_by_task.get(task, [])
    rng.shuffle(candidates)
    scene = (row.get("meta") or {}).get("scene_id")
    frame = (row.get("meta") or {}).get("frame_id")
    for candidate, src in candidates:
        src_meta = src.get("meta") or {}
        if candidate != chosen and (src_meta.get("scene_id"), src_meta.get("frame_id")) != (scene, frame):
            return candidate, "mismatched_same_task_answer"
    bad = BAD_BY_TASK.get(task) or ["I cannot determine the answer from the provided camera views."]
    rejected = rng.choice(bad)
    if rejected == chosen:
        rejected = "I cannot determine the answer from the provided camera views."
    return rejected, "generic_bad_answer"


def build_pairs(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    answers = grouped_answers(rows)
    out = []
    for row in rows:
        rejected, source = make_rejected(row, answers, rng)
        pair = {
            "messages": row["messages"],
            "images": row.get("images", []),
            "rejected_response": rejected,
            "meta": {
                **(row.get("meta") or {}),
                "preference_source": "synthetic_rule_based",
                "rejected_source": source,
            },
        }
        out.append(pair)
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-train", required=True)
    ap.add_argument("--sft-val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-size", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rows = sample_rows(load_jsonl(Path(args.sft_train)), args.train_size, args.seed)
    val_rows = sample_rows(load_jsonl(Path(args.sft_val)), args.val_size, args.seed + 1)

    train_pairs = build_pairs(train_rows, args.seed + 2)
    val_pairs = build_pairs(val_rows, args.seed + 3)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "drivelm_pref_train.jsonl", train_pairs)
    write_jsonl(out_dir / "drivelm_pref_val.jsonl", val_pairs)

    stats = {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "train_tasks": Counter(task_of(x) for x in train_pairs),
        "val_tasks": Counter(task_of(x) for x in val_pairs),
        "rejected_sources": Counter(x["meta"]["rejected_source"] for x in train_pairs + val_pairs),
        "out_dir": str(out_dir),
    }
    print(json.dumps(stats, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
