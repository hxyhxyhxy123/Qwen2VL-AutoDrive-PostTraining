"""Answer matching + grouped metric computation.

Operates on *prediction records* (one row per sample x model x input_setting),
the schema produced by src/inference/run_infer.py:

  {
    "id", "model", "input_setting", "task_type", "difficulty",
    "answer_type",            # "exact" | "choice" | "label"
    "options"  (optional),    # list[str] for choice questions
    "label_space" (optional), # list[str] for classification / maneuver
    "gt",                     # ground-truth answer string
    "pred_answer",            # normalized model answer (from parse_json)
    "strict_valid",           # bool: raw output was strict JSON w/ required keys
    "latency_s"
  }

Headline metrics are decoupled per Principle #5: JSON Valid Rate is reported on
its own and never folded into accuracy / F1.
"""
from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Sequence

import numpy as np

# --------------------------------------------------------------------- matching
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    s = (s or "").strip()
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _choice_letter(s: str) -> str | None:
    """Return a single option letter if the string is just 'b' / 'b)' / '(b)'."""
    m = re.fullmatch(r"\(?([a-z])\)?", s.strip().lower())
    return m.group(1) if m else None


def is_correct(rec: Dict[str, Any], matching: Dict[str, Any]) -> bool:
    lc = matching.get("lowercase", True)
    sp = matching.get("strip_punctuation", True)
    letter_match = matching.get("choice_letter_match", True)

    pred = normalize_text(str(rec.get("pred_answer", "")), lc, sp)
    gt = normalize_text(str(rec.get("gt", "")), lc, sp)
    if pred == "":
        return False
    if pred == gt:
        return True

    answer_type = rec.get("answer_type", "exact")
    options = rec.get("options") or []

    if answer_type == "choice" and letter_match and options:
        # build letter<->text maps: A->options[0] ...
        norm_opts = [normalize_text(o, lc, sp) for o in options]
        letters = list(string.ascii_lowercase[: len(options)])
        text2letter = {t: l for l, t in zip(letters, norm_opts)}
        letter2text = {l: t for l, t in zip(letters, norm_opts)}

        pl = _choice_letter(pred)
        gl = _choice_letter(gt)
        # normalize both sides to their option text where possible
        pred_text = letter2text.get(pl, pred) if pl else pred
        gt_text = letter2text.get(gl, gt) if gl else gt
        if pred_text == gt_text:
            return True
        # or compare by letter
        pred_letter = pl or text2letter.get(pred)
        gt_letter = gl or text2letter.get(gt)
        if pred_letter and gt_letter and pred_letter == gt_letter:
            return True

    return False


def score_records(records: List[Dict[str, Any]], matching: Dict[str, Any]) -> None:
    """In-place: add integer `correct` field to each record."""
    for r in records:
        r["correct"] = int(is_correct(r, matching))


# ----------------------------------------------------------------------- F1
def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    if not labels:
        return float("nan")
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


# ----------------------------------------------------------- group aggregation
def _mean(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_group(records: List[Dict[str, Any]], metric_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute headline metrics for a single (model, input_setting) group."""
    qa_types = set(metric_groups.get("qa_accuracy", []))
    man_types = set(metric_groups.get("maneuver_accuracy", []))
    f1_types = set(metric_groups.get("macro_f1_over", []))

    qa = [r["correct"] for r in records if r["task_type"] in qa_types]
    man = [r["correct"] for r in records if r["task_type"] in man_types]
    allc = [r["correct"] for r in records]
    hard = [r["correct"] for r in records if r.get("difficulty") == "hard"]
    strict = [int(bool(r.get("strict_valid"))) for r in records]
    lat = [float(r["latency_s"]) for r in records if r.get("latency_s") is not None]

    f1_recs = [r for r in records if r["task_type"] in f1_types and r.get("label_space")]
    if f1_recs:
        labels = sorted({l for r in f1_recs for l in r["label_space"]})
        from .metrics import normalize_text as _n  # self ref ok

        y_true = [_n(str(r["gt"])) for r in f1_recs]
        y_pred = [_n(str(r["pred_answer"])) for r in f1_recs]
        labels_n = [_n(l) for l in labels]
        mf1 = macro_f1(y_true, y_pred, labels_n)
    else:
        mf1 = float("nan")

    return {
        "n": len(records),
        "qa_accuracy": _mean(qa),
        "maneuver_accuracy": _mean(man),
        "macro_f1": mf1,
        "overall_accuracy": _mean(allc),
        "hard_accuracy": _mean(hard),
        "json_valid_rate": _mean(strict),
        "latency_s": _mean(lat),
        "_n_qa": len(qa),
        "_n_maneuver": len(man),
        "_n_hard": len(hard),
    }


def group_key(rec: Dict[str, Any]) -> tuple:
    return (rec["model"], rec["input_setting"])


def aggregate_all(records: List[Dict[str, Any]], metric_groups: Dict[str, List[str]]) -> Dict[tuple, Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    return {k: aggregate_group(v, metric_groups) for k, v in groups.items()}
