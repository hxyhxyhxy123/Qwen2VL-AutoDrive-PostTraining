"""ms-swift GRPO reward plugin for DriveLM QA.

This reward is intentionally transparent and conservative. It rewards lexical
agreement with the DriveLM reference answer and lightly penalizes overly long
or empty completions. It is not a substitute for human judging, but it provides
a reproducible RL signal for an internship-scale GSPO-style experiment.
"""
from __future__ import annotations

import re
import string
from typing import List

from swift.rewards import ORM, orms


PUNCT_TABLE = str.maketrans("", "", string.punctuation)
UNCERTAIN_PATTERNS = (
    "cannot determine",
    "not enough information",
    "unable to determine",
    "can't determine",
    "unknown",
)


def normalize(text: str) -> str:
    text = (text or "").lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize(pred).split()
    gt_tokens = normalize(gt).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    used = [False] * len(gt_tokens)
    common = 0
    for tok in pred_tokens:
        for idx, gt_tok in enumerate(gt_tokens):
            if not used[idx] and tok == gt_tok:
                used[idx] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def length_penalty(pred: str, gt: str) -> float:
    pred_len = max(1, len(normalize(pred).split()))
    gt_len = max(1, len(normalize(gt).split()))
    if pred_len > max(24, 3 * gt_len):
        return 0.12
    if pred_len < 2 and gt_len > 4:
        return 0.08
    return 0.0


class DriveLMSoftReward(ORM):
    """Reward completions against DriveLM reference answers."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        for pred, gt in zip(completions, solution):
            pred = str(pred or "").strip()
            gt = str(gt or "").strip()
            if not pred or not gt:
                rewards.append(0.0)
                continue

            score = token_f1(pred, gt)
            if normalize(pred) == normalize(gt):
                score = min(1.0, score + 0.2)

            pred_norm = normalize(pred)
            if any(pattern in pred_norm for pattern in UNCERTAIN_PATTERNS) and token_f1(pred, gt) < 0.45:
                score -= 0.1

            score -= length_penalty(pred, gt)
            rewards.append(max(0.0, min(1.0, score)))
        return rewards


orms["drivelm_soft"] = DriveLMSoftReward


class DriveLMSoftFormatReward(ORM):
    """Token-F1 reward with lightweight format and brevity shaping."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        for pred, gt in zip(completions, solution):
            pred = str(pred or "").strip()
            gt = str(gt or "").strip()
            if not pred or not gt:
                rewards.append(0.0)
                continue

            f1 = token_f1(pred, gt)
            score = f1
            if normalize(pred) == normalize(gt):
                score += 0.2

            pred_norm = normalize(pred)
            words = pred_norm.split()
            if 3 <= len(words) <= 32:
                score += 0.05
            if "\n" in pred or len(words) > max(32, 3 * max(1, len(normalize(gt).split()))):
                score -= 0.12
            if any(pattern in pred_norm for pattern in UNCERTAIN_PATTERNS) and f1 < 0.45:
                score -= 0.1

            rewards.append(max(0.0, min(1.0, score)))
        return rewards


orms["drivelm_soft_format"] = DriveLMSoftFormatReward
