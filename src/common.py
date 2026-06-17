"""Shared IO / config / path helpers used across the pipeline.

Kept dependency-light (only numpy + pyyaml) so the eval pipeline runs on a
no-GPU box without torch/transformers installed.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- IO
def read_jsonl(path: str | os.PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} is not valid JSON: {e}") from e
    return rows


def iter_jsonl(path: str | os.PathLike) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------ misc
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_image_path(rel_or_abs: str, image_root: str | os.PathLike = ".") -> Path:
    """Resolve an image path stored in the dataset.

    Paths are stored relative to image_root (default repo cwd). Absolute paths
    are returned as-is.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return Path(image_root) / p
