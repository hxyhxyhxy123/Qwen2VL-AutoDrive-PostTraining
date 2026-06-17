"""Visual-faithfulness control settings (Principle #4).

For every sample we can render four input variants:
  - image          : the real image(s)
  - text_only      : no image at all (measures language prior)
  - blank_image    : a neutral gray image of the same size (measures "needs *an*
                     image vs. needs *the* image")
  - shuffled_image : an image from a DIFFERENT sample (measures whether the model
                     actually grounds in the correct image)

`build_inputs` returns (pil_images, used_image_paths). For text_only it returns
([], []). The shuffled mapping is a deterministic derangement so results are
reproducible and every sample is guaranteed a *different* image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..common import resolve_image_path

SETTINGS = ("image", "text_only", "blank_image", "shuffled_image")
_BLANK_COLOR = (128, 128, 128)
_BLANK_SIZE = (448, 448)


def build_shuffle_map(ids: List[str], seed: int = 1234) -> Dict[str, str]:
    """Deterministic derangement: id -> a different id whose image we'll use."""
    import random

    rng = random.Random(seed)
    n = len(ids)
    if n <= 1:
        return {ids[0]: ids[0]} if ids else {}
    order = list(range(n))
    # rotate by a random non-zero offset => guaranteed no fixed point
    offset = rng.randrange(1, n)
    mapping = {}
    for i, _id in enumerate(ids):
        mapping[_id] = ids[(i + offset) % n]
    return mapping


def _load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _blank_like(ref_path: Path | None):
    from PIL import Image

    size = _BLANK_SIZE
    if ref_path is not None and Path(ref_path).exists():
        try:
            with Image.open(ref_path) as im:
                size = im.size
        except Exception:
            pass
    return Image.new("RGB", size, _BLANK_COLOR)


def build_inputs(
    sample: Dict,
    setting: str,
    image_root: str = ".",
    shuffle_map: Dict[str, str] | None = None,
    id_to_sample: Dict[str, Dict] | None = None,
) -> Tuple[List, List[str]]:
    """Return (list_of_PIL_images, list_of_source_image_paths)."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}, expected one of {SETTINGS}")

    own_paths = [str(p) for p in sample.get("images", [])]

    if setting == "text_only":
        return [], []

    if setting == "image":
        imgs = [_load_image(resolve_image_path(p, image_root)) for p in own_paths]
        return imgs, own_paths

    if setting == "blank_image":
        ref = resolve_image_path(own_paths[0], image_root) if own_paths else None
        return [_blank_like(ref)], ["<blank>"]

    # shuffled_image
    assert shuffle_map is not None and id_to_sample is not None, (
        "shuffled_image needs shuffle_map and id_to_sample"
    )
    other_id = shuffle_map[sample["id"]]
    other = id_to_sample[other_id]
    other_paths = [str(p) for p in other.get("images", [])]
    if not other_paths:  # degenerate: fall back to blank so we never use own image
        return [_blank_like(None)], ["<blank-fallback>"]
    imgs = [_load_image(resolve_image_path(other_paths[0], image_root))]
    return imgs, [other_paths[0]]
