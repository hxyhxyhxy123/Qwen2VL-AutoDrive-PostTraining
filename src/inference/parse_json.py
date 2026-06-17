"""Robust JSON extraction & repair for model outputs.

The output contract asks for a single JSON object {"answer", "reasoning"}.
Models frequently violate it (code fences, prose around the JSON, single
quotes, trailing commas, Python literals). We try increasingly aggressive
repairs and report whether the *original* output was already valid JSON, which
feeds the JSON Valid Rate metric.

`json_valid` is True only when a JSON object containing the required key(s) is
recoverable. We deliberately separate two notions:
  - strict_valid : original string parsed with json.loads without any repair
  - json_valid   : a usable object was recovered (possibly after light repair)
JSON Valid Rate in the leaderboard uses `strict_valid` so it measures the
model's real formatting reliability, not our repair effort.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

REQUIRED_KEYS = ("answer",)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _find_balanced_object(s: str) -> Optional[str]:
    """Return the first balanced {...} substring, or None."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _light_repairs(s: str) -> str:
    # Python literals -> JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s


def _try_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_json(text: str) -> Tuple[Dict[str, Any], bool, bool]:
    """Parse a model output.

    Returns (parsed, strict_valid, recovered_valid).
      parsed          : dict with at least "answer"/"reasoning" keys (best effort,
                        possibly {"answer": "<raw text>"} as a last resort)
      strict_valid    : True if the raw text was a JSON object w/ required keys,
                        with no repair beyond surrounding whitespace.
      recovered_valid : True if a JSON object w/ required keys was recovered.
    """
    raw = (text or "").strip()

    # 1) strict: whole string is JSON
    obj = _try_load(raw)
    strict = obj is not None and all(k in obj for k in REQUIRED_KEYS)
    if strict:
        return _normalize(obj), True, True

    # 2) strip a single code fence and retry strictly
    m = _FENCE_RE.search(raw)
    if m:
        inner = m.group(1).strip()
        obj2 = _try_load(inner)
        if obj2 is not None and all(k in obj2 for k in REQUIRED_KEYS):
            # was fenced, so not "strict" by our definition, but recovered.
            return _normalize(obj2), False, True

    # 3) first balanced object + light repairs
    candidate = _find_balanced_object(raw)
    if candidate is not None:
        for attempt in (candidate, _light_repairs(candidate)):
            obj3 = _try_load(attempt)
            if obj3 is not None and all(k in obj3 for k in REQUIRED_KEYS):
                return _normalize(obj3), False, True
        # object recovered but missing required key -> still partially usable
        obj3 = _try_load(_light_repairs(candidate))
        if obj3 is not None:
            return _normalize(obj3), False, False

    # 4) total fallback: treat the raw text as the answer
    return {"answer": raw, "reasoning": ""}, False, False


def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    answer = obj.get("answer", obj.get("maneuver", obj.get("label", "")))
    reasoning = obj.get("reasoning", obj.get("rationale", ""))
    out = {"answer": _to_str(answer), "reasoning": _to_str(reasoning)}
    # keep any extra fields for structured tasks
    for k, v in obj.items():
        if k not in out and k not in ("maneuver", "label", "rationale"):
            out[k] = v
    return out


def _to_str(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)


if __name__ == "__main__":
    # quick self-check
    samples = [
        '{"answer": "keep", "reasoning": "clear road"}',
        '```json\n{"answer":"turn_left","reasoning":"car ahead"}\n```',
        "Sure! {'answer': 'decelerate', 'reasoning': 'pedestrian',}",
        "The vehicle should keep going straight.",
        '{"answer": "yes",}',
    ]
    for s in samples:
        parsed, strict, rec = extract_json(s)
        print(f"strict={strict} recovered={rec} -> {parsed}")
