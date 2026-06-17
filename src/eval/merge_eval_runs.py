"""Merge prediction JSONL files from multiple eval runs into one leaderboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.eval.run_qwen_heldout_eval import aggregate, write_csv, write_leaderboard


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, help="Eval output dir to merge.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    records: List[Dict] = []
    seen: set[Tuple[str, str, str]] = set()
    sources = []
    for run_dir_s in args.run_dir:
        run_dir = Path(run_dir_s)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        sources.append(str(run_dir))
        files = sorted(run_dir.glob("predictions_*.jsonl"))
        if not files and (run_dir / "heldout_predictions_all.jsonl").exists():
            files = [run_dir / "heldout_predictions_all.jsonl"]
        for path in files:
            for rec in _load_jsonl(path):
                key = (str(rec.get("model")), str(rec.get("input_setting")), str(rec.get("id")))
                if key in seen:
                    continue
                seen.add(key)
                records.append(rec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = aggregate(records, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_csv(out_dir / "heldout_metrics.csv", metrics)
    write_leaderboard(out_dir / "heldout_leaderboard.md", metrics)
    with (out_dir / "heldout_predictions_all.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "out_dir": str(out_dir),
        "sources": sources,
        "records": len(records),
        "metrics": metrics,
    }
    (out_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
