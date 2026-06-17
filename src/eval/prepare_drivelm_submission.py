"""Prepare DriveLM official submission.json from an output.json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def non_empty(value: str, field: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"Missing required submission field: {field}")
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--submission-json", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--authors", required=True, help="Comma-separated author names")
    ap.add_argument("--email", required=True)
    ap.add_argument("--institution", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--expected-results", type=int, default=15480)
    args = ap.parse_args()

    with open(args.output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError("DriveLM output must be a JSON list")
    ids = [str(x.get("id")) for x in results if isinstance(x, dict)]
    if len(results) != args.expected_results or len(set(ids)) != args.expected_results:
        raise ValueError(
            f"Expected {args.expected_results} results with unique ids, "
            f"got len={len(results)} unique={len(set(ids))}"
        )

    submission = {
        "method": non_empty(args.method, "method"),
        "team": non_empty(args.team, "team"),
        "authors": [x.strip() for x in args.authors.split(",") if x.strip()],
        "email": non_empty(args.email, "email"),
        "institution": non_empty(args.institution, "institution"),
        "country": non_empty(args.country, "country"),
        "results": results,
    }
    if not submission["authors"]:
        raise ValueError("Missing required submission field: authors")

    out = Path(args.submission_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(json.dumps({"submission_json": str(out), "results": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
