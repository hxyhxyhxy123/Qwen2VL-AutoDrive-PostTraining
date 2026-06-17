#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

"${PYTHON_BIN}" -m uvicorn src.demo.cloud_event_service:app --host "${HOST}" --port "${PORT}"
