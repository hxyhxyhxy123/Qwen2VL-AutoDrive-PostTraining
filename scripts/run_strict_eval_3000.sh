#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SIZE="${SIZE:-3000}"
LIMIT="${LIMIT:-${SIZE}}"
EVAL_JSONL="${EVAL_JSONL:-data/processed/drivelm_strict_scene_eval_${SIZE}.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/eval/strict_scene_${SIZE}_${STAMP}}"

"${PYTHON_BIN}" -m src.data.build_strict_group_eval \
  --train-json data/raw/drivelm_nuscenes/v1_1_train_nus.json \
  --image-root data/raw/drivelm_nuscenes/nuscenes \
  --processed-dir data/processed \
  --out "${EVAL_JSONL}" \
  --size "${SIZE}" \
  --seed "${SEED:-2026}"

"${PYTHON_BIN}" -m src.eval.run_qwen_heldout_eval \
  --eval-jsonl "${EVAL_JSONL}" \
  --model-path model_cache/Qwen/Qwen2.5-VL-3B-Instruct \
  --sft-adapter checkpoints/qwen3b-lora-sft/v0-20260614-193604/checkpoint-188 \
  --orpo-adapter checkpoints/qwen3b-lora-orpo/v0-20260614-200914/checkpoint-125 \
  --input-settings image \
  --out-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-300}" \
  --seed "${SEED:-2026}"

echo "${OUT_DIR}" > outputs/eval/latest_strict_scene_eval_dir.txt
