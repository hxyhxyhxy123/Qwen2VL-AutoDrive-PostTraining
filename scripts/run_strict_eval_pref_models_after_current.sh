#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_PIDFILE="${CURRENT_PIDFILE:-outputs/eval/strict_scene_3000_run.pid}"
FIRST_DIR_FILE="${FIRST_DIR_FILE:-outputs/eval/latest_strict_scene_eval_dir.txt}"
EVAL_JSONL="${EVAL_JSONL:-data/processed/drivelm_strict_scene_eval_3000.jsonl}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PREF_OUT_DIR="${PREF_OUT_DIR:-outputs/eval/strict_scene_3000_pref_models_${STAMP}}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-outputs/eval/strict_scene_3000_all_models_${STAMP}}"
STATUS="${STATUS:-outputs/eval/strict_scene_3000_all_models_status.json}"

echo "{\"stage\":\"waiting_first_eval\",\"updated_at\":\"$(date +%FT%T%z)\"}" > "${STATUS}"

if [ -f "${CURRENT_PIDFILE}" ]; then
  current_pid="$(cat "${CURRENT_PIDFILE}")"
  while kill -0 "${current_pid}" 2>/dev/null; do
    sleep 60
  done
fi

if [ ! -f "${FIRST_DIR_FILE}" ]; then
  echo "{\"stage\":\"failed\",\"detail\":\"missing first eval dir file ${FIRST_DIR_FILE}\",\"updated_at\":\"$(date +%FT%T%z)\"}" > "${STATUS}"
  exit 1
fi

FIRST_DIR="$(cat "${FIRST_DIR_FILE}")"
echo "{\"stage\":\"running_pref_models\",\"first_dir\":\"${FIRST_DIR}\",\"pref_out_dir\":\"${PREF_OUT_DIR}\",\"updated_at\":\"$(date +%FT%T%z)\"}" > "${STATUS}"

"${PYTHON_BIN}" -m src.eval.run_qwen_heldout_eval \
  --eval-jsonl "${EVAL_JSONL}" \
  --model-path model_cache/Qwen/Qwen2.5-VL-3B-Instruct \
  --no-include-base \
  --adapter dpo=checkpoints/qwen3b-lora-dpo/v0-20260615-010607/checkpoint-125 \
  --adapter sft_gspo=checkpoints/qwen3b-lora-gspo-sftstart/v0-20260615-113054/checkpoint-60 \
  --input-settings image \
  --out-dir "${PREF_OUT_DIR}" \
  --limit "${LIMIT:-3000}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-300}" \
  --seed "${SEED:-2026}"

echo "{\"stage\":\"merging\",\"first_dir\":\"${FIRST_DIR}\",\"pref_out_dir\":\"${PREF_OUT_DIR}\",\"merged_out_dir\":\"${MERGED_OUT_DIR}\",\"updated_at\":\"$(date +%FT%T%z)\"}" > "${STATUS}"

"${PYTHON_BIN}" -m src.eval.merge_eval_runs \
  --run-dir "${FIRST_DIR}" \
  --run-dir "${PREF_OUT_DIR}" \
  --out-dir "${MERGED_OUT_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-300}" \
  --seed "${SEED:-2026}"

echo "${MERGED_OUT_DIR}" > outputs/eval/latest_strict_scene_3000_all_models_dir.txt
echo "{\"stage\":\"completed\",\"first_dir\":\"${FIRST_DIR}\",\"pref_out_dir\":\"${PREF_OUT_DIR}\",\"merged_out_dir\":\"${MERGED_OUT_DIR}\",\"updated_at\":\"$(date +%FT%T%z)\"}" > "${STATUS}"
