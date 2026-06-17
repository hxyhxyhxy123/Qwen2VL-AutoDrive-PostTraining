#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/AutoDrive-VLM-RLBench}
DATA_RAW="$PROJECT/data/raw/drivelm_nuscenes"
MODEL="$PROJECT/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
mkdir -p "$PROJECT/outputs/official"

latest_checkpoint() {
  local root="$1"
  find "$root" -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's#(.*checkpoint-)([0-9]+)$#\2 \0#' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

SFT_ADAPTER=${SFT_ADAPTER:-$(latest_checkpoint "$PROJECT/checkpoints/qwen3b-lora-sft" || true)}
ORPO_ADAPTER=${ORPO_ADAPTER:-$(latest_checkpoint "$PROJECT/checkpoints/qwen3b-lora-orpo" || true)}

BEST_NAME=${BEST_NAME:-orpo}
if [ "$BEST_NAME" = "orpo" ] && [ -n "${ORPO_ADAPTER:-}" ]; then
  BEST_ADAPTER=${BEST_ADAPTER:-$ORPO_ADAPTER}
elif [ "$BEST_NAME" = "sft" ] && [ -n "${SFT_ADAPTER:-}" ]; then
  BEST_ADAPTER=${BEST_ADAPTER:-$SFT_ADAPTER}
else
  BEST_NAME="sft"
  BEST_ADAPTER=${BEST_ADAPTER:-$SFT_ADAPTER}
fi
if [ -z "$BEST_ADAPTER" ] || [ ! -f "$BEST_ADAPTER/adapter_model.safetensors" ]; then
  echo "No valid best adapter found for official inference" >&2
  exit 1
fi

OUT="$PROJECT/outputs/official/${BEST_NAME}_val_output.json"
LOG="$PROJECT/outputs/official/${BEST_NAME}_val_infer.log"
echo "best_model=$BEST_NAME" | tee "$PROJECT/outputs/official/latest_best_model.txt"
echo "best_adapter=$BEST_ADAPTER" | tee -a "$PROJECT/outputs/official/latest_best_model.txt"
echo "official_output=$OUT" | tee -a "$PROJECT/outputs/official/latest_best_model.txt"

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python "$PROJECT/src/eval/run_drivelm_official_infer.py" \
  --val-json "$DATA_RAW/v1_1_val_nus_q_only.json" \
  --image-root "$DATA_RAW/nuscenes" \
  --model-path "$MODEL" \
  --adapter "$BEST_ADAPTER" \
  --output "$OUT" \
  --max-new-tokens 160 \
  --flush-every 25 2>&1 | tee "$LOG"
