#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/AutoDrive-VLM-RLBench}
DATA_OUT="$PROJECT/data/processed"
MODEL="$PROJECT/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
LOG_DIR="$PROJECT/outputs/training"
TRAIN_OUT="$PROJECT/checkpoints/qwen3b-lora-dpo"
mkdir -p "$DATA_OUT" "$LOG_DIR" "$TRAIN_OUT"

log() {
  echo "[$(date -Is)] $*"
}

latest_checkpoint() {
  local root="$1"
  find "$root" -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's#(.*checkpoint-)([0-9]+)$#\2 \0#' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

SFT_ADAPTER=${SFT_ADAPTER:-$(latest_checkpoint "$PROJECT/checkpoints/qwen3b-lora-sft")}
if [ -z "${SFT_ADAPTER:-}" ] || [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Cannot find a valid SFT adapter under $PROJECT/checkpoints/qwen3b-lora-sft" >&2
  exit 1
fi

build_preferences() {
  log "building synthetic DriveLM preference pairs for DPO"
  /root/miniconda3/bin/python "$PROJECT/src/train/build_preference_pairs.py" \
    --sft-train "$DATA_OUT/drivelm_sft_train.jsonl" \
    --sft-val "$DATA_OUT/drivelm_sft_val.jsonl" \
    --out-dir "$DATA_OUT" \
    --train-size "${DPO_TRAIN_SIZE:-1000}" \
    --val-size "${DPO_VAL_SIZE:-200}" \
    --seed 52 | tee "$LOG_DIR/build_drivelm_preferences_dpo.json"

  head -n 16 "$DATA_OUT/drivelm_pref_train.jsonl" > "$DATA_OUT/drivelm_pref_smoke_train.jsonl"
  head -n 8 "$DATA_OUT/drivelm_pref_val.jsonl" > "$DATA_OUT/drivelm_pref_smoke_val.jsonl"
}

run_smoke() {
  log "starting P3 DPO smoke from adapter: $SFT_ADAPTER"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS="${MAX_PIXELS:-100352}" /root/miniconda3/bin/swift rlhf \
    --rlhf_type dpo \
    --model "$MODEL" \
    --adapters "$SFT_ADAPTER" \
    --dataset "$DATA_OUT/drivelm_pref_smoke_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_pref_smoke_val.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --eval_steps 1 \
    --save_steps 2 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 1536 \
    --beta 0.1 \
    --output_dir "$PROJECT/checkpoints/smoke-qwen3b-dpo" \
    --warmup_ratio 0 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
}

run_train() {
  log "starting formal P3 DPO: ${DPO_TRAIN_SIZE:-1000} preference pairs / ${DPO_VAL_SIZE:-200} val, 1 epoch"
  start_ts=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS="${MAX_PIXELS:-100352}" /root/miniconda3/bin/swift rlhf \
    --rlhf_type dpo \
    --model "$MODEL" \
    --adapters "$SFT_ADAPTER" \
    --dataset "$DATA_OUT/drivelm_pref_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_pref_val.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 1536 \
    --beta 0.1 \
    --output_dir "$TRAIN_OUT" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
  end_ts=$(date +%s)
  log "formal P3 DPO finished in $((end_ts - start_ts)) seconds"
}

build_preferences
run_smoke
run_train
