#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/AutoDrive-VLM-RLBench}
DATA_OUT="$PROJECT/data/processed"
MODEL="$PROJECT/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
LOG_DIR="$PROJECT/outputs/training"
TRAIN_OUT="$PROJECT/checkpoints/qwen3b-lora-orpo"
mkdir -p "$DATA_OUT" "$LOG_DIR" "$TRAIN_OUT"

log() {
  echo "[$(date -Is)] $*"
}

latest_sft_adapter() {
  find "$PROJECT/checkpoints/qwen3b-lora-sft" -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's#(.*checkpoint-)([0-9]+)$#\2 \0#' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

SFT_ADAPTER=${SFT_ADAPTER:-$(latest_sft_adapter)}
if [ -z "${SFT_ADAPTER:-}" ] || [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Cannot find a valid SFT adapter under $PROJECT/checkpoints/qwen3b-lora-sft" >&2
  exit 1
fi

build_preferences() {
  log "building synthetic DriveLM preference pairs from P1 SFT data"
  /root/miniconda3/bin/python "$PROJECT/src/train/build_preference_pairs.py" \
    --sft-train "$DATA_OUT/drivelm_sft_train.jsonl" \
    --sft-val "$DATA_OUT/drivelm_sft_val.jsonl" \
    --out-dir "$DATA_OUT" \
    --train-size 1000 \
    --val-size 200 \
    --seed 42 | tee "$LOG_DIR/build_drivelm_preferences.json"

  head -n 16 "$DATA_OUT/drivelm_pref_train.jsonl" > "$DATA_OUT/drivelm_pref_smoke_train.jsonl"
  head -n 8 "$DATA_OUT/drivelm_pref_val.jsonl" > "$DATA_OUT/drivelm_pref_smoke_val.jsonl"

  /root/miniconda3/bin/python - <<'PY'
import json, os
paths = [
    '/root/autodl-tmp/AutoDrive-VLM-RLBench/data/processed/drivelm_pref_train.jsonl',
    '/root/autodl-tmp/AutoDrive-VLM-RLBench/data/processed/drivelm_pref_val.jsonl',
]
for path in paths:
    with open(path, encoding='utf-8') as f:
        row = json.loads(next(f))
    assert row.get('rejected_response'), path
    missing = [p for p in row.get('images', []) if not os.path.exists(p)]
    print(path, 'first_row_images', len(row.get('images', [])), 'missing', len(missing))
    if missing:
        raise SystemExit(f'missing image: {missing[0]}')
PY
}

run_smoke() {
  log "starting P2 ORPO smoke from adapter: $SFT_ADAPTER"
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=200704 /root/miniconda3/bin/swift rlhf \
    --rlhf_type orpo \
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
    --max_length 2048 \
    --beta 0.1 \
    --output_dir "$PROJECT/checkpoints/smoke-qwen3b-orpo" \
    --warmup_ratio 0 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
}

run_train() {
  log "starting formal P2 ORPO: 1000 preference pairs / 200 val, 1 epoch"
  start_ts=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=200704 /root/miniconda3/bin/swift rlhf \
    --rlhf_type orpo \
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
    --max_length 2048 \
    --beta 0.1 \
    --output_dir "$TRAIN_OUT" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
  end_ts=$(date +%s)
  log "formal P2 ORPO finished in $((end_ts - start_ts)) seconds"
}

build_preferences
run_smoke
run_train
