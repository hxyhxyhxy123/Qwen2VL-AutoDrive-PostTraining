#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/AutoDrive-VLM-RLBench}
DATA_RAW="$PROJECT/data/raw/drivelm_nuscenes"
DATA_OUT="$PROJECT/data/processed"
TRAIN_JSON="$DATA_RAW/v1_1_train_nus.json"
MODEL="$PROJECT/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_OUT="$PROJECT/checkpoints/qwen3b-lora-sft"
LOG_DIR="$PROJECT/outputs/training"
mkdir -p "$DATA_OUT" "$TRAIN_OUT" "$LOG_DIR"

TRAIN_ZIP_SIZE=3483205396
VAL_ZIP_SIZE=704864335

log() {
  echo "[$(date -Is)] $*"
}

file_size() {
  if [ -f "$1" ]; then
    stat -c '%s' "$1"
  else
    echo 0
  fi
}

wait_for_downloads() {
  log "waiting for DriveLM downloads"
  while true; do
    train_size=$(file_size "$DATA_RAW/drivelm_nus_imgs_train.zip")
    val_size=$(file_size "$DATA_RAW/drivelm_nus_imgs_val.zip")
    if [ "$train_size" -ge "$TRAIN_ZIP_SIZE" ] && [ "$val_size" -ge "$VAL_ZIP_SIZE" ]; then
      log "DriveLM zip files present: train=$train_size val=$val_size"
      break
    fi
    if ! pgrep -f 'download_drivelm_v11_robust.sh|download_drivelm_v11.sh|OpenDriveLab/DriveLM' >/dev/null 2>&1; then
      log "download process not visible yet; continuing to wait for files"
    fi
    sleep 60
  done
}

extract_images() {
  log "extracting DriveLM train images"
  unzip -q -n "$DATA_RAW/drivelm_nus_imgs_train.zip" -d "$DATA_RAW"
  log "extracting DriveLM val images"
  unzip -q -n "$DATA_RAW/drivelm_nus_imgs_val.zip" -d "$DATA_RAW"
}

build_data() {
  log "building balanced SFT datasets"
  /root/miniconda3/bin/python "$PROJECT/src/data/build_drivelm_swift.py" \
    --train-json "$TRAIN_JSON" \
    --image-root "$DATA_RAW/nuscenes" \
    --out-dir "$DATA_OUT" \
    --train-size 3000 \
    --val-size 500 \
    --seed 42 | tee "$LOG_DIR/build_drivelm_swift.json"

  head -n 32 "$DATA_OUT/drivelm_sft_train.jsonl" > "$DATA_OUT/drivelm_sft_smoke_train.jsonl"
  head -n 16 "$DATA_OUT/drivelm_sft_val.jsonl" > "$DATA_OUT/drivelm_sft_smoke_val.jsonl"

  /root/miniconda3/bin/python - <<'PY'
import json, os
paths = [
    '/root/autodl-tmp/AutoDrive-VLM-RLBench/data/processed/drivelm_sft_train.jsonl',
    '/root/autodl-tmp/AutoDrive-VLM-RLBench/data/processed/drivelm_sft_val.jsonl',
]
for path in paths:
    with open(path, encoding='utf-8') as f:
        row = json.loads(next(f))
    missing = [p for p in row['images'] if not os.path.exists(p)]
    print(path, 'first_row_images', len(row['images']), 'missing', len(missing))
    if missing:
        raise SystemExit(f'missing image: {missing[0]}')
PY
}

run_smoke() {
  log "starting 32-sample smoke LoRA-SFT"
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=200704 /root/miniconda3/bin/swift sft \
    --model "$MODEL" \
    --dataset "$DATA_OUT/drivelm_sft_smoke_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_sft_smoke_val.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
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
    --output_dir "$PROJECT/checkpoints/smoke-qwen3b-lora" \
    --warmup_ratio 0 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2
}

run_train() {
  log "starting formal P1 LoRA-SFT: 3000 train / 500 val, 1 epoch"
  start_ts=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=200704 /root/miniconda3/bin/swift sft \
    --model "$MODEL" \
    --dataset "$DATA_OUT/drivelm_sft_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_sft_val.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$TRAIN_OUT" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
  end_ts=$(date +%s)
  log "formal P1 LoRA-SFT finished in $((end_ts - start_ts)) seconds"
}

wait_for_downloads
extract_images
build_data
run_smoke
run_train
