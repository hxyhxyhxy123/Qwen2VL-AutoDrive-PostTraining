#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/AutoDrive-VLM-RLBench}
DATA_OUT="$PROJECT/data/processed"
DATA_RAW="$PROJECT/data/raw/drivelm_nuscenes"
MODEL="$PROJECT/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
LOG_DIR="$PROJECT/outputs/training"
TRAIN_OUT="$PROJECT/checkpoints/qwen3b-lora-gspo-sftstart"
PLUGIN="$PROJECT/src/train/drivelm_reward_plugin.py"
STATUS="$PROJECT/outputs/pipeline/sft_gspo_ablation_status.json"
mkdir -p "$DATA_OUT" "$LOG_DIR" "$TRAIN_OUT" "$PROJECT/outputs/pipeline" "$PROJECT/outputs/eval"

log() {
  echo "[$(date -Is)] $*"
}

write_status() {
  local stage="$1"
  local detail="${2:-running}"
  /root/miniconda3/bin/python - <<PY
import json, time
payload = {"stage": "$stage", "detail": "$detail", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
with open("$STATUS", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False))
PY
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
ORPO_ADAPTER=${ORPO_ADAPTER:-$(latest_checkpoint "$PROJECT/checkpoints/qwen3b-lora-orpo")}
if [ -z "${SFT_ADAPTER:-}" ] || [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Cannot find SFT adapter" >&2
  exit 1
fi
if [ -z "${ORPO_ADAPTER:-}" ] || [ ! -f "$ORPO_ADAPTER/adapter_model.safetensors" ]; then
  echo "Cannot find ORPO adapter for eval comparison" >&2
  exit 1
fi

build_rollout_data() {
  write_status "build_sft_gspo_rollout" "500 train / 100 val"
  /root/miniconda3/bin/python "$PROJECT/src/data/build_gspo_rollout.py" \
    --sft-train "$DATA_OUT/drivelm_sft_train.jsonl" \
    --sft-val "$DATA_OUT/drivelm_sft_val.jsonl" \
    --out-dir "$DATA_OUT" \
    --train-size "${SFT_GSPO_TRAIN_SIZE:-500}" \
    --val-size "${SFT_GSPO_VAL_SIZE:-100}" \
    --seed 2718 | tee "$LOG_DIR/build_drivelm_sft_gspo.json"

  head -n 8 "$DATA_OUT/drivelm_gspo_train.jsonl" > "$DATA_OUT/drivelm_gspo_sftstart_smoke_train.jsonl"
  head -n 4 "$DATA_OUT/drivelm_gspo_val.jsonl" > "$DATA_OUT/drivelm_gspo_sftstart_smoke_val.jsonl"
}

common_args() {
  cat <<ARGS
--rlhf_type grpo
--model $MODEL
--adapters $SFT_ADAPTER
--external_plugins $PLUGIN
--reward_funcs drivelm_soft_format
--tuner_type lora
--torch_dtype bfloat16
--lora_rank 16
--lora_alpha 32
--target_modules all-linear
--freeze_vit true
--freeze_aligner true
--max_length 1536
--max_completion_length 96
--num_generations ${SFT_GSPO_NUM_GENERATIONS:-4}
--temperature 1.0
--top_p 0.9
--learning_rate 1e-6
--beta 0.0
--epsilon 3e-4
--epsilon_high 4e-4
--importance_sampling_level sequence
--steps_per_generation ${SFT_GSPO_STEPS_PER_GENERATION:-4}
--advantage_estimator grpo
--loss_type grpo
--log_completions true
--dataloader_num_workers 4
--dataset_num_proc 4
ARGS
}

run_smoke() {
  write_status "sft_gspo_smoke" "running"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS="${MAX_PIXELS:-100352}" /root/miniconda3/bin/swift rlhf \
    $(common_args) \
    --dataset "$DATA_OUT/drivelm_gspo_sftstart_smoke_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_gspo_sftstart_smoke_val.jsonl" \
    --num_train_epochs 1 \
    --max_steps 2 \
    --per_device_train_batch_size "${SFT_GSPO_SMOKE_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${SFT_GSPO_SMOKE_EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps 1 \
    --eval_steps 1 \
    --save_steps 2 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir "$PROJECT/checkpoints/smoke-qwen3b-gspo-sftstart" \
    --warmup_ratio 0
}

run_train() {
  write_status "sft_gspo_train" "running"
  start_ts=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS="${MAX_PIXELS:-100352}" /root/miniconda3/bin/swift rlhf \
    $(common_args) \
    --dataset "$DATA_OUT/drivelm_gspo_train.jsonl" \
    --val_dataset "$DATA_OUT/drivelm_gspo_val.jsonl" \
    --num_train_epochs 1 \
    --max_steps "${SFT_GSPO_MAX_STEPS:-60}" \
    --per_device_train_batch_size "${SFT_GSPO_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${SFT_GSPO_EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${SFT_GSPO_GRAD_ACCUM:-2}" \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 3 \
    --logging_steps 2 \
    --output_dir "$TRAIN_OUT" \
    --warmup_ratio 0.03
  end_ts=$(date +%s)
  write_status "sft_gspo_train" "completed in $((end_ts - start_ts)) seconds"
}

log "SFT->GSPO ablation from $SFT_ADAPTER"
build_rollout_data
run_smoke
run_train
adapter=$(latest_checkpoint "$TRAIN_OUT")
write_status "completed" "adapter=$adapter; evaluate with scripts/run_strict_eval_pref_models_after_current.sh"
