#!/usr/bin/env bash
# Launch HDCARP GRPO training on OSM data (nohup -> logs/).
# Usage:
#   bash scripts/train.sh
#   GROUP_SIZE=16 bash scripts/train.sh
#   RESUME_FROM=outputs/checkpoints/grpo_osm/last.ckpt bash scripts/train.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# === Hyperparameters =========================================================
SEED=6868
MAX_EPOCH="${MAX_EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-256}"
EMBED_DIM=128
NUM_ENCODER_LAYERS=12
NUM_HEADS=8
FLEET="2,3,5,7,10"
VARIANT="${VARIANT:-P}"
GROUP_SIZE="${GROUP_SIZE:-16}"
CURRICULUM="${CURRICULUM:-small}" # small | medium | large | (empty = all)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/grpo_osm}"
ACCELERATOR=gpu
DEVICES=1

PATH_TRAIN_DATA=data/osm_train
PATH_VAL_DATA=data/osm_val
PATH_TEST_DATA=data/osm_test

RESUME_FROM="${RESUME_FROM:-}"

echo "group_size=$GROUP_SIZE | max_epoch=$MAX_EPOCH | batch=$BATCH_SIZE | layers=$NUM_ENCODER_LAYERS | fleet=$FLEET | variant=$VARIANT"
[ -n "$CURRICULUM" ] && echo "curriculum=$CURRICULUM" || echo "curriculum=all"
[ -n "$RESUME_FROM" ] && echo "resume_from=$RESUME_FROM" || echo "resume_from=(none)"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_${TS}.out"

CURRICULUM_ARGS=()
[ -n "$CURRICULUM" ] && CURRICULUM_ARGS=(--curriculum "$CURRICULUM")

RESUME_ARGS=()
[ -n "$RESUME_FROM" ] && RESUME_ARGS=(--resume_from "$RESUME_FROM")

nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python src/train.py \
    --seed "$SEED" \
    --max_epoch "$MAX_EPOCH" \
    --batch_size "$BATCH_SIZE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --num_encoder_layers "$NUM_ENCODER_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --num_vehicle "$FLEET" \
    --variant "$VARIANT" \
    --group_size "$GROUP_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --path_train_data "$PATH_TRAIN_DATA" \
    --path_val_data "$PATH_VAL_DATA" \
    --path_test_data "$PATH_TEST_DATA" \
    "${CURRICULUM_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    > "$LOG" 2>&1 &

echo "Training started (PID $!)"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
