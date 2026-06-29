#!/usr/bin/env bash
# Launch HDCARP RL training via uv (nohup -> logs/). Edit hyperparameters below.
set -euo pipefail

cd "$(dirname "$0")/.."

# === MODE ===================================================================
#   "validate"          : short sanity run to confirm learning curve drops.
#   "curriculum_small"  : phase 1 — small instances only (20:40).
#   "curriculum_medium" : phase 2 — small + medium; warm-starts from phase 1.
#   "curriculum_large"  : phase 3 — full ladder; warm-starts from phase 2.
#   "full"              : single full-scale run (no curriculum).
# Override: MODE=curriculum_small bash scripts/train.sh
# Warm-start: RESUME_FROM=outputs/checkpoints/curriculum_small/epoch\=166.ckpt MODE=curriculum_medium bash scripts/train.sh
MODE="${MODE:-validate}"

# --- shared across modes ---
SEED=6868
NUM_HEADS=8
GROUP_SIZE="${GROUP_SIZE:-16}"
FLEET="2,3,5,7,10"
VARIANT=P
NUM_LOC=40        # single-size fallback (unused when SIZES is set)
NUM_ARC=80
ACCELERATOR=gpu
DEVICES=1

# --- per-mode hyperparameters ---
if [ "$MODE" = "validate" ]; then
    MAX_EPOCH=40
    BATCH_SIZE=128
    MINI_BATCH_SIZE=128
    TRAIN_DATA_SIZE=10000
    VAL_DATA_SIZE=1000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=6
    SIZES="20:40,30:60,40:80,50:100,40:120"
    CHECKPOINT_DIR=outputs/checkpoints/validate_clP
    RESUME_FROM="${RESUME_FROM:-}"

elif [ "$MODE" = "curriculum_small" ]; then
    MAX_EPOCH=200
    BATCH_SIZE=2048
    MINI_BATCH_SIZE=512
    TRAIN_DATA_SIZE=100000
    VAL_DATA_SIZE=10000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=12
    SIZES="20:40"
    CHECKPOINT_DIR=outputs/checkpoints/curriculum_small
    PATH_VAL_DATA="${PATH_VAL_DATA:-data/bench_small_val.data}"
    RESUME_FROM="${RESUME_FROM:-}"

elif [ "$MODE" = "curriculum_medium" ]; then
    MAX_EPOCH=200
    BATCH_SIZE=4096
    MINI_BATCH_SIZE=1024
    TRAIN_DATA_SIZE=100000
    VAL_DATA_SIZE=10000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=12
    SIZES="20:40,30:60,40:80"
    CHECKPOINT_DIR=outputs/checkpoints/curriculum_medium
    RESUME_FROM="${RESUME_FROM:-outputs/checkpoints/curriculum_small/last.ckpt}"

elif [ "$MODE" = "curriculum_large" ]; then
    MAX_EPOCH=200
    BATCH_SIZE=4096
    MINI_BATCH_SIZE=1024
    TRAIN_DATA_SIZE=100000
    VAL_DATA_SIZE=10000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=12
    SIZES="20:40,30:60,40:80,50:100,40:120"
    CHECKPOINT_DIR=outputs/checkpoints/curriculum_large
    RESUME_FROM="${RESUME_FROM:-outputs/checkpoints/curriculum_medium/last.ckpt}"

else  # full
    MAX_EPOCH=100
    BATCH_SIZE=4096
    MINI_BATCH_SIZE=1024
    TRAIN_DATA_SIZE=100000
    VAL_DATA_SIZE=10000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=12
    SIZES="20:40,30:60,40:80,50:100,40:120"
    CHECKPOINT_DIR=outputs/checkpoints/clP_ladder
    RESUME_FROM="${RESUME_FROM:-}"
fi

echo "MODE=$MODE | group_size=$GROUP_SIZE | max_epoch=$MAX_EPOCH | sizes=$SIZES"
echo "batch=$BATCH_SIZE | train_data=$TRAIN_DATA_SIZE | layers=$NUM_ENCODER_LAYERS | fleet=$FLEET"
[ -n "$RESUME_FROM" ] && echo "resume_from=$RESUME_FROM" || echo "resume_from=(none)"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_${MODE}_${TS}.out"

# Build optional resume arg
RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS=(--resume_from "$RESUME_FROM")
fi

# Run in the background via nohup so training survives a terminal close.
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py \
    --seed "$SEED" \
    --max_epoch "$MAX_EPOCH" \
    --batch_size "$BATCH_SIZE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --num_encoder_layers "$NUM_ENCODER_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --num_loc "$NUM_LOC" \
    --num_arc "$NUM_ARC" \
    --num_vehicle "$FLEET" \
    --sizes "$SIZES" \
    --variant "$VARIANT" \
    --group_size "$GROUP_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    ${PATH_VAL_DATA:+--path_val_data "$PATH_VAL_DATA"} \
    "${RESUME_ARGS[@]}" \
    > "$LOG" 2>&1 &

echo "Training ($MODE) started (PID $!)"
echo "Log: $LOG"
echo "Theo doi: tail -f $LOG"
