#!/usr/bin/env bash
# Launch HDCARP RL training via uv.
# Edit the hyperparameters below directly.
set -euo pipefail

cd "$(dirname "$0")"

# --- Hyperparameters ---
SEED=6868
MAX_EPOCH=1000
BATCH_SIZE=4096
MINI_BATCH_SIZE=512
TRAIN_DATA_SIZE=100000
VAL_DATA_SIZE=10000
EMBED_DIM=128
NUM_ENCODER_LAYERS=12
NUM_HEADS=8
# Phase 6: train over a SIZE LADDER (bucketed; each batch stays single-size).
# Pairs are "nloc:narc"; ¼-split gives |A_r| = 3*floor(narc/4):
#   20:40->30  30:60->45  40:80->60  50:100->75  40:120->90   (all <= cap 100)
# This is what gives size-generalization. NUM_LOC/NUM_ARC are the single-size
# fallback (used only when SIZES is empty). Fleet M is fixed (policy isn't
# M-conditioned, so sweeping M would be noise) and topology stays unit_square.
SIZES="20:40,30:60,40:80,50:100,40:120"
NUM_LOC=40
NUM_ARC=80
VARIANT=P
CHECKPOINT_DIR=checkpoints/clP_ladder
ACCELERATOR=gpu
DEVICES=1

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_${TS}.out"

# Run in the background via nohup so training survives a terminal close.
nohup uv run python train.py \
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
    --sizes "$SIZES" \
    --variant "$VARIANT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    > "$LOG" 2>&1 &

echo "Training started (PID $!)"
echo "Log: $LOG"
echo "Theo doi: tail -f $LOG"
