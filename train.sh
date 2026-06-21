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
# Paper F1: |A| = n*d, d in {1.5,2,2.5,3}. Here n=40, d=2 -> |A|=80, and the
# ¼-split (F2) gives |A_r| = 3*floor(80/4) = 60 required arcs (paper's main cfg).
# Constraint (Phase 0 §0.6): NUM_ARC >= NUM_LOC (a Hamiltonian cycle needs n arcs).
NUM_LOC=40
NUM_ARC=80
VARIANT=P
CHECKPOINT_DIR=checkpoints/cl123_Ar60
ACCELERATOR=gpu
DEVICES=1

uv run python train.py \
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
    --variant "$VARIANT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES"
