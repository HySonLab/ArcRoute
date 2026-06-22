#!/usr/bin/env bash
# Launch HDCARP RL training via uv (nohup -> logs/). Edit hyperparameters below.
set -euo pipefail

cd "$(dirname "$0")"

# === MODE ===================================================================
#   "validate" : short sanity run (~1-2h) to confirm the learning curve drops
#                before committing to a full run. WATCH logs/, stop early once
#                train/val reward is clearly improving (save_last keeps a ckpt).
#   "full"     : the real training run.
# Override on the CLI:   MODE=full ./train.sh
MODE="${MODE:-validate}"

# --- shared across modes ---
SEED=6868
NUM_HEADS=8
# D2 Phase 4: ALGO selects the learning signal — "grpo" (group lexicographic rank,
# critic-free; reward_mode auto=vector) or "ppo" (old weighted-sum + critic) for
# A/B. GROUP_SIZE = K samples/instance for GRPO. Override:  ALGO=ppo ./train.sh
ALGO="${ALGO:-grpo}"
GROUP_SIZE="${GROUP_SIZE:-8}"
# Phase 6: train over a SIZE LADDER (bucketed; each batch stays single-size).
#   "nloc:narc"; ¼-split gives |A_r| = 3*floor(narc/4): 30,45,60,75,90 (<=100).
SIZES="20:40,30:60,40:80,50:100,40:120"
# Phase 3: FLEET sweeps M PER-INSTANCE in the reward (Scheduler). The policy is
# M-agnostic, so one model serves any M; M mixes freely in a batch (scalar).
FLEET="2,3,5,7,10"
VARIANT=P                 # Phase 0: hierarchical/HDCARP-P is the default
NUM_LOC=40                # single-size fallback (only used if SIZES is empty)
NUM_ARC=80
ACCELERATOR=gpu
DEVICES=1

# --- per-mode hyperparameters ---
if [ "$MODE" = "validate" ]; then
    MAX_EPOCH=40
    # GRPO's effective batch is BATCH_SIZE * GROUP_SIZE; keep it modest at validate.
    BATCH_SIZE=128
    MINI_BATCH_SIZE=128
    TRAIN_DATA_SIZE=10000
    VAL_DATA_SIZE=1000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=6      # lighter than full (12) so epochs are fast
    CHECKPOINT_DIR=checkpoints/validate_clP
else
    MAX_EPOCH=1000
    BATCH_SIZE=4096
    MINI_BATCH_SIZE=512
    TRAIN_DATA_SIZE=100000
    VAL_DATA_SIZE=10000
    EMBED_DIM=128
    NUM_ENCODER_LAYERS=12
    CHECKPOINT_DIR=checkpoints/clP_ladder
fi

echo "MODE=$MODE | algo=$ALGO | group_size=$GROUP_SIZE | max_epoch=$MAX_EPOCH | "\
"train_data=$TRAIN_DATA_SIZE | layers=$NUM_ENCODER_LAYERS | fleet=$FLEET | variant=$VARIANT"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_${MODE}_${TS}.out"

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
    --num_vehicle "$FLEET" \
    --sizes "$SIZES" \
    --variant "$VARIANT" \
    --algo "$ALGO" \
    --group_size "$GROUP_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    > "$LOG" 2>&1 &

echo "Training ($MODE) started (PID $!)"
echo "Log: $LOG"
echo "Theo doi: tail -f $LOG"
