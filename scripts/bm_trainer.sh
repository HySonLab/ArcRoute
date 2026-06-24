#!/usr/bin/env bash
# Train GRPO and PPO with identical hyperparameters, then compare outputs.
#
# All artifacts go to outputs/bm_<timestamp>/:
#   ckpt/grpo/   — GRPO checkpoints
#   ckpt/ppo/    — PPO checkpoints
#   logs/grpo.out, logs/ppo.out, logs/runner.out
#
# The entire pipeline runs in the background (nohup). Follow along with:
#   tail -f logs/bm_runner_<timestamp>.out
set -euo pipefail

cd "$(dirname "$0")/.."

# === SHARED HYPERPARAMETERS (same for GRPO and PPO) =========================
SEED=6868
MAX_EPOCH=40
BATCH_SIZE=128
MINI_BATCH_SIZE=128
TRAIN_DATA_SIZE=10000
VAL_DATA_SIZE=1000
TEST_DATA_SIZE=512
EMBED_DIM=128
NUM_ENCODER_LAYERS=6
NUM_HEADS=8
SIZES="20:40,30:60,40:80,50:100,40:120"
FLEET="2,3,5,7,10"
VARIANT=P
GROUP_SIZE=8        # GRPO only
ACCELERATOR=gpu
DEVICES=1
NUM_WORKERS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Eval params: single representative size from the ladder
EVAL_NUM_LOC=40
EVAL_NUM_ARC=80
EVAL_FLEET=3

# === OUTPUT DIRECTORIES =====================================================
TS=$(date +%Y%m%d_%H%M%S)
BM_DIR="outputs/bm_${TS}"
CKPT_GRPO="${BM_DIR}/ckpt/grpo"
CKPT_PPO="${BM_DIR}/ckpt/ppo"
DATA_DIR="${BM_DIR}/data"
LOG_DIR="${BM_DIR}/logs"

mkdir -p "$CKPT_GRPO" "$CKPT_PPO" "$DATA_DIR" "$LOG_DIR" logs

RUNNER_LOG="logs/bm_runner_${TS}.out"
GRPO_LOG="${LOG_DIR}/grpo.out"
PPO_LOG="${LOG_DIR}/ppo.out"
EVAL_LOG="${LOG_DIR}/eval.out"

DATA_TRAIN="${DATA_DIR}/train.data"
DATA_VAL="${DATA_DIR}/val.data"
DATA_TEST="${DATA_DIR}/test.data"

# Common args as a string (safe to embed in bash -c "...")
COMMON_ARGS="--seed $SEED \
 --max_epoch $MAX_EPOCH \
 --batch_size $BATCH_SIZE \
 --mini_batch_size $MINI_BATCH_SIZE \
 --train_data_size $TRAIN_DATA_SIZE \
 --val_data_size $VAL_DATA_SIZE \
 --embed_dim $EMBED_DIM \
 --num_encoder_layers $NUM_ENCODER_LAYERS \
 --num_heads $NUM_HEADS \
 --sizes '$SIZES' \
 --num_vehicle '$FLEET' \
 --variant $VARIANT \
 --accelerator $ACCELERATOR \
 --devices $DEVICES \
 --num_workers $NUM_WORKERS \
 --path_train_data '$DATA_TRAIN' \
 --path_val_data '$DATA_VAL' \
 --path_test_data '$DATA_TEST'"

echo "BM dir : $BM_DIR"
echo "Runner : $RUNNER_LOG"
echo "Follow : tail -f $RUNNER_LOG"

nohup bash -c "
set -euo pipefail

echo 'START \$(date)'
echo 'BM_DIR=$BM_DIR'
echo ''

# ---- GRPO ------------------------------------------------------------------
echo '=== [1/3] Training GRPO (max_epoch=$MAX_EPOCH) ==='
uv run python scripts/train.py \
    $COMMON_ARGS \
    --algo grpo \
    --group_size $GROUP_SIZE \
    --checkpoint_dir '$CKPT_GRPO' \
    > '$GRPO_LOG' 2>&1
echo 'GRPO DONE \$(date)'
echo ''

# Find best GRPO checkpoint (epoch=*.ckpt, fallback to last.ckpt)
GRPO_BEST=\$(ls '$CKPT_GRPO'/epoch=*.ckpt 2>/dev/null | head -1 || echo '$CKPT_GRPO/last.ckpt')

# ---- PPO -------------------------------------------------------------------
echo '=== [2/3] Training PPO  (max_epoch=$MAX_EPOCH) ==='
uv run python scripts/train.py \
    $COMMON_ARGS \
    --algo ppo \
    --checkpoint_dir '$CKPT_PPO' \
    > '$PPO_LOG' 2>&1
echo 'PPO DONE \$(date)'
echo ''

PPO_BEST=\$(ls '$CKPT_PPO'/epoch=*.ckpt 2>/dev/null | head -1 || echo '$CKPT_PPO/last.ckpt')

# ---- EVAL ------------------------------------------------------------------
echo '=== [3/3] Evaluating GRPO vs PPO ==='
echo \"GRPO ckpt : \$GRPO_BEST\"
echo \"PPO  ckpt : \$PPO_BEST\"
uv run python scripts/eval_bm.py \
    --grpo_ckpt \"\$GRPO_BEST\" \
    --ppo_ckpt  \"\$PPO_BEST\" \
    --num_loc   $EVAL_NUM_LOC \
    --num_arc   $EVAL_NUM_ARC \
    --fleet     $EVAL_FLEET \
    --embed_dim $EMBED_DIM \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_heads $NUM_HEADS \
    --num_test  $TEST_DATA_SIZE \
    --batch_size $BATCH_SIZE \
    --variant   $VARIANT \
    2>&1 | tee '$EVAL_LOG'

echo ''
echo 'ALL DONE \$(date)'
echo \"Eval saved to $EVAL_LOG\"
" > "$RUNNER_LOG" 2>&1 &

echo "Training started (PID $!)"
echo "Theo doi: tail -f $RUNNER_LOG"
