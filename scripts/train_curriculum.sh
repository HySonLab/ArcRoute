#!/usr/bin/env bash
# Sequential 3-phase curriculum training for HDCARP-GRPO.
#
#   Phase 1 (small):  data/bench_small_{train,val}.data   — 200 epochs, from scratch
#   Phase 2 (medium): data/bench_medium_{train,val}.data  — 200 epochs, warm-start best phase 1
#   Phase 3 (large):  data/bench_large_{train,val}.data   — 200 epochs, warm-start best phase 2
#
# Pre-generate benchmark data first:
#   uv run python scripts/gen_data_benchmark.py
#
# Warm-start loads policy weights only (epoch + optimizer reset so each phase
# trains 200 fresh epochs on its own distribution).
#
# Usage:
#   bash scripts/train_curriculum.sh                        # all 3 phases
#   PHASES="2 3" bash scripts/train_curriculum.sh           # skip phase 1 (needs phase 1 ckpt)
#   PHASES="3" RESUME_P3=path/to/ckpt.ckpt bash scripts/train_curriculum.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PHASES="${PHASES:-1 2 3}"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_curriculum_${TS}.out"

# Re-launch self under nohup on first call so terminal close is safe.
if [ -z "${_CL_BG:-}" ]; then
    echo "Curriculum phases: $PHASES"
    echo "Log: $LOG"
    echo "Theo doi: tail -f $LOG"
    _CL_BG=1 PHASES="$PHASES" nohup bash "$0" > "$LOG" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# ---------------------------------------------------------------------------
# Hyperparameters (shared across all phases)
# ---------------------------------------------------------------------------
SEED=6868
GROUP_SIZE=16
FLEET="2,3,5,7,10"
VARIANT=P
NUM_LOC=40
NUM_ARC=80
MAX_EPOCH=200
BATCH_SIZE=2048
MINI_BATCH_SIZE=512
EMBED_DIM=128
NUM_ENCODER_LAYERS=12
NUM_HEADS=8
ACCELERATOR=gpu
DEVICES=1

DIR_SMALL=outputs/checkpoints/curriculum_small
DIR_MEDIUM=outputs/checkpoints/curriculum_medium
DIR_LARGE=outputs/checkpoints/curriculum_large

# ---------------------------------------------------------------------------
# best_ckpt DIR — returns best epoch=NNN.ckpt, falls back to last.ckpt
# ---------------------------------------------------------------------------
best_ckpt() {
    local dir="$1"
    local f
    f=$(ls "$dir"/epoch=*.ckpt 2>/dev/null | sort -V | tail -1)
    if [ -n "$f" ]; then
        echo "$f"
    elif [ -f "$dir/last.ckpt" ]; then
        echo "$dir/last.ckpt"
    else
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# run_phase LABEL VAL_DATA SIZES CKPT_DIR RESUME
# Train data is regenerated on-the-fly; only val is fixed benchmark data.
# ---------------------------------------------------------------------------
run_phase() {
    local label="$1" val_data="$2" sizes="$3" ckpt_dir="$4" resume="$5"

    echo ""
    echo "============================================================"
    echo "  PHASE    : $label"
    echo "  val_data : $val_data"
    echo "  sizes    : $sizes"
    echo "  epochs   : $MAX_EPOCH"
    [ -n "$resume" ] \
        && echo "  resume   : $resume" \
        || echo "  resume   : (scratch)"
    echo "============================================================"

    [ -f "$val_data" ] || { echo "ERROR: val data not found: $val_data"; echo "Run: uv run python scripts/gen_data_benchmark.py"; exit 1; }

    # Clear cached train data so generator rebuilds for this phase's sizes.
    rm -f data/train_data.data

    RESUME_ARGS=()
    [ -n "$resume" ] && RESUME_ARGS=(--resume_from "$resume")

    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/train.py \
        --seed               "$SEED" \
        --max_epoch          "$MAX_EPOCH" \
        --batch_size         "$BATCH_SIZE" \
        --mini_batch_size    "$MINI_BATCH_SIZE" \
        --train_data_size    100000 \
        --val_data_size      10000 \
        --embed_dim          "$EMBED_DIM" \
        --num_encoder_layers "$NUM_ENCODER_LAYERS" \
        --num_heads          "$NUM_HEADS" \
        --num_loc            "$NUM_LOC" \
        --num_arc            "$NUM_ARC" \
        --num_vehicle        "$FLEET" \
        --sizes              "$sizes" \
        --variant            "$VARIANT" \
        --group_size         "$GROUP_SIZE" \
        --path_val_data      "$val_data" \
        --checkpoint_dir     "$ckpt_dir" \
        --accelerator        "$ACCELERATOR" \
        --devices            "$DEVICES" \
        "${RESUME_ARGS[@]}"

    local best
    best=$(best_ckpt "$ckpt_dir")
    echo ""
    echo "Phase $label done. Best checkpoint: ${best:-(none saved yet)}"
}

# ---------------------------------------------------------------------------
# Run phases
# ---------------------------------------------------------------------------
for phase in $PHASES; do
    case "$phase" in
    1)
        run_phase "curriculum_small" \
            "data/bench_small_val.data" \
            "20:40" \
            "$DIR_SMALL" \
            ""
        ;;
    2)
        RESUME="${RESUME_P2:-$(best_ckpt "$DIR_SMALL")}"
        [ -n "$RESUME" ] || { echo "ERROR: no checkpoint in $DIR_SMALL — run phase 1 first."; exit 1; }
        run_phase "curriculum_medium" \
            "data/bench_medium_val.data" \
            "20:40,30:60,40:80" \
            "$DIR_MEDIUM" \
            "$RESUME"
        ;;
    3)
        RESUME="${RESUME_P3:-$(best_ckpt "$DIR_MEDIUM")}"
        [ -n "$RESUME" ] || { echo "ERROR: no checkpoint in $DIR_MEDIUM — run phase 2 first."; exit 1; }
        run_phase "curriculum_large" \
            "data/bench_large_val.data" \
            "20:40,30:60,40:80,50:100,40:120" \
            "$DIR_LARGE" \
            "$RESUME"
        ;;
    *)
        echo "ERROR: unknown phase '$phase' (valid: 1 2 3)"
        exit 1
        ;;
    esac
done

echo ""
echo "All phases complete."
FINAL=$(best_ckpt "$DIR_LARGE")
echo "Final best checkpoint: ${FINAL:-(none)}"
