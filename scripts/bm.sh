#!/usr/bin/env bash
# bm.sh — Benchmark entry point for all HDCARP solvers.
#
# Usage: SOLVER=<name> [overrides] bash scripts/bm.sh
#
# Solver       Command
# ----------   -------------------------------------------------------
# ea           Evolutionary Algorithm (EAHCARP)
# ils          Iterated Local Search (ILSHCARP)
# aco          Ant Colony Optimisation (ACOHCARP)
# lp           MILP exact solver via SCIP
# rl           RL policy (GRPO/PPO checkpoint)
#
# Required env vars:
#   SOLVER     one of: ea ils aco lp rl
#
# Common overrides (env vars, all optional):
#   DATA_DIR   path to benchmark directory containing <size>/<file>.npz
#              (default: data/5m)
#   VARIANT    P or U  (default: P)
#   SEED       random seed  (default: 6868)
#   M          fleet size override; omit to use instance default
#
# Solver-specific overrides:
#   EA / ACO
#     N_EPOCH        (default: 100)
#     N_POPULATION   EA only  (default: 200)
#     N_ANT          ACO only (default: 50)
#
#   ILS
#     MAX_ITER        (default: 200)
#     NUM_INIT_SAMPLE (default: 5)
#     STRENGTH        (default: 3)
#     ACCEPT_MODE     best|sa  (default: best)
#
#   LP
#     TIME_LIMIT   seconds per instance  (default: 3600)
#     THREADS      SCIP threads          (default: 8) — currently a NO-OP:
#                  community SCIP branch-and-bound is single-threaded; see
#                  the note in src/solvers/lp.py::_configure.
#
#   RL
#     CKPT        checkpoint .ckpt file  (required for rl)
#     NUM_SAMPLE  beam width             (default: 100)
#     EPS         eps-lex T1 tolerance for shortlisting (default: 0.005 = 0.5%)
#     HYBRID      1 = two-tier funnel: eps-lex shortlist + local search on it (default: 0)
#     SHORTLIST   HYBRID=1 only: #candidates that get local search (default: 10)
#     TOPK_T1     HYBRID=1 only: extra #candidates by raw T1 rank, local-searched too,
#                 on top of the eps-lex shortlist (default: 0 = disabled). Pairs with a
#                 wider NUM_SAMPLE -- e.g. NUM_SAMPLE=400 TOPK_T1=100 measured 82.75%->
#                 95.31% "T1 not worse than ILS" on osm_test_small, see logs/status_bm.md.
#     ORACLE_HEAD HYBRID=1 only: 1 = add the exact-T1-oracle forced-prefix arm
#                 (variant P) unioned with the pool arm (default: 0 = disabled).
#                 Sub-tunables (sample count, shortlist sizes, eps-bands, tiebreak)
#                 aren't exposed here -- historical sweeps never varied them from
#                 RLHybHCARP's defaults; construct it directly in Python if needed.
#                 Measured on osm_test_small (on top of NUM_SAMPLE=400 TOPK_T1=100):
#                 T1-not-worse-than-ILS 95.31%->100%, T2 52.9%->86.0%, lex-win
#                 86.4%->98.3%, at ~3.4x ILS speed.
#
# Output:
#   Prints solver lines to stdout AND saves to logs/bm_<solver>_<variant>_<ts>.out
#
# Examples:
#   SOLVER=ils bash scripts/bm.sh
#   SOLVER=ea DATA_DIR=data/2m VARIANT=U N_EPOCH=200 bash scripts/bm.sh
#   SOLVER=lp TIME_LIMIT=300 THREADS=4 bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/last.ckpt NUM_SAMPLE=200 bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/last.ckpt HYBRID=1 NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01 TOPK_T1=100 bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/last.ckpt HYBRID=1 SHORTLIST=10 bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt HYBRID=1 NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01 TOPK_T1=100 ORACLE_HEAD=1 bash scripts/bm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ── required ──────────────────────────────────────────────────────────────────
SOLVER="${SOLVER:-}"
if [[ -z "$SOLVER" ]]; then
    echo "Error: SOLVER is required. Set SOLVER=ea|ils|aco|lp|rl"
    exit 1
fi

# ── common ────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/5m}"
VARIANT="${VARIANT:-P}"
SEED="${SEED:-6868}"
M="${M:-}"

# ── ea / aco ──────────────────────────────────────────────────────────────────
N_EPOCH="${N_EPOCH:-100}"
N_POPULATION="${N_POPULATION:-200}"
N_ANT="${N_ANT:-50}"

# ── ils ───────────────────────────────────────────────────────────────────────
MAX_ITER="${MAX_ITER:-200}"
NUM_INIT_SAMPLE="${NUM_INIT_SAMPLE:-5}"
STRENGTH="${STRENGTH:-3}"
ACCEPT_MODE="${ACCEPT_MODE:-best}"

# ── lp ────────────────────────────────────────────────────────────────────────
TIME_LIMIT="${TIME_LIMIT:-3600}"
THREADS="${THREADS:-8}"

# ── rl ────────────────────────────────────────────────────────────────────────
CKPT="${CKPT:-}"
NUM_SAMPLE="${NUM_SAMPLE:-100}"
EPS="${EPS:-0.005}"
HYBRID="${HYBRID:-0}"          # 1 = two-tier funnel (eps-lex shortlist + local search)
SHORTLIST="${SHORTLIST:-10}"   # HYBRID=1 only: #candidates that get local search
TOPK_T1="${TOPK_T1:-0}"        # HYBRID=1 only: extra #candidates by raw T1 rank, LS'd too
                                # (0 = disabled; e.g. NUM_SAMPLE=400 TOPK_T1=100 is "bậc 3 + P1")
ORACLE_HEAD="${ORACLE_HEAD:-0}"                    # HYBRID=1 only: exact-T1-oracle arm

# ── output ────────────────────────────────────────────────────────────────────
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
OUT="logs/bm_${SOLVER}_${VARIANT}_${TS}.out"

echo "Solver   : $SOLVER"
echo "Data     : $DATA_DIR"
echo "Variant  : $VARIANT"
echo "Log      : $OUT"
echo

M_ARGS=()
[[ -n "$M" ]] && M_ARGS=(--M "$M")

# ── dispatch ──────────────────────────────────────────────────────────────────
case "$SOLVER" in

  ea)
    uv run python -m solvers.ea \
        --seed "$SEED" --variant "$VARIANT" \
        --path "$DATA_DIR" \
        --n_epoch "$N_EPOCH" --n_population "$N_POPULATION" \
        "${M_ARGS[@]}" | tee "$OUT"
    ;;

  ils)
    uv run python -m solvers.ils \
        --seed "$SEED" --variant "$VARIANT" \
        --path "$DATA_DIR" \
        --max_iter "$MAX_ITER" --num_init_sample "$NUM_INIT_SAMPLE" \
        --strength "$STRENGTH" --accept_mode "$ACCEPT_MODE" \
        "${M_ARGS[@]}" | tee "$OUT"
    ;;

  aco)
    uv run python -m solvers.aco \
        --seed "$SEED" --variant "$VARIANT" \
        --path "$DATA_DIR" \
        --n_epoch "$N_EPOCH" --n_ant "$N_ANT" \
        "${M_ARGS[@]}" | tee "$OUT"
    ;;

  lp)
    # lp.py takes a single --path <file>; loop over the benchmark directory here.
    shopt -s nullglob
    files=("$DATA_DIR"/*/*.npz)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Error: no .npz files found under $DATA_DIR/*/*.npz" >&2
        exit 1
    fi
    echo "Found ${#files[@]} instances" | tee "$OUT"
    for f in "${files[@]}"; do
        uv run python -m solvers.lp \
            --path "$f" --variant "$VARIANT" \
            --time_limit "$TIME_LIMIT" --threads "$THREADS"
    done | tee -a "$OUT"
    ;;

  rl)
    if [[ -z "$CKPT" ]]; then
        echo "Error: CKPT is required for the rl solver." >&2
        echo "  Set CKPT=outputs/checkpoints/last.ckpt" >&2
        exit 1
    fi
    HYBRID_ARGS=()
    [[ "$HYBRID" == "1" ]] && HYBRID_ARGS=(--hybrid --shortlist "$SHORTLIST" --topk_t1 "$TOPK_T1")
    [[ "$ORACLE_HEAD" == "1" ]] && HYBRID_ARGS+=(--oracle_head)
    uv run python -m solvers.rl \
        --seed "$SEED" --variant "$VARIANT" \
        --ckpt "$CKPT" --path "$DATA_DIR" \
        --num_sample "$NUM_SAMPLE" --eps "$EPS" \
        "${M_ARGS[@]}" "${HYBRID_ARGS[@]}" | tee "$OUT"
    ;;

  *)
    echo "Unknown solver: $SOLVER" >&2
    echo "Must be one of: ea ils aco lp rl" >&2
    exit 1
    ;;

esac

echo
echo "Done. Results saved to $OUT"
