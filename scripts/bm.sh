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
#   DATA_DIR   path to a benchmark directory -- either nested `<city>/<bucket>.npz`
#              (osm_test*) or flat `*.npz` (data/test_bm's small/medium/large.npz)
#              (default: data/test_bm -- the paper benchmark, 100 instances/tier)
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
#     CKPT        checkpoint .ckpt file  (required for rl -- no default: silently
#                 defaulting to the P checkpoint would mislead a VARIANT=U run)
#     NUM_SAMPLE  beam width, PER ORACLE_K CANDIDATE funnel run (default: 150 --
#                 production value, see below; P3 note: total sample cost is
#                 ORACLE_K x NUM_SAMPLE, not just NUM_SAMPLE)
#     EPS         eps-lex T1 tolerance for shortlisting (default: 0.01 = 1%)
#     HYBRID      1 = two-tier funnel: eps-lex shortlist + local search on it (default: 1)
#     SHORTLIST   HYBRID=1 only: #candidates that get local search (default: 25)
#     TOPK_T1     HYBRID=1 only: extra #candidates by raw T1 rank, local-searched too,
#                 on top of the eps-lex shortlist (default: 100). Measured 82.75%->
#                 95.31% "T1 not worse than ILS" on the old osm_test_small sweep,
#                 see git history of logs/status_bm.md.
#     ORACLE_HEAD HYBRID=1 only: 1 = add heuristic class-1 forced-prefix rows
#                 (variant P only -- correctly a no-op under VARIANT=U, safe to
#                 leave on) into the pool, ORACLE_K candidates at a time,
#                 lex-best kept (default: 1). P3: class-1 candidates now
#                 ALWAYS come from the heuristic (no more exact-DP branch --
#                 see solvers/oracle.py); ORACLE_K is the compensating hedge.
#     ORACLE_K    HYBRID=1, ORACLE_HEAD=1 only: number of class-1 heuristic-seed
#                 candidates, one full funnel run each, lex-best kept (default: 3).
#                 Measured (data/test_bm, 5-instance probe): on large.npz (n1
#                 above the old exact-DP range), K=3 at a ~1.5x total-sample
#                 budget beats K=1 at the SAME total budget on T1 AND T2 AND
#                 T3 -- a single wider K=1 sample can't fix a structurally
#                 suboptimal forced prefix, only more prefixes can. On
#                 medium.npz (n1 mostly inside the old exact-DP range) K=3 is
#                 a wash (T1 unaffected, small T2 win, T3 flat-to-worse).
#                 Sub-tunables (oracle_r, oracle_shortlist) aren't exposed
#                 here -- construct RLHybHCARP directly in Python if needed.
#     UPGRADE_TO_U HYBRID=1, VARIANT=P only: 1 = also derive a zero-training
#                 variant-U answer from the P pipeline's own winning route via a
#                 move-level lex-safe local search (default: 0). Measured on
#                 data/test_bm/small.npz: fires 77/100 instances, 0 T1 regressions,
#                 T2/T3 improve -- see logs/status_bm.md.
#
# Defaults above (NUM_SAMPLE=150 SHORTLIST=25 EPS=0.01 TOPK_T1=100 ORACLE_HEAD=1
# ORACLE_K=3) are the production-recommended RL config as of the P3 refactor
# (see docs/grpo.md) -- override per-var if you need the old single-candidate
# config (NUM_SAMPLE=400 ORACLE_K=1 was the pre-P3 production default; NUM_SAMPLE=100
# EPS=0.005 SHORTLIST=10 TOPK_T1=0 ORACLE_HEAD=0 HYBRID=0 was the pre-P1/P2 default).
#
# Output:
#   Prints solver lines to stdout AND saves to logs/bm_<solver>_<variant>_<ts>.out
#
# Examples:
#   SOLVER=ils bash scripts/bm.sh                      # ILS-P on data/test_bm/small.npz
#   SOLVER=ils VARIANT=U bash scripts/bm.sh            # ILS-U, same data
#   SOLVER=ea DATA_DIR=data/2m VARIANT=U N_EPOCH=200 bash scripts/bm.sh
#   SOLVER=lp TIME_LIMIT=300 THREADS=4 bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/grpo_osm_u/epoch=192.ckpt VARIANT=U bash scripts/bm.sh
#   SOLVER=rl CKPT=outputs/checkpoints/last.ckpt NUM_SAMPLE=200 HYBRID=0 bash scripts/bm.sh   # lighter/faster ad-hoc check

set -euo pipefail
cd "$(dirname "$0")/.."

# ── required ──────────────────────────────────────────────────────────────────
SOLVER="${SOLVER:-}"
if [[ -z "$SOLVER" ]]; then
    echo "Error: SOLVER is required. Set SOLVER=ea|ils|aco|lp|rl"
    exit 1
fi

# ── common ────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/test_bm}"
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
NUM_SAMPLE="${NUM_SAMPLE:-150}" # PER ORACLE_K candidate funnel run (P3: total ~= ORACLE_K x NUM_SAMPLE)
EPS="${EPS:-0.01}"
HYBRID="${HYBRID:-1}"          # 1 = two-tier funnel (eps-lex shortlist + local search)
SHORTLIST="${SHORTLIST:-25}"   # HYBRID=1 only: #candidates that get local search
TOPK_T1="${TOPK_T1:-100}"      # HYBRID=1 only: extra #candidates by raw T1 rank, LS'd too
                                # (0 = disabled; production default is "bậc 3 + P1")
ORACLE_HEAD="${ORACLE_HEAD:-1}"                    # HYBRID=1 only: heuristic class-1 forced-prefix arm (variant P only; no-op under U)
ORACLE_K="${ORACLE_K:-3}"                          # HYBRID=1, ORACLE_HEAD=1 only: # class-1 heuristic-seed candidates (P3)
UPGRADE_TO_U="${UPGRADE_TO_U:-0}"                  # HYBRID=1, variant P only: zero-training U answer via lex-safe post-process

# ── output ────────────────────────────────────────────────────────────────────
# UPGRADE_TO_U=1 decodes with VARIANT=P but reports a variant-U answer (the
# practical "RL for variant U" result -- see logs/status_bm.md) -- tag the
# filename U so this isn't confused with a plain-P run.
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
TAG="$VARIANT"
[[ "$SOLVER" == "rl" && "$UPGRADE_TO_U" == "1" ]] && TAG="U"
# Tier tag from DATA_DIR so small/medium/large runs don't collide/get confused
# by timestamp alone -- data/test_bm/small.npz -> "small", a directory ->
# its basename (e.g. data/osm_test_small -> "osm_test_small").
SIZE_TAG="$(basename "$DATA_DIR" .npz)"
OUT="logs/bm_${SOLVER}_${TAG}_${SIZE_TAG}_${TS}.out"

VARIANT_LINE="Variant  : $VARIANT"
[[ "$TAG" == "U" && "$VARIANT" == "P" ]] && VARIANT_LINE="$VARIANT_LINE (decode) + UPGRADE_TO_U -> reported as variant-U"

echo "Solver   : $SOLVER"
echo "Data     : $DATA_DIR"
echo "$VARIANT_LINE"
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
    # DATA_DIR may be a single .npz file, a flat dir of .npz files, or a
    # nested city/bucket.npz dir.
    shopt -s nullglob
    if [[ "$DATA_DIR" == *.npz && -f "$DATA_DIR" ]]; then
        files=("$DATA_DIR")
    else
        files=("$DATA_DIR"/*/*.npz)
        [[ ${#files[@]} -eq 0 ]] && files=("$DATA_DIR"/*.npz)
    fi
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Error: no .npz files found under $DATA_DIR" >&2
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
    [[ "$ORACLE_HEAD" == "1" ]] && HYBRID_ARGS+=(--oracle_head --oracle_k "$ORACLE_K")
    [[ "$UPGRADE_TO_U" == "1" ]] && HYBRID_ARGS+=(--upgrade_to_u)
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
