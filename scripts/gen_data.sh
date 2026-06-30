#!/usr/bin/env bash
# Generate OSM HDCARP instances for one city or all cities in parallel.
#
# Usage:
#   bash scripts/gen_data.sh --city <city>  --mode <small|medium|large|all> [OPTIONS]
#   bash scripts/gen_data.sh --all          --mode <small|medium|large|all> [OPTIONS]
#
# Options:
#   --city       Single city (key in CITIES, e.g. "paris")
#   --all        Run all 28 cities (use with --workers)
#   --mode       small | medium | large | all  (default: all)
#   --workers    N parallel city jobs for --all mode (default: 4)
#   --per-bucket Override instance count per bucket (default: production counts)
#   --out-dir    Output root dir (default: data/osm_train)
#   --seed       Base RNG seed (default: 42)
#
# Examples:
#   bash scripts/gen_data.sh --city new_york --mode small
#   bash scripts/gen_data.sh --city new_york --mode all
#   bash scripts/gen_data.sh --all --mode all --workers 4
#   bash scripts/gen_data.sh --all --mode small --per-bucket 10  # test run
#
# Output: data/osm_train/<city>/<bucket>.npz
# Log:    logs/gen_<city>_<mode>_<timestamp>.out

set -euo pipefail

CITY=""
RUN_ALL=0
MODE="all"
WORKERS=4
PER_BUCKET_OVERRIDE=0
OUT_DIR="data/osm_train"
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --city)        CITY="$2";                shift 2 ;;
        --all)         RUN_ALL=1;                shift 1 ;;
        --mode)        MODE="$2";                shift 2 ;;
        --workers)     WORKERS="$2";             shift 2 ;;
        --per-bucket)  PER_BUCKET_OVERRIDE="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2";             shift 2 ;;
        --seed)        SEED="$2";                shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CITY" && $RUN_ALL -eq 0 ]]; then
    echo "Usage: $0 --city <city> --mode <MODE> [OPTIONS]"
    echo "       $0 --all         --mode <MODE> --workers N [OPTIONS]"
    exit 1
fi

# ------------------------------------------------------------------ #
# --all: dispatch one gen_data.sh per city via xargs
# ------------------------------------------------------------------ #
if [[ $RUN_ALL -eq 1 ]]; then
    ALL_CITIES="new_york chicago boston toronto mexico_city san_francisco \
                buenos_aires sao_paulo bogota \
                london paris rome barcelona amsterdam berlin prague \
                cairo lagos marrakesh nairobi \
                tokyo hanoi mumbai singapore shanghai chandigarh \
                sydney melbourne"

    PB_FLAG=""
    [[ $PER_BUCKET_OVERRIDE -gt 0 ]] && PB_FLAG="--per-bucket ${PER_BUCKET_OVERRIDE}"

    echo "[gen_data] --all mode=${MODE} workers=${WORKERS}"
    echo "$ALL_CITIES" | tr ' ' '\n' | \
        xargs -P "$WORKERS" -I{} \
        bash scripts/gen_data.sh --city {} --mode "$MODE" \
            --out-dir "$OUT_DIR" --seed "$SEED" $PB_FLAG
    echo "[gen_data] all cities done"
    exit 0
fi

# ------------------------------------------------------------------ #
# Single city
# ------------------------------------------------------------------ #
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/gen_${CITY}_${MODE}_${TIMESTAMP}.out"

echo "[gen_data] city=${CITY} mode=${MODE} out_dir=${OUT_DIR} seed=${SEED}" | tee -a "$LOG"
[[ $PER_BUCKET_OVERRIDE -gt 0 ]] && echo "[gen_data] per-bucket override: ${PER_BUCKET_OVERRIDE}" | tee -a "$LOG"

uv run python - "$CITY" "$MODE" "$PER_BUCKET_OVERRIDE" "$OUT_DIR" "$SEED" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, time
from utils.data_utils import gen_city_bucket, CURRICULUM, eligible_buckets

city, mode, n_override, out_dir, seed = \
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])

PER_BUCKET = {
    "20_30": 643,  "20_40": 786,
    "30_45": 571,  "30_60": 571,  "40_60": 571,
    "50_75": 500,  "40_80": 571,
    "50_100": 571, "80_120": 500, "80_133": 500,
}

all_buckets  = sum(CURRICULUM.values(), []) if mode == "all" else CURRICULUM[mode]
city_ok      = eligible_buckets(city)
buckets      = [b for b in all_buckets if b in city_ok]
skipped_tier = [b for b in all_buckets if b not in city_ok]
if skipped_tier:
    print(f"[gen_data] tier-skip {city}: {skipped_tier}")
print(f"[gen_data] mode={mode} buckets={buckets}")

t0 = time.time()
skipped = []
for bucket in buckets:
    n = n_override if n_override > 0 else PER_BUCKET[bucket]
    bucket_seed = seed ^ (hash(city + bucket) & 0xFFFFFFFF)
    try:
        gen_city_bucket(city, bucket, n, seed=bucket_seed, out_dir=out_dir)
    except RuntimeError as e:
        print(f"[gen_data] SKIP {city}/{bucket}: {e}")
        skipped.append(bucket)

if skipped:
    print(f"[gen_data] skipped buckets: {skipped}")
print(f"[gen_data] done — total {time.time()-t0:.1f}s")
PYEOF

echo "[gen_data] log saved to $LOG"
