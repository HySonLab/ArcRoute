#!/usr/bin/env bash
# Full-100 pure-RL (HYBRID=0 ORACLE_HEAD=0) sweep, all tiers.
set -uo pipefail
cd /home/quangnm/projects/ArcRoute
SD="/tmp/claude-1002/-home-quangnm-projects-HDCARP-with-ML/52d2d33a-2951-49c4-9aca-a044747e7ac2/scratchpad"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A EP=( [small]=044 [medium]=037 [large]=041 )
GRID_small="8000 16000 32000 64000"
GRID_medium="16000 32000 40000"
GRID_large="8000 12000 16000"
for tier in small medium large; do
  var="GRID_${tier}"; grid="${!var}"
  for ns in $grid; do
    out="$SD/pure_${tier}_${ns}.out"
    echo "SWEEP $tier ns=$ns START $(date +%T)"
    SOLVER=rl CKPT=outputs/checkpoints/grpo_reload_fix/$tier/epoch=${EP[$tier]}.ckpt \
    HYBRID=0 ORACLE_HEAD=0 NUM_SAMPLE=$ns EPS=0.01 \
    DATA_DIR=data/test_bm/$tier.npz VARIANT=P bash scripts/bm.sh > "$out" 2>&1 || true
    if grep -qiE "out of memory|CUDA error|RuntimeError|Traceback" "$out"; then
      echo "SWEEP $tier ns=$ns ERROR/OOM (n=$(grep -c ':::' "$out"))"
    else
      mt=$(grep ':::' "$out"|awk -F':::' '{print $3}'|awk '{s+=$1;n++}END{if(n)printf "%.4f",s/n;else print "NA"}')
      echo "SWEEP $tier ns=$ns DONE mean_time=${mt}s n=$(grep -c ':::' "$out") $(date +%T)"
    fi
  done
done
echo "SWEEP ALL DONE $(date +%T)"
