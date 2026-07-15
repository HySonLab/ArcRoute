#!/usr/bin/env bash
set -euo pipefail
cd /home/quangnm/projects/ArcRoute
SD="/tmp/claude-1002/-home-quangnm-projects-HDCARP-with-ML/52d2d33a-2951-49c4-9aca-a044747e7ac2/scratchpad"
declare -A EP=( [small]=044 [medium]=037 [large]=041 )
for tier in small medium large; do
  echo "=== PROD $tier START $(date +%T) ==="
  SOLVER=rl CKPT=outputs/checkpoints/grpo_reload_fix/$tier/epoch=${EP[$tier]}.ckpt \
  HYBRID=1 ORACLE_HEAD=1 ORACLE_K=3 NUM_SAMPLE=150 SHORTLIST=25 TOPK_T1=100 EPS=0.01 \
  DATA_DIR=data/test_bm/$tier.npz VARIANT=P bash scripts/bm.sh \
    > "$SD/prod_${tier}.out" 2>&1
  echo "=== PROD $tier DONE $(date +%T) ==="
done
echo "ALL PROD DONE $(date +%T)"
