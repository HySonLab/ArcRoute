#!/usr/bin/env bash
set -euo pipefail
cd /home/quangnm/projects/ArcRoute
SD="/tmp/claude-1002/-home-quangnm-projects-HDCARP-with-ML/52d2d33a-2951-49c4-9aca-a044747e7ac2/scratchpad"
export PYTHONUNBUFFERED=1
echo "=== PROD large START $(date +%T) ==="
SOLVER=rl CKPT=outputs/checkpoints/grpo_reload_fix/large/epoch=041.ckpt \
HYBRID=1 ORACLE_HEAD=1 ORACLE_K=3 NUM_SAMPLE=150 SHORTLIST=25 TOPK_T1=100 EPS=0.01 \
DATA_DIR=data/test_bm/large.npz VARIANT=P bash scripts/bm.sh \
  > "$SD/prod_large.out" 2>&1
echo "=== PROD large DONE $(date +%T) ==="
