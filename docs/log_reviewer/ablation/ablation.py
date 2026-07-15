"""Inference-arm ablation: A1-greedy, A1-best-of-R, A2 (no candidates), A3 (candidates, no LS).
A4 (full HARMONY) is NOT run here -- reuse the main production bm.sh RL logs for that row.
Usage: python ablation.py <tier> <variant> <ckpt>
"""
import sys, time, json
sys.path.insert(0, 'src')
import numpy as np
import torch

from solvers.rl import RLHCARP, RLHybHCARP, eps_lex_select
from utils.ops import batchify, iter_instances

tier, variant, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = f"data/test_bm/{tier}.npz"
R = 150  # matches production num_sample

results = {"A1_greedy": [], "A1_bestofR": [], "A2_no_candidates": [], "A3_candidates_no_ls": []}
times = {k: [] for k in results}

# --- A1: plain policy, greedy and best-of-R (shared model load) ---
al1 = RLHCARP(ckpt, variant)
for i, data in enumerate(iter_instances(DATA, M=None)):
    al1._load_data(*data)
    # greedy
    t0 = time.time()
    with torch.inference_mode():
        g_out = al1.policy(batchify(al1.td, 1), env=al1.env, decode_type='greedy')
    dt = time.time() - t0
    T = g_out["reward"].cpu().numpy()[0].tolist()
    results["A1_greedy"].append(T)
    times["A1_greedy"].append(dt)
    # best-of-R
    t0 = time.time()
    T2 = al1(num_sample=R, eps=0.01)[0].tolist()
    dt = time.time() - t0
    results["A1_bestofR"].append(T2)
    times["A1_bestofR"].append(dt)
    print(f"A1 {tier} {variant} [{i}]: greedy={T} bestofR={T2}", flush=True)
del al1

# --- A2: pool funnel, oracle_head=False (P and U both meaningful here for A2 timing,
#         though paper notes A2==A4 for U by construction -- we still record it) ---
al2 = RLHybHCARP(ckpt, variant, shortlist=25, eps=0.01, topk_t1=100, oracle_head=False)
for i, data in enumerate(iter_instances(DATA, M=None)):
    al2._load_data(*data)
    t0 = time.time()
    T = al2(num_sample=R)[0].tolist()
    dt = time.time() - t0
    results["A2_no_candidates"].append(T)
    times["A2_no_candidates"].append(dt)
    print(f"A2 {tier} {variant} [{i}]: {T}", flush=True)
del al2

# --- A3: candidates forced, but local search disabled (P only -- oracle_head requires P) ---
if variant == 'P':
    al3 = RLHybHCARP(ckpt, variant, shortlist=25, eps=0.01, topk_t1=100, oracle_head=True, oracle_k=3)
    al3._local_search_one = lambda *a, **kw: (None, None)  # disable LS: always keep pre-LS candidate
    for i, data in enumerate(iter_instances(DATA, M=None)):
        al3._load_data(*data)
        t0 = time.time()
        T = al3(num_sample=R)[0].tolist()
        dt = time.time() - t0
        results["A3_candidates_no_ls"].append(T)
        times["A3_candidates_no_ls"].append(dt)
        print(f"A3 {tier} {variant} [{i}]: {T}", flush=True)
    del al3

out = {"tier": tier, "variant": variant, "ckpt": ckpt, "results": results, "times": times}
outfile = f"/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm/ablation_{tier}_{variant}.json"
with open(outfile, "w") as f:
    json.dump(out, f)
print(f"Saved -> {outfile}")
