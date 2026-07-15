"""Learning-free-hybrid ablation arm (A5): HARMONY's pool funnel with GCH
rollouts instead of the frozen policy. Extends ablation.py's A1-A4 with a
fifth arm on the SAME data/test_bm/{tier}.npz instances, SAME R=150, SAME
funnel invocation -- only the rollout source changes (no policy, no GPU).

Config mirrors full HARMONY (A4) exactly: shortlist=25, eps=0.01, topk_t1=100,
oracle_head=True (K=3 class-1 candidates), R=150. HDCARP-U uses the same
P-anchored upgrade_to_u pipeline A4-U uses.

Usage: python ablation_a5.py <tier> <variant>
Saves: ablation_learning_free_{tier}_{variant}.json  (raw per-instance T + time)
"""
import sys, os, time, json
sys.path.insert(0, 'src')
import numpy as np

from solvers.rl_gch import GCHHybHCARP
from utils.ops import iter_instances

tier, variant = sys.argv[1], sys.argv[2]
DATA = f"data/test_bm/{tier}.npz"
R = 150  # matches production num_sample / ablation.py

# Mirror A4: HDCARP-U is answered via the P-anchored upgrade_to_u pipeline
# (see docs/status_bm.md CURRENT config), so construct with variant='P' and
# report the upgraded, U-scored answer.
upg = (variant == 'U')
al = GCHHybHCARP('P', shortlist=25, eps=0.01, topk_t1=100,
                 oracle_head=True, oracle_k=3, upgrade_to_u=upg)

Ts, times = [], []
for i, data in enumerate(iter_instances(DATA, M=None)):
    al._load_data(*data)
    t0 = time.time()
    T = al(num_sample=R)[0].tolist()
    dt = time.time() - t0
    Ts.append(T)
    times.append(dt)
    print(f"A5 {tier} {variant} [{i}]: {[round(x,3) for x in T]} ({dt:.2f}s)", flush=True)

out = {"tier": tier, "variant": variant, "config": "learning_free_hybrid_gch",
       "results": {"A5_learning_free": Ts}, "times": {"A5_learning_free": times}}
here = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(here, f"ablation_learning_free_{tier}_{variant}.json")
with open(outfile, "w") as f:
    json.dump(out, f)
print(f"Saved -> {outfile}  (n={len(Ts)})")
