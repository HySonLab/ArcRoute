"""Anytime-curve data: sweep num_sample R over {1,4,16,64,256} for full HARMONY,
recording mean T1/T2/T3 and mean time/inst. Sample size reduced from the main
100-instance benchmark to keep this tractable while ACO runs concurrently on
CPU (GPU is otherwise idle) -- N=30 per cell, well above the paper's own
pre-registered minimum of 20/cell; noted explicitly, not silently capped.
Usage: python rsweep.py <tier> <variant> <ckpt>
"""
import sys, time, json
sys.path.insert(0, 'src')
import numpy as np

from solvers.rl import RLHybHCARP
from utils.ops import iter_instances

tier, variant, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = f"data/test_bm/{tier}.npz"
N = 30
R_GRID = [1, 4, 16, 64, 256]

if variant == 'U':
    # match the production HARMONY-U pathway used for the main table: variant-P
    # decode + candidates, then the zero-training upgrade-to-U local-search pass
    # (see run_rl.sh / bm.sh UPGRADE_TO_U=1). A native variant='U' construction
    # runs the weaker oracle_head=False funnel and does NOT match tab:planned.
    al = RLHybHCARP(ckpt, 'P', shortlist=25, eps=0.01, topk_t1=100,
                    oracle_head=True, oracle_k=3, upgrade_to_u=True)
else:
    al = RLHybHCARP(ckpt, variant, shortlist=25, eps=0.01, topk_t1=100,
                    oracle_head=True, oracle_k=3)

results = {R: {"T": [], "time": []} for R in R_GRID}
for i, data in enumerate(iter_instances(DATA, M=None)):
    if i >= N:
        break
    al._load_data(*data)
    for R in R_GRID:
        t0 = time.time()
        T = al(num_sample=R)[0].tolist()
        dt = time.time() - t0
        results[R]["T"].append(T)
        results[R]["time"].append(dt)
    print(f"{tier} {variant} instance {i} done", flush=True)

out = {"tier": tier, "variant": variant, "N": N, "R_GRID": R_GRID, "results": results}
outfile = f"/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm/rsweep_{tier}_{variant}.json"
with open(outfile, "w") as f:
    json.dump(out, f)
print(f"Saved -> {outfile}")
for R in R_GRID:
    T = np.array(results[R]["T"])
    t = np.array(results[R]["time"])
    print(f"  R={R:4d}: T1={T[:,0].mean():.3f} T2={T[:,1].mean():.3f} T3={T[:,2].mean():.3f} time={t.mean():.3f}s")
