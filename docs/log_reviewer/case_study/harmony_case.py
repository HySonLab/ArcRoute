"""Run the FULL HARMONY hybrid pipeline (RLHybHCARP, production config:
num_sample=150, shortlist=25, eps=0.01, topk_t1=100, oracle_head, oracle_k=3)
on ONE instance and save the winning routes to a .sol file.

Needed because `python -m solvers.rl --file` runs the RAW policy only
(sampling + eps-lex pick, no funnel); the --hybrid flags apply only in batch
--path mode, which does not save routes. This script replicates
RLHybHCARP.__call__'s body verbatim but keeps `best_vehicles` (which
__call__ discards), then converts vehicles -> routes exactly the way
rl.py's file mode does, and writes via utils.ops.save_sol.

Usage (from /home/quangnm/projects/ArcRoute):
    uv run python <this file> <instance.npz> <ckpt> <out.sol> [num_sample]
"""
import sys, time
sys.path.insert(0, 'src')
import numpy as np
import torch

from solvers.rl import RLHybHCARP
from utils.ops import iter_instances, save_sol

fpath, ckpt, out_sol = sys.argv[1], sys.argv[2], sys.argv[3]
num_sample = int(sys.argv[4]) if len(sys.argv) > 4 else 150
SEED = 6868
torch.manual_seed(SEED)
np.random.seed(SEED)

al = RLHybHCARP(ckpt, 'P', shortlist=25, eps=0.01, topk_t1=100,
                oracle_head=True, oracle_k=3)
data = next(iter_instances(fpath, M=None))
al._load_data(*data)

t0 = time.time()
# ---- body of RLHybHCARP.__call__, keeping best_vehicles -------------------
td0 = al.td[0]
M = int(td0["num_vehicle"])
N1 = td0["service_times"].shape[0]
adj = td0["adj"].cpu().numpy().astype(np.float64)
clss = td0["clss"].cpu().numpy().astype(np.int64)
demands = td0["demand"].cpu().numpy().astype(np.float64)
classes = [c for c in np.unique(clss) if c != 0]

forced_list = al._oracle_forced_prefixes(td0, N1, adj, clss, demands, M)
results = [al._run_funnel_once(td0, M, N1, adj, clss, demands,
                               classes, forced, num_sample)
           for forced in forced_list]
best_T, best_vehicles = min(results, key=lambda r: tuple(r[0].tolist()))
elapsed = time.time() - t0

# ---- vehicles -> routes, exactly as rl.py file mode does ------------------
best_routes = []
for trips in best_vehicles:
    route = [0]
    for trip in trips:
        route.extend(int(a) for a in trip)
        route.append(0)
    if len(route) > 2:
        best_routes.append(np.array(route, dtype=np.int32))

print(f"HARMONY  T=({best_T[0]:.4f}, {best_T[1]:.4f}, {best_T[2]:.4f})  "
      f"K_prefixes={len(forced_list)}  time={elapsed:.3f}s")
save_sol(out_sol, best_routes, best_T,
         instance=fpath, checkpoint=ckpt, variant='P', vehicles=M,
         solver='harmony_hybrid', num_sample=num_sample, time_s=f"{elapsed:.3f}")
print(f"Saved -> {out_sol}")
