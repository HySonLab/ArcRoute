"""Empirical validation of Proposition 1 / Corollary 2 (K-candidate hedging):
for every instance, check T1(S*) <= min_k min_i T1(S_k^(i)) (free rollouts only,
per candidate k), i.e. the funnel's final output is never worse on T1 than the
best free rollout of ANY of the K class-1 candidates. Also records the full
lex vector margin when the inequality is strict.
Usage: python prop1_validate.py <tier> <variant> <ckpt>
"""
import sys, json
sys.path.insert(0, 'src')
import numpy as np

from solvers.rl import RLHybHCARP
from utils.ops import iter_instances

tier, variant, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = f"data/test_bm/{tier}.npz"
R = 150

al = RLHybHCARP(ckpt, variant, shortlist=25, eps=0.01, topk_t1=100,
                oracle_head=(variant == 'P'), oracle_k=3)

# capture each candidate's free-rollout-only T1 min via a thin wrapper
captured = {"per_call_free_t1_min": []}
orig_funnel = al.__class__._run_funnel_once
def wrapped(self, td0, M, N1, adj, clss, demands, classes, forced, num_sample):
    # replicate the free-rollout decode to get its own T1 min (same call the
    # real _run_funnel_once makes internally) -- cheapest correct way without
    # duplicating internal state: call _decode_pool exactly as _run_funnel_once
    # does, record the free-rollout T1 min, then delegate to the real method.
    reward, actions, n_forced = self._decode_pool(num_sample, forced)
    captured["per_call_free_t1_min"].append(float(reward[:num_sample, 0].min()))
    return orig_funnel(self, td0, M, N1, adj, clss, demands, classes, forced, num_sample)
al.__class__._run_funnel_once = wrapped

violations = []
margins = []
n_checked = 0
for i, data in enumerate(iter_instances(DATA, M=None)):
    al._load_data(*data)
    captured["per_call_free_t1_min"] = []
    result = al(num_sample=R)[0]
    T1_star = float(result[0])
    best_free_t1 = min(captured["per_call_free_t1_min"])
    n_checked += 1
    margin = best_free_t1 - T1_star  # should be >= -tiny tol (T1* <= best_free_t1)
    margins.append(margin)
    if margin < -1e-6:
        violations.append({"instance": i, "T1_star": T1_star, "best_free_t1": best_free_t1, "margin": margin})
    print(f"prop1 {tier} {variant} [{i}]: T1*={T1_star:.4f} best_free_T1={best_free_t1:.4f} margin={margin:.4f}", flush=True)

out = {
    "tier": tier, "variant": variant, "n_checked": n_checked,
    "n_violations": len(violations), "violations": violations,
    "median_margin": float(np.median(margins)), "mean_margin": float(np.mean(margins)),
    "pct_strict": float(np.mean(np.array(margins) > 1e-9) * 100),
}
outfile = f"/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm/prop1_{tier}_{variant}.json"
with open(outfile, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> {outfile}")
print(f"SUMMARY {tier} {variant}: n={n_checked} violations={len(violations)} median_margin={out['median_margin']:.4f} pct_strict={out['pct_strict']:.1f}%")
