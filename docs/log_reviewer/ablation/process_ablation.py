"""Compute per-config mean T1/T2/T3 gap% vs best-found-per-instance (A1g/A1R/A2/A3/A4),
plus mean time/inst, for the ablation table. A4 = main production HARMONY log.
Usage: python process_ablation.py <tier> <variant>
"""
import sys, json, re
import numpy as np

tier, variant = sys.argv[1], sys.argv[2]
BASE = "/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm"

def parse_main_log(path):
    T, times = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'\[\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\]\]\s*:::\s*([\d.eE+-]+)', line)
            if m:
                T.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                times.append(float(m.group(4)))
    return np.array(T), np.array(times)

abl = json.load(open(f"{BASE}/ablation_{tier}_{variant}.json"))
configs = {}
for key in ["A1_greedy", "A1_bestofR", "A2_no_candidates"]:
    if abl["results"][key]:
        configs[key] = (np.array(abl["results"][key]), np.array(abl["times"][key]))
if variant == "P" and abl["results"].get("A3_candidates_no_ls"):
    configs["A3_candidates_no_ls"] = (np.array(abl["results"]["A3_candidates_no_ls"]), np.array(abl["times"]["A3_candidates_no_ls"]))

# find the main production log for A4 (glob for the right tier/variant)
import glob
if variant == "P":
    pat = f"/home/quangnm/projects/ArcRoute/logs/bm_rl_P_{tier}_*.out"
else:
    pat = f"/home/quangnm/projects/ArcRoute/logs/bm_rl_U_{tier}_*.out"
files = sorted(glob.glob(pat))
if files:
    T4, t4 = parse_main_log(files[-1])
    configs["A4_full"] = (T4, t4)

n = min(len(v[0]) for v in configs.values())
stacked = np.stack([v[0][:n] for v in configs.values()], axis=0)  # (n_configs, n, 3)
best = stacked.min(axis=0)  # (n, 3) best-per-instance-per-class

print(f"=== {tier} {variant} (n={n} instances) ===")
for name, (T, times) in configs.items():
    T = T[:n]
    gap = (T - best) / np.maximum(best, 1e-9) * 100
    print(f"{name:20s} T1gap={gap[:,0].mean():6.2f}% T2gap={gap[:,1].mean():6.2f}% T3gap={gap[:,2].mean():6.2f}% time={times[:n].mean():.2f}s")
