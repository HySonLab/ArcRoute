"""Compute the learning-free-hybrid (A5) gaps and check whether adding A5 to
the reference perturbs the published A1-A4 gaps. Reuses the exact A1-A4 raw
arrays (ablation_{tier}_{variant}.json), the A4 production log
(logs/bm_rl_{v}_{tier}.out), and the new A5 raw array
(ablation_learning_free_{tier}_{variant}.json).

Gap = (T - best) / max(best, 1e-9) * 100, best = per-instance-per-class min
over the compared configs -- identical formula to process_ablation.py.

Usage: python process_ablation_a5.py <tier> <variant>
"""
import sys, os, json, re, glob
import numpy as np

tier, variant = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def parse_main_log(path):
    T, times = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'\[\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\]\]\s*:::\s*([\d.eE+-]+)', line)
            if m:
                T.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                times.append(float(m.group(4)))
    return np.array(T), np.array(times)


abl = json.load(open(f"{HERE}/ablation_{tier}_{variant}.json"))
configs = {}
for key in ["A1_greedy", "A1_bestofR", "A2_no_candidates"]:
    if abl["results"].get(key):
        configs[key] = (np.array(abl["results"][key]), np.array(abl["times"][key]))
if variant == "P" and abl["results"].get("A3_candidates_no_ls"):
    configs["A3_candidates_no_ls"] = (np.array(abl["results"]["A3_candidates_no_ls"]),
                                      np.array(abl["times"]["A3_candidates_no_ls"]))

# A4 (full HARMONY) from the canonical production log
a4log = f"{REPO}/logs/bm_rl_{variant}_{tier}.out"
if not os.path.exists(a4log):
    hits = sorted(glob.glob(f"{REPO}/logs/bm_rl_{variant}_{tier}*.out"))
    a4log = hits[-1] if hits else None
T4, t4 = parse_main_log(a4log)
configs["A4_full"] = (T4, t4)

# A5 (learning-free hybrid)
a5 = json.load(open(f"{HERE}/ablation_learning_free_{tier}_{variant}.json"))
T5 = np.array(a5["results"]["A5_learning_free"])
t5 = np.array(a5["times"]["A5_learning_free"])

order14 = list(configs.keys())            # A1..A4
n = min([len(v[0]) for v in configs.values()] + [len(T5)])

stack14 = np.stack([configs[k][0][:n] for k in order14], axis=0)
best14 = stack14.min(axis=0)
stack15 = np.concatenate([stack14, T5[:n][None]], axis=0)
best15 = stack15.min(axis=0)


def gaps(T, best):
    T = T[:n]
    g = (T - best) / np.maximum(best, 1e-9) * 100
    return g[:, 0].mean(), g[:, 1].mean(), g[:, 2].mean()


print(f"=== {tier} {variant} (n={n}) ===")
print("-- gaps vs A1-A4 reference (published) --")
for k in order14:
    g = gaps(configs[k][0], best14)
    print(f"  {k:20s} T1={g[0]:6.2f}% T2={g[1]:6.2f}% T3={g[2]:6.2f}% time={configs[k][1][:n].mean():.2f}s")
g5_14 = gaps(T5, best14)
print(f"  {'A5_learning_free':20s} T1={g5_14[0]:6.2f}% T2={g5_14[1]:6.2f}% T3={g5_14[2]:6.2f}% time={t5[:n].mean():.2f}s")

print("-- gaps vs A1-A5 reference (A5 included in best-found) --")
drift = False
for k in order14:
    g = gaps(configs[k][0], best15)
    g_old = gaps(configs[k][0], best14)
    d = max(abs(g[0]-g_old[0]), abs(g[1]-g_old[1]), abs(g[2]-g_old[2]))
    if d > 0.005:
        drift = True
    print(f"  {k:20s} T1={g[0]:6.2f}% T2={g[1]:6.2f}% T3={g[2]:6.2f}% (max shift {d:.3f}pp)")
g5_15 = gaps(T5, best15)
print(f"  {'A5_learning_free':20s} T1={g5_15[0]:6.2f}% T2={g5_15[1]:6.2f}% T3={g5_15[2]:6.2f}% time={t5[:n].mean():.2f}s")
print(f"A1-A4 drift when A5 added to reference: {'YES' if drift else 'NO'}")
