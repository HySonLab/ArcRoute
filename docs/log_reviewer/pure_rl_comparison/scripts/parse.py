"""Parse bm.sh .out logs -> (N,3) T arrays + times. Compare prod vs pure-RL."""
import re, sys, os
import numpy as np
sys.path.append('/home/quangnm/projects/ArcRoute/src')
from eval.stats import pairwise_wilcoxon

LINE = re.compile(r'^.*? ::: \[\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]\] ::: ([-\d.eE+]+)\s*$')

def parse(path):
    Ts, times = [], []
    with open(path) as fh:
        for ln in fh:
            m = LINE.match(ln.rstrip('\n'))
            if m:
                Ts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                times.append(float(m.group(4)))
    return np.array(Ts), np.array(times)

def compare(prod_path, pure_path, tier, budget):
    P, tp = parse(prod_path)
    R, tr = parse(pure_path)
    n = min(len(P), len(R))
    assert len(P) == len(R) == n, f"len mismatch {len(P)} vs {len(R)}"
    print(f"\n===== {tier}  (pure-RL NUM_SAMPLE={budget}) =====")
    print(f"n={n}   prod mean time={tp.mean():.3f}s   pureRL mean time={tr.mean():.3f}s")
    print(f"{'cls':>3} | {'prod mean':>10} | {'pureRL mean':>11} | {'gap%(R vs P)':>12} | "
          f"{'pureRL-wins':>11} | {'prod-wins':>9} | {'ties':>5} | {'wilcoxon p':>11}")
    for k, name in enumerate(['T1','T2','T3']):
        a, b = P[:, k], R[:, k]  # a=prod, b=pureRL, lower better
        # gap of pureRL relative to prod
        gap = ((b - a) / a * 100.0)
        meangap = gap.mean()
        pure_wins = int((b < a - 1e-9).sum())
        prod_wins = int((a < b - 1e-9).sum())
        ties = n - pure_wins - prod_wins
        stat, p = pairwise_wilcoxon(a, b)
        print(f"{name:>3} | {a.mean():>10.4f} | {b.mean():>11.4f} | {meangap:>+11.2f}% | "
              f"{pure_wins:>11} | {prod_wins:>9} | {ties:>5} | {p:>11.3e}")
    return P, R, tp, tr

if __name__ == '__main__':
    # usage: parse.py prod.out pure.out tier budget
    compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
