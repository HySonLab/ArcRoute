"""Full comparison: production hybrid vs pure-RL sweep, per tier/class."""
import sys, os
import numpy as np
sys.path.insert(0, '/home/quangnm/projects/ArcRoute/src')
sys.path.insert(0, os.path.dirname(__file__))
from parse import parse
from eval.stats import pairwise_wilcoxon

SD = os.path.join(os.path.dirname(__file__), '..', 'full_runs')
PROD_TIME = {'small': None, 'medium': None, 'large': None}
GRID = {
    'small':  [8000, 16000, 32000, 64000],
    'medium': [16000, 32000, 40000],
    'large':  [8000, 12000, 16000],
}

def load_prod(tier):
    return parse(f'{SD}/prod_{tier}.out')

def load_pure(tier, ns):
    p = f'{SD}/pure_{tier}_{ns}.out'
    if not os.path.exists(p):
        return None, None
    T, t = parse(p)
    if len(T) < 100:
        return None, None
    return T, t

def main():
    for tier in ['small', 'medium', 'large']:
        P, tp = load_prod(tier)
        prod_time = tp.mean()
        print('='*100)
        print(f'TIER {tier.upper()}   production hybrid: mean_time={prod_time:.3f}s   '
              f'T1/T2/T3={P[:,0].mean():.4f}/{P[:,1].mean():.4f}/{P[:,2].mean():.4f}')
        print('-'*100)
        # per-config pure-RL summary
        rows = []
        for ns in GRID[tier]:
            R, tr = load_pure(tier, ns)
            if R is None:
                print(f'  ns={ns:>6}: MISSING/incomplete'); continue
            rows.append((ns, R, tr))
            print(f'  ns={ns:>6}: mean_time={tr.mean():.3f}s  '
                  f'T1/T2/T3={R[:,0].mean():.4f}/{R[:,1].mean():.4f}/{R[:,2].mean():.4f}')
        if not rows:
            continue
        # time-matched point: smallest ns whose mean time >= prod_time; else the max ns (memory-capped)
        matched = None
        for ns, R, tr in rows:
            if tr.mean() >= prod_time:
                matched = (ns, R, tr); break
        capped = matched is None
        if matched is None:
            matched = rows[-1]  # max feasible budget
        ns, R, tr = matched
        tag = 'MAX-FEASIBLE (memory-capped, UNDER time budget)' if capped else 'time-matched (>= prod time)'
        print('-'*100)
        print(f'  >>> comparison point: ns={ns} ({tag}), pure-RL mean_time={tr.mean():.3f}s '
              f'vs prod {prod_time:.3f}s')
        print(f'  {"cls":>3} | {"prod":>9} | {"pureRL":>9} | {"gap%":>8} | {"pRL wins":>8} | '
              f'{"prod wins":>9} | {"ties":>4} | {"wilcoxon p":>11} | {"best-any-budget":>15}')
        for k, name in enumerate(['T1', 'T2', 'T3']):
            a, b = P[:, k], R[:, k]
            gap = ((b.mean() - a.mean()) / a.mean() * 100.0)
            pw = int((b < a - 1e-9).sum()); pl = int((a < b - 1e-9).sum()); pt = 100 - pw - pl
            _, p = pairwise_wilcoxon(a, b)
            # best pure-RL mean for this class across ALL budgets
            best_any = min(RR[:, k].mean() for _, RR, _ in rows)
            print(f'  {name:>3} | {a.mean():>9.4f} | {b.mean():>9.4f} | {gap:>+7.2f}% | '
                  f'{pw:>8} | {pl:>9} | {pt:>4} | {p:>11.3e} | {best_any:>15.4f}')
        print()

if __name__ == '__main__':
    main()
