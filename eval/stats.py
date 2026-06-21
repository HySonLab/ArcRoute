"""Statistical-rigor helpers for HDCARP experiments (plan Phase 5).

Provides the comparisons a Q1/A* paper needs:
  - gap-to-BKS (best-known solution), best & mean
  - Wilcoxon signed-rank (paired, two algorithms)
  - Friedman test (>=3 algorithms) + average ranks (CD-diagram inputs)
  - tightness tau = sum(q) / (M*Q) distribution per cell (Smith-Miles 2023)

All functions are pure and unit-tested (tests/test_stats.py). Run a quick
self-check on synthetic data:
    uv run python eval/stats.py --selftest
"""
import argparse
import glob
import os

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare


# --------------------------------------------------------------------------- #
# Gap to best-known solution
# --------------------------------------------------------------------------- #
def gap_to_bks(obj, bks):
    """Relative gap (obj - bks) / bks. 0 when obj == bks; >0 = worse."""
    obj = np.asarray(obj, dtype=float)
    bks = np.asarray(bks, dtype=float)
    return (obj - bks) / bks


def gap_summary(obj, bks):
    """Best & mean gap-to-BKS over a set of instances (as percentages)."""
    g = gap_to_bks(obj, bks) * 100.0
    return {"best_gap_pct": float(g.min()), "mean_gap_pct": float(g.mean())}


# --------------------------------------------------------------------------- #
# Significance tests
# --------------------------------------------------------------------------- #
def pairwise_wilcoxon(a, b):
    """Wilcoxon signed-rank on paired results of two algorithms. Returns
    (statistic, p_value). Identical inputs -> p=1.0 (no difference)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return 0.0, 1.0
    stat, p = wilcoxon(a, b)
    return float(stat), float(p)


def friedman(*algos):
    """Friedman test across >=3 algorithms (each a per-instance result vector).
    Returns (statistic, p_value)."""
    if len(algos) < 3:
        raise ValueError("Friedman test needs >= 3 algorithms")
    stat, p = friedmanchisquare(*algos)
    return float(stat), float(p)


def average_ranks(matrix):
    """Average rank of each algorithm (columns) over instances (rows); rank 1 =
    best (smallest, since HDCARP minimises makespan). These feed a CD diagram."""
    m = np.asarray(matrix, dtype=float)
    ranks = np.apply_along_axis(_rankdata, 1, m)
    return ranks.mean(axis=0)


def _rankdata(row):
    """Average ranks (1=smallest), ties share the mean rank."""
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty(len(row), dtype=float)
    ranks[order] = np.arange(1, len(row) + 1)
    # average tied ranks
    _, inv, counts = np.unique(row, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


# --------------------------------------------------------------------------- #
# Tightness reporting
# --------------------------------------------------------------------------- #
def tightness_from_dir(path):
    """Collect the tightness tau metadata from every .npz under `path`."""
    taus = []
    for f in sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True)):
        meta = np.load(f)
        if "tau" in meta.files:
            taus.append(float(meta["tau"]))
    return np.asarray(taus)


def describe(values):
    """mean +/- std (and min/max) of a 1-D array, for per-cell reporting."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


# --------------------------------------------------------------------------- #
# Self-check on synthetic data
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.RandomState(0)
    bks = rng.rand(30) * 10 + 1
    good = bks * 1.02            # 2% worse
    bad = bks * 1.10             # 10% worse

    assert abs(gap_summary(bks, bks)["mean_gap_pct"]) < 1e-9      # obj==BKS -> 0
    assert abs(gap_summary(good, bks)["mean_gap_pct"] - 2.0) < 1e-6

    _, p_same = pairwise_wilcoxon(good, good)
    _, p_diff = pairwise_wilcoxon(good, bad)
    assert p_same == 1.0 and 0.0 <= p_diff <= 1.0

    _, pf = friedman(bks, good, bad)
    assert 0.0 <= pf <= 1.0

    ranks = average_ranks(np.column_stack([bks, good, bad]))
    assert ranks[0] < ranks[1] < ranks[2]                        # bks best, bad worst

    print("eval/stats.py self-test OK "
          f"(gap good=2.00%, wilcoxon p_diff={p_diff:.3g}, "
          f"friedman p={pf:.3g}, ranks={np.round(ranks, 2)})")


def main():
    p = argparse.ArgumentParser(description="HDCARP statistical-rigor helpers.")
    p.add_argument("--selftest", action="store_true", help="run a synthetic self-check")
    p.add_argument("--tightness", type=str, default=None,
                   help="report tau distribution over .npz under this dir")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    if args.tightness:
        print("tightness tau:", describe(tightness_from_dir(args.tightness)))


if __name__ == "__main__":
    main()
