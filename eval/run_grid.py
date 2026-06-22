#!/usr/bin/env python
"""Phase 5 — eval grid (dynamic_plan).

Solve every OOD instance under each fleet size M with the trained RL policy, then
summarise the hierarchical makespan (T_1, T_2, T_3) by M / size / density /
topology. M is a SOLVE-TIME parameter: the policy is M-agnostic, M is re-applied
only in the Scheduler (common/scheduler.py), so one model serves any M.

Examples
--------
    # real eval (needs a trained checkpoint)
    uv run python eval/run_grid.py --ckpt checkpoints/clP_ladder/last.ckpt \
        --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 \
        --out eval/results_P.csv

    # structural smoke (no model): exercises loading + aggregation + output
    uv run python eval/run_grid.py --path data/ood --M 2,3,5,7 --dry-run --limit 8

Note: `env.get_objective` evaluates a batch via run_parallel(num_epochs=10), so
`--num_sample` must be >= 10 (we clamp). Sampling decode + keep the best by T_1
(HRDA-style), matching baseline/rl_hyb.
"""
import os
import sys
import csv
import time
import glob
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from eval.stats import describe, gap_summary, pairwise_wilcoxon, friedman

POS = ("T1", "T2", "T3")


# --------------------------------------------------------------------------- #
# Instance loading + metadata (no model needed)
# --------------------------------------------------------------------------- #
def read_meta(file):
    d = np.load(file)
    keys = d.files
    return {
        "file": os.path.basename(file),
        "topology": str(d["topology"]) if "topology" in keys else "unknown",
        "n_req": int(d["n_req"]) if "n_req" in keys else int(d["req"].shape[0]),
        "density": float(d["d"]) if "d" in keys else float("nan"),
        "tau": float(d["tau"]) if "tau" in keys else float("nan"),
    }


def load_instance_td(file, M):
    """Build a batch-1 td (env.reset input keys) from a .npz, with fleet M."""
    import torch
    from tensordict import TensorDict
    from common.ops import import_instance

    dms, P, _M, demands, clss, s, d, _edge = import_instance(file, M=M)
    td_in = TensorDict(
        {
            "demands": torch.tensor(demands, dtype=torch.float32)[None],
            "service_times": torch.tensor(s, dtype=torch.float32)[None],
            "traversal_times": torch.tensor(d, dtype=torch.float32)[None],
            "adj": torch.tensor(dms, dtype=torch.float32)[None],
            "clss": torch.tensor(clss, dtype=torch.int64)[None],
            "num_vehicle": torch.tensor([int(M)], dtype=torch.int64),
        },
        batch_size=[1],
    )
    return td_in, len(demands)


def iter_files(path, limit=None):
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    return files[:limit] if limit else files


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
class RLSolver:
    """Loads a PPO checkpoint and solves an instance under a given (M, variant)
    by sampling `num_sample` rollouts and keeping the best by T_1."""

    def __init__(self, ckpt, device="cuda"):
        import torch
        from rl.ppo import PPO

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = PPO.load_from_checkpoint(ckpt, map_location=self.device)
        self.policy = self.model.policy.to(self.device).eval()
        self.env = self.model.env

    def solve(self, file, M, variant, num_sample):
        import torch
        from common.ops import batchify

        td_in, _ = load_instance_td(file, M)
        self.env.variant = variant
        td = self.env.reset(td_in)
        td = batchify(td, num_sample).to(self.device)
        t0 = time.time()
        with torch.inference_mode():
            out = self.policy(td, env=self.env, decode_type="sampling")
            obj = np.asarray(self.env.get_objective(td, out["actions"]))
        # lexicographic best over (T_1,T_2,T_3): T_1 primary, then T_2, then T_3
        # (np.lexsort uses the LAST key as primary). Ties on T_1 -> break by T_2/T_3.
        best = obj[np.lexsort((obj[:, 2], obj[:, 1], obj[:, 0]))[0]]
        return np.asarray(best, dtype=float), time.time() - t0


class DryRunSolver:
    """No model — synthetic T that decreases with M (more vehicles -> lower
    makespan). Lets --dry-run validate loading + aggregation + output."""

    def solve(self, file, M, variant, num_sample):
        meta = read_meta(file)
        base = max(meta["n_req"], 1) / max(int(M), 1)   # ~ work per vehicle
        T = np.array([base, base * 1.1, base * 1.2], dtype=float)
        return T, 0.0


# --------------------------------------------------------------------------- #
# Grid run + output
# --------------------------------------------------------------------------- #
def run_grid(solver, files, Ms, variants, num_sample, algo="rl"):
    rows = []
    for f in files:
        meta = read_meta(f)
        for variant in variants:
            for M in Ms:
                try:
                    T, dt = solver.solve(f, M, variant, num_sample)
                except Exception as e:                  # keep going; record failure
                    print(f"  ! {meta['file']} M={M} {variant}: {e}", file=sys.stderr)
                    continue
                # D2 Phase 6: `algo` tags the learning signal (grpo/ppo) so a paired
                # A/B win-rate can be computed across the SAME (file, M, variant) grid.
                rows.append({**meta, "algo": algo, "variant": variant, "M": int(M),
                             "T1": float(T[0]), "T2": float(T[1]), "T3": float(T[2]),
                             "time_s": round(float(dt), 4)})
    return rows


def yield_curve(solver, files, M, variant, Ks):
    """D2 Phase 6: best-of-K yield — for each K, the lex-best T over K samples.
    Returns one row per (file, K). lex-best T is monotone non-worsening in K."""
    rows = []
    for f in files:
        meta = read_meta(f)
        for K in Ks:
            try:
                T, dt = solver.solve(f, M, variant, max(int(K), 10))
            except Exception as e:
                print(f"  ! {meta['file']} K={K}: {e}", file=sys.stderr)
                continue
            rows.append({**meta, "variant": variant, "M": int(M), "K": int(K),
                         "T1": float(T[0]), "T2": float(T[1]), "T3": float(T[2]),
                         "time_s": round(float(dt), 4)})
    return rows


def write_csv(rows, out):
    if not rows:
        print("no rows to write", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Union of base + optional (algo/K) columns; keep base order stable.
    base = ["file", "topology", "n_req", "density", "tau",
            "variant", "M", "T1", "T2", "T3", "time_s"]
    extra = [k for k in ("algo", "K") if any(k in r for r in rows)]
    fields = base[:6] + extra + base[6:]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


# --------------------------------------------------------------------------- #
# Summary (eval/stats.py)
# --------------------------------------------------------------------------- #
def _group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return g


def summarize(rows):
    if not rows:
        return
    by_variant = _group(rows, "variant")
    for variant, vrows in sorted(by_variant.items()):
        print(f"\n===== variant {variant} =====")

        # T_1 by fleet M (+ monotonicity sanity: mean T_1 should not rise with M)
        print("T_1 by M:")
        means = []
        for M, mrows in sorted(_group(vrows, "M").items()):
            st = describe([r["T1"] for r in mrows])
            means.append((M, st["mean"]))
            print(f"  M={M:<3} n={st['n']:<4} mean={st['mean']:.3f} "
                  f"std={st['std']:.3f} [{st['min']:.2f},{st['max']:.2f}]")
        mono = all(b <= a + 1e-9 for (_, a), (_, b) in zip(means, means[1:]))
        print(f"  monotone (T_1 non-increasing in M): {mono}")

        # break-downs
        for axis in ("topology", "n_req", "density"):
            print(f"T_1 by {axis}:")
            for v, vr in sorted(_group(vrows, axis).items(), key=lambda kv: str(kv[0])):
                st = describe([r["T1"] for r in vr])
                print(f"  {axis}={v}: n={st['n']} mean={st['mean']:.3f}")


def paired_win_rate(rows_a, rows_b):
    """D2 Phase 6: lexicographic win-rate of A vs B over the shared (file, M,
    variant) grid (paired). Returns the eval.stats.win_rate dict; t1_regression
    MUST be 0 (no T_1 regression invariant)."""
    from eval.stats import win_rate

    def key(r):
        return (r["file"], int(r["M"]), r["variant"])

    bmap = {key(r): r for r in rows_b}
    A, B = [], []
    for r in rows_a:
        k = key(r)
        if k in bmap:
            A.append([r["T1"], r["T2"], r["T3"]])
            B.append([bmap[k]["T1"], bmap[k]["T2"], bmap[k]["T3"]])
    return win_rate(A, B)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 eval grid (RL over fleet M).")
    p.add_argument("--ckpt", type=str, default=None, help="PPO checkpoint (.ckpt)")
    p.add_argument("--path", type=str, default="data/ood",
                   help="OOD dir (recursed for *.npz) or a single .npz")
    p.add_argument("--M", type=str, default="2,3,5,7,10", help="comma fleet sizes")
    p.add_argument("--variant", type=str, default="P", help="comma variants e.g. P,U")
    p.add_argument("--num_sample", type=int, default=100,
                   help="rollouts/instance; best by T_1 is kept (>=10)")
    p.add_argument("--limit", type=int, default=None, help="cap #files (quick runs)")
    p.add_argument("--out", type=str, default="eval/results.csv")
    p.add_argument("--algo", type=str, default="rl",
                   help="tag (grpo/ppo/...) written to the algo column for A/B")
    p.add_argument("--yield_curve", action="store_true",
                   help="best-of-K yield: sweep num_sample/K and record lex-best T")
    p.add_argument("--Ks", type=str, default="1,2,4,8,16,32",
                   help="K sweep for --yield_curve")
    p.add_argument("--dry-run", action="store_true",
                   help="no model: synthetic T to validate the scaffold")
    return p.parse_args()


def main():
    a = parse_args()
    Ms = [int(x) for x in a.M.split(",")]
    variants = [v.strip() for v in a.variant.split(",")]
    num_sample = max(a.num_sample, 10)                  # get_objective needs >=10
    files = iter_files(a.path, a.limit)
    print(f"{len(files)} files | M={Ms} | variants={variants} | "
          f"{'DRY-RUN' if a.dry_run else 'ckpt=' + str(a.ckpt)}")

    if a.dry_run:
        solver = DryRunSolver()
    else:
        if not a.ckpt:
            sys.exit("--ckpt is required (or use --dry-run)")
        solver = RLSolver(a.ckpt)

    if a.yield_curve:
        Ks = [int(x) for x in a.Ks.split(",")]
        rows = yield_curve(solver, files, Ms[0], variants[0], Ks)
        write_csv(rows, a.out)
        print("\nbest-of-K yield (lex-best T per K; should not worsen as K grows):")
        from eval.run_grid import _group
        for K, kr in sorted(_group(rows, "K").items()):
            st = describe([r["T1"] for r in kr])
            print(f"  K={K:<3} n={st['n']} T1_mean={st['mean']:.3f}")
        return

    rows = run_grid(solver, files, Ms, variants, num_sample, algo=a.algo)
    write_csv(rows, a.out)
    summarize(rows)
    # TODO: RL-vs-baseline (EA/ACO/ILS/LP) under the SAME (M, variant) -> feed
    #       pairwise_wilcoxon / friedman / gap_summary here once baseline objs
    #       are collected (baselines are separate CLIs; add an adapter).


if __name__ == "__main__":
    main()
