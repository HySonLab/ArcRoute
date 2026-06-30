"""Iterated Local Search solver for HDCARP.

Usage (single instance — route log):
    uv run python -m solvers.ils --file data/ood/osm_cityB/40/34_13_632.npz \\
        --variant P --vehicles 2 --max_iter 200 --log outputs/ils_route_log.txt

Usage (batch benchmark):
    uv run python -m solvers.ils --path data/5m --variant P --max_iter 200
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import contextlib
import io
from glob import glob
from time import time

import numpy as np

from solvers.meta import ILSHCARP
from utils.ops import check_feasibility, gen_tours, print_route_log, save_sol


def parse_args():
    parser = argparse.ArgumentParser(description="ILSHCARP")
    parser.add_argument('--seed', type=int, default=6868)
    parser.add_argument('--variant', default='P', choices=['P', 'U'])
    parser.add_argument('--num_init_sample', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--strength', type=int, default=3)
    parser.add_argument('--accept_mode', default='best', choices=['best', 'sa'])
    # single-instance log mode
    parser.add_argument('--file', default=None, help='.npz instance; enables log mode')
    parser.add_argument('--vehicles', type=int, default=None, help='override fleet size')
    parser.add_argument('--log', default='outputs/ils_route_log.txt')
    parser.add_argument('--solution', default=None)
    # batch benchmark mode
    parser.add_argument('--path', default='data/5m', help='directory of .npz instances')
    parser.add_argument('--M', type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    if args.file:
        # ── single-instance log mode ───────────────────────────────────────
        es = np.load(args.file)
        solver = ILSHCARP(strength=args.strength, accept_mode=args.accept_mode)
        solver.import_instance(args.file, M=args.vehicles)

        if not check_feasibility(solver.demands, solver.nv):
            raise SystemExit(1)

        t0 = time()
        best_T = solver(
            max_iter=args.max_iter, variant=args.variant,
            num_init_sample=args.num_init_sample, seed=args.seed, verbose=True,
        )[0]
        elapsed = time() - t0

        best_routes = gen_tours(solver._last_best_action)
        req = es["req"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(f"Instance : {args.file}")
            print(f"P={es['P']} classes | M={solver.nv} vehicles | "
                  f"{len(req)} required arcs | solved in {elapsed:.3f}s\n")
            print_route_log(best_routes, req, solver.demands, solver.vars["adj"], best_T)
        text = buf.getvalue()
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as fh:
            fh.write(text)
        print(f"Saved -> {args.log}")
        sol_path = args.solution or (os.path.splitext(args.log)[0] + ".sol")
        save_sol(sol_path, best_routes, best_T,
                 instance=args.file, variant=args.variant, vehicles=solver.nv)
        print(f"Saved -> {sol_path}")
    else:
        # ── batch benchmark mode ───────────────────────────────────────────
        files = sorted(glob(args.path + '/*/*.npz'))
        solver = ILSHCARP(strength=args.strength, accept_mode=args.accept_mode)
        for f in files:
            solver.import_instance(f, M=args.M)
            t1 = time()
            print(f, ':::',
                  solver(max_iter=args.max_iter, variant=args.variant,
                         num_init_sample=args.num_init_sample, seed=args.seed),
                  ':::', time() - t1)
