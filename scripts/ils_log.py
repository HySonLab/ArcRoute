"""Run ILS on one instance and write the route log to a file.

Usage:
    uv run python scripts/ils_log.py [--file <path.npz>] [--variant P|U]
                                     [--seed 42] [--num_sample 20]
                                     [--log outputs/route_log.txt]
    uv run python scripts/ils_log.py \
        --file data/ood/osm_cityA/40/38_17_358.npz \
        --variant P \
        --vehicles 2 \
        --num_sample 20 \
        --log outputs/my_solution.txt
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from solvers.cal_reward import get_Ts
from solvers.meta import InsertCheapestHCARP
from utils.local_search import ls
from utils.nb_utils import deserialize_tours_batch


def print_route_log(routes, req, Ts, solver):
    adj = solver.vars["adj"]
    arc_info = {
        i + 1: dict(
            tail=int(req[i, 0]),
            head=int(req[i, 1]),
            clss=int(req[i, 3]),
            service=float(req[i, 4]),
            demand=float(solver.demands[i + 1]),
        )
        for i in range(len(req))
    }

    T_str = "  ".join(f"T{k + 1}={v:.4f}" for k, v in enumerate(Ts))
    print(f"\n{T_str}\n")

    for vid, route in enumerate(routes):
        arcs = route[1:-1]
        total_load = solver.demands[arcs].sum()
        bar = "=" * 60
        print(bar)
        print(f"  VEHICLE {vid + 1}   load={total_load:.4f}/1.0000")
        print(bar)

        # ── node chain ───────────────────────────────────────────────
        chain = ["depot"]
        prev_head = None
        for arc_idx in arcs:
            info = arc_info[arc_idx]
            t, h = info["tail"], info["head"]
            if prev_head is not None and prev_head != t:
                chain.append(f"~~{prev_head}->{t}~~")
            chain.append(f"{t}->{h}")
            prev_head = h
        chain.append("depot")

        line, indent = "  ", "    "
        for k, part in enumerate(chain):
            sep = " -> " if k > 0 else ""
            if len(line) + len(sep) + len(part) > 70:
                print(line)
                line = indent + part
            else:
                line += sep + part
        print(line)
        print()

        # ── sequential arc table (full route order) ──────────────────
        hdr = f"  {'seq':>3}  {'arc':>3}  {'edge':>7}  {'cls'}  {'demand':>7}  {'service':>7}  {'dead':>7}  {'cum_load':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        cum = 0.0
        for pos, arc_idx in enumerate(arcs):
            info = arc_info[arc_idx]
            prev_arc = arcs[pos - 1] if pos > 0 else 0
            dead = float(adj[prev_arc, arc_idx])
            cum += info["demand"]
            print(
                f"  {pos + 1:>3}  {arc_idx:>3}  "
                f"{info['tail']:>2}->{info['head']:>2}  "
                f"  {info['clss']}  "
                f"{info['demand']:>7.4f}  "
                f"{info['service']:>7.4f}  "
                f"{dead:>7.4f}  "
                f"{cum:>8.4f}"
            )
        print()

        # ── per-class breakdown ──────────────────────────────────────
        for p in [1, 2, 3]:
            class_arcs = [
                (pos, a) for pos, a in enumerate(arcs) if arc_info[a]["clss"] == p
            ]
            if not class_arcs:
                continue
            class_load = sum(arc_info[a]["demand"] for _, a in class_arcs)
            print(f"  Class {p}  ({len(class_arcs)} arcs, load={class_load:.4f})")
            for pos, arc_idx in class_arcs:
                info = arc_info[arc_idx]
                prev_arc = arcs[pos - 1] if pos > 0 else 0
                dead = float(adj[prev_arc, arc_idx])
                print(
                    f"    [seq {pos + 1:2d}] arc {arc_idx:2d}  "
                    f"{info['tail']:2d} -> {info['head']:2d}   "
                    f"demand={info['demand']:.4f}  "
                    f"service={info['service']:.4f}  "
                    f"dead={dead:.4f}"
                )
            print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/ood/osm_cityB/40/34_13_632.npz")
    parser.add_argument("--variant", default="P", choices=["P", "U"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_sample", type=int, default=20)
    parser.add_argument("--log", default="outputs/route_log.txt")
    parser.add_argument(
        "--solution",
        default=None,
        help="Save solution to this .txt file (default: <log>.sol)",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=None,
        help="Number of vehicles (default: use value stored in instance)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    es = np.load(args.file)

    solver = InsertCheapestHCARP()
    solver.import_instance(args.file, M=args.vehicles)

    from time import time

    t0 = time()
    raw_actions = [
        solver.get_once(solver.M, solver.clss) for _ in range(args.num_sample)
    ]
    raw_actions = [a for a in raw_actions if a is not None]

    if not raw_actions:
        total_demand = float(solver.demands[1:].sum())
        max_capacity = solver.nv * 1.0
        print(f"ERROR: No feasible solution found with {solver.nv} vehicle(s).")
        print(
            f"  Total demand = {total_demand:.4f}  >  fleet capacity = {max_capacity:.1f}"
        )
        min_vehicles = int(np.ceil(total_demand - 1e-6))
        print(f"  Minimum vehicles required: {min_vehicles}")
        return

    tours_batch = ls(solver.vars, variant=args.variant, actions=raw_actions)
    actions_arr = deserialize_tours_batch(tours_batch, solver.nseq)
    Ts_all = get_Ts(solver.vars, actions=actions_arr)
    best_idx = (Ts_all @ np.array([1e3, 1e1, 1e-1])).argmin()
    best_T = Ts_all[best_idx]
    best_routes = tours_batch[best_idx]
    elapsed = time() - t0

    req = es["req"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Instance : {args.file}")
        print(
            f"P={es['P']} classes | M={solver.nv} vehicles | "
            f"{len(req)} required arcs | solved in {elapsed:.3f}s\n"
        )
        print_route_log(best_routes, req, best_T, solver)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w") as fh:
        fh.write(text)
    print(f"Saved -> {args.log}")

    # save human-readable solution file
    sol_path = args.solution or (os.path.splitext(args.log)[0] + ".sol")
    with open(sol_path, "w") as fh:
        fh.write(f"instance: {args.file}\n")
        fh.write(f"variant: {args.variant}\n")
        fh.write(f"vehicles: {solver.nv}\n")
        fh.write(f"T1: {best_T[0]:.6f}  T2: {best_T[1]:.6f}  T3: {best_T[2]:.6f}\n")
        for i, r in enumerate(best_routes):
            fh.write(f"route {i + 1}: {' '.join(str(a) for a in r)}\n")
    print(f"Saved -> {sol_path}")


if __name__ == "__main__":
    main()
