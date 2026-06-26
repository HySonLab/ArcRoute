#!/usr/bin/env python3
"""MILP-P / MILP-U route log — same output format as ils_log.py / ea_log.py / aco_log.py.

Usage:
  uv run python scripts/lp_log.py --path data/ood/osm_cityB/40/38_14_859.npz
  uv run python scripts/lp_log.py --path ... --variant U --time_limit 300
  uv run python scripts/lp_log.py --path ... --variant P --log outputs/milp_p_log.txt
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from solvers.lp import solve_milp_p, solve_milp_u
from utils.ops import import_instance


# --------------------------------------------------------------------------- #
# Route reconstruction from MILP solution variables
# --------------------------------------------------------------------------- #

def _greedy_order(arc_list, adj, start=0):
    """Order arc_list by greedy nearest-neighbour from start arc (1-based indices)."""
    remaining = list(arc_list)
    ordered = []
    cur = start
    while remaining:
        best_i = min(range(len(remaining)), key=lambda i: adj[cur, remaining[i]])
        nxt = remaining.pop(best_i)
        ordered.append(nxt)
        cur = nxt
    return ordered


def _arc_to_idx_map(req_raw):
    """Return dict (tail, head) -> 1-based arc index from raw req array."""
    return {(int(req_raw[i, 0]), int(req_raw[i, 1])): i + 1
            for i in range(len(req_raw))}


def reconstruct_routes_p(model, x_vars, G, req_raw, adj):
    """Per-vehicle arc sequences for MILP-P.

    Arcs are grouped by priority class (class 1 first, then 2, ...) and
    within each class ordered greedily by minimum deadheading from the
    previous arc.
    """
    M_n = G['M']
    P = G['P']
    req_arcs = G['req_arcs']
    cls_g = G['cls']
    a2i = _arc_to_idx_map(req_raw)

    routes = []
    for m in range(M_n):
        route_arcs = []
        prev = 0
        for k in range(1, P + 1):
            arcs_k = [a2i[a] for a in req_arcs
                      if cls_g[a] == k and model.getVal(x_vars[(m, a)]) > 0.5]
            if arcs_k:
                ordered = _greedy_order(arcs_k, adj, prev)
                route_arcs.extend(ordered)
                prev = route_arcs[-1]
        routes.append(np.array([0] + route_arcs + [0], dtype=np.int32))
    return routes


def reconstruct_routes_u(model, x_vars, G, req_raw, adj):
    """Per-vehicle arc sequences for MILP-U.

    Arcs are grouped by hierarchy level h (level 1 first, ...) and within
    each level ordered greedily. The hierarchy level determines service order
    (not priority class), so different classes can interleave.
    """
    M_n = G['M']
    P = G['P']
    req_arcs = G['req_arcs']
    a2i = _arc_to_idx_map(req_raw)

    routes = []
    for m in range(M_n):
        route_arcs = []
        prev = 0
        for h in range(1, P + 1):
            arcs_h = [a2i[a] for a in req_arcs
                      if model.getVal(x_vars[(m, a, h)]) > 0.5]
            if arcs_h:
                ordered = _greedy_order(arcs_h, adj, prev)
                route_arcs.extend(ordered)
                prev = route_arcs[-1]
        routes.append(np.array([0] + route_arcs + [0], dtype=np.int32))
    return routes


# --------------------------------------------------------------------------- #
# Log printer (same columns as ils_log.py)
# --------------------------------------------------------------------------- #

def print_route_log(routes, req_raw, demands, adj, Tvec):
    nseq = len(req_raw)
    P = int(req_raw[:, 3].max())

    arc_info = {
        i + 1: dict(
            tail=int(req_raw[i, 0]),
            head=int(req_raw[i, 1]),
            clss=int(req_raw[i, 3]),
            service=float(req_raw[i, 4]),
            demand=float(demands[i + 1]),
        )
        for i in range(nseq)
    }

    T_str = "  ".join(f"T{k+1}={v:.4f}" for k, v in enumerate(Tvec))
    print(f"\n{T_str}\n")

    for vid, route in enumerate(routes):
        arcs = route[1:-1]
        total_load = float(demands[arcs].sum()) if len(arcs) > 0 else 0.0
        bar = "=" * 60
        print(bar)
        print(f"  VEHICLE {vid + 1}   load={total_load:.4f}/1.0000")
        print(bar)

        # node chain
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

        # arc table
        hdr = (f"  {'seq':>3}  {'arc':>3}  {'edge':>7}  {'cls'}"
               f"  {'demand':>7}  {'service':>7}  {'dead':>7}  {'cum_load':>8}")
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

        # per-class breakdown
        for p in range(1, P + 1):
            class_arcs = [(pos, a) for pos, a in enumerate(arcs)
                          if arc_info[a]["clss"] == p]
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


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="MILP-P/U route log")
    ap.add_argument("--path", required=True, help="Path to .npz instance file")
    ap.add_argument("--variant", default="P", choices=["P", "U"],
                    help="HDCARP variant: P (precedence) or U (upgrading)")
    ap.add_argument("--time_limit", type=int, default=3600,
                    help="SCIP time limit in seconds (default 3600)")
    ap.add_argument("--threads", type=int, default=8,
                    help="Max SCIP threads (default 8)")
    ap.add_argument("--log", default=None,
                    help="Write log to this file in addition to stdout")
    args = ap.parse_args()

    es = np.load(args.path)
    req_raw = np.asarray(es['req'])
    nseq = len(req_raw)
    M_n = int(es['M'])
    P_n = int(es['P'])

    # Load normalised adj and demands (arc-to-arc cost, demand/C)
    dms, _, _, demands, _, _, _, _ = import_instance(args.path)

    print(f"MILP-{args.variant} solver")
    print(f"Instance : {args.path}")
    print(f"P={P_n} classes | M={M_n} vehicles | {nseq} required arcs")
    print(f"Time limit: {args.time_limit}s | Threads: {args.threads}")
    print("Solving...", flush=True)

    solver_fn = solve_milp_p if args.variant == "P" else solve_milp_u
    t0 = time.time()
    result = solver_fn(es, time_limit=args.time_limit, threads=args.threads,
                       verbose=False, return_model=True)
    elapsed = time.time() - t0

    Tvec, model, vars_dict = result
    if Tvec is None:
        print(f"\nNo feasible solution found within {elapsed:.1f}s.")
        return

    G = vars_dict['G']
    x = vars_dict['x']

    print(f"Solved in {elapsed:.3f}s\n")

    if args.variant == "P":
        routes = reconstruct_routes_p(model, x, G, req_raw, dms)
    else:
        routes = reconstruct_routes_u(model, x, G, req_raw, dms)

    header = (
        f"MILP-{args.variant} solver\n"
        f"Instance : {args.path}\n"
        f"P={P_n} classes | M={M_n} vehicles | {nseq} required arcs"
        f" | solved in {elapsed:.3f}s\n"
    )

    if args.log:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(header)
            print_route_log(routes, req_raw, demands, dms, Tvec)
        out = buf.getvalue()
        print(out)
        with open(args.log, "w") as f:
            f.write(out)
        print(f"Log written to {args.log}")
    else:
        print(header)
        print_route_log(routes, req_raw, demands, dms, Tvec)


if __name__ == "__main__":
    main()
