"""Validate a saved ILS solution against its instance.

Inputs:
  --instance  <path.npz>          the problem instance (from data/)
  --solution  <solution.npz>      solution saved by ils_log.py
  --log       <output.txt>        where to write the report (default: stdout only)

Checks:
  1. Coverage      – every required arc served exactly once
  2. Capacity      – per-route demand <= 1.0 (normalized)
  3. Depot sentinels – each route starts and ends with arc 0
  4. Class ordering – variant P: class sequence non-decreasing per route
  5. Deadhead adj  – adj[i,j] == Floyd-Warshall(head_i -> tail_j) for all pairs
  6. Service times – loaded service_time array matches req[:,4]
  7. T1/T2/T3 (ls routes)   – manual timing on solution routes
  8. T1/T2/T3 (Scheduler)   – Scheduler re-partitions then re-checks
     Note: Scheduler._build_P strips 0-separators and re-partitions by
     class-balanced chunks, so its routes may differ from the saved ones.

Usage:
    uv run python scripts/validate_solution.py \\
        --instance data/ood/osm_cityB/40/34_13_632.npz \\
        --solution outputs/route_log.npz \\
        --log outputs/validate.txt
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from solvers.scheduler import Scheduler
from utils.ops import (convert_adjacency_matrix, floyd_warshall,
                       dist_edges, import_instance)

PASS = '[PASS]'
FAIL = '[FAIL]'
INFO = '[INFO]'
EPS  = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path):
    """Load .npz instance and derive adj, service_time, clss, demands."""
    es  = np.load(path)
    dms, P, M, demands, clss, svc, d, edge_idxs = import_instance(es)
    adj = dms.astype(np.float64)
    svc = svc.astype(np.float64)
    clss = clss.astype(np.int64)
    demands = demands.astype(np.float64)
    return es, adj, svc, clss, demands


def load_solution(path):
    """Load solution .sol text file saved by ils_log.py.

    Format (one key: value per line):
      instance: <path>
      variant: P|U
      T1: <f>  T2: <f>  T3: <f>
      route 1: 0 a1 a2 ... 0
      route 2: 0 b1 b2 ... 0
      ...

    Returns:
      routes   – list of 1D int32 arrays, each [0, a1, ..., ak, 0]
      T_saved  – np.ndarray (3,) T1/T2/T3 at save time
      variant  – 'P' or 'U'
      instance – original instance path string
    """
    routes, T_saved, variant, instance, vehicles = [], np.zeros(3), 'P', '', None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, _, rest = line.partition(':')
            key = key.strip()
            rest = rest.strip()
            if key == 'instance':
                instance = rest
            elif key == 'variant':
                variant = rest
            elif key == 'vehicles':
                vehicles = int(rest)
            elif key == 'T1':
                parts = rest.replace('T1:', '').replace('T2:', '').replace('T3:', '').split()
                T_saved = np.array([float(p) for p in parts[:3]])
            elif key.startswith('route'):
                routes.append(np.array([int(x) for x in rest.split()], dtype=np.int32))
    # fall back to number of routes if field absent (old format)
    if vehicles is None:
        vehicles = len(routes)
    return routes, T_saved, variant, instance, vehicles


def build_node_dms(es):
    req    = es['req']
    nonreq = es['nonreq']
    all_e  = np.vstack([req, nonreq])
    adj_n  = convert_adjacency_matrix(all_e[:, 0], all_e[:, 1], all_e[:, 5])
    return floyd_warshall(adj_n)


# ─────────────────────────────────────────────────────────────────────────────
# timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def trip_times(trip, adj, svc):
    """Arc completion times + total trip duration (including return to depot)."""
    path = np.array([0] + list(trip) + [0], dtype=np.int64)
    t    = svc[path].copy()
    t[1:] += adj[path[:-1], path[1:]]
    cum  = np.cumsum(t)
    n    = len(trip)
    return cum[1:n + 1], float(cum[-1])


def manual_Ts_routes(routes, adj, svc, clss, pos_val=(1, 2, 3)):
    per = {p: [0.0] for p in pos_val}
    for route in routes:
        trip = route[1:-1]
        if len(trip) == 0:
            continue
        comps, _ = trip_times(trip, adj, svc)
        for arc, t in zip(trip, comps):
            c = int(clss[arc])
            if c in per:
                per[c].append(float(t))
    return np.array([max(per[p]) for p in pos_val])


def manual_Ts_vehicles(vehicles, adj, svc, clss, pos_val=(1, 2, 3)):
    per = {p: [0.0] for p in pos_val}
    for veh_trips in vehicles:
        offset = 0.0
        for trip in veh_trips:
            comps, dur = trip_times(trip, adj, svc)
            for arc, t in zip(trip, comps):
                c = int(clss[arc])
                if c in per:
                    per[c].append(offset + float(t))
            offset += dur
    return np.array([max(per[p]) for p in pos_val])


# ─────────────────────────────────────────────────────────────────────────────
# checks
# ─────────────────────────────────────────────────────────────────────────────

def check_coverage(routes, n_req):
    served   = sorted(int(a) for r in routes for a in r[1:-1])
    expected = list(range(1, n_req + 1))
    if served == expected:
        return True, f'{PASS} Coverage: all {n_req} arcs served exactly once'
    missing = sorted(set(expected) - set(served))
    extra   = sorted(set(served)   - set(expected))
    dups    = len(served) - len(set(served))
    return False, (f'{FAIL} Coverage: missing={missing}  extra={extra}  '
                   f'duplicates={dups}')


def check_capacity(routes, demands):
    lines, ok = [], True
    for vid, route in enumerate(routes):
        load = float(demands[route[1:-1]].sum())
        over = load > 1.0 + EPS
        ok  &= not over
        tag  = FAIL if over else PASS
        note = f'  OVER by {load - 1:.2e}' if over else ''
        lines.append(f'  {tag} Vehicle {vid+1}: load={load:.6f}/1.000000{note}')
    return ok, lines


def check_sentinels(routes):
    lines, ok = [], True
    for vid, route in enumerate(routes):
        bad = (route[0] != 0 or route[-1] != 0)
        ok &= not bad
        tag = FAIL if bad else PASS
        lines.append(f'  {tag} Vehicle {vid+1}: starts={route[0]} ends={route[-1]}')
    return ok, lines


def check_class_order(routes, clss, variant):
    if variant != 'P':
        return True, [f'  {PASS} Class ordering: skipped (variant=U)']
    lines, ok = [], True
    for vid, route in enumerate(routes):
        arcs = route[1:-1]
        cs   = [int(clss[a]) for a in arcs]
        bad  = [(i+1, cs[i-1], cs[i]) for i in range(1, len(cs)) if cs[i] < cs[i-1]]
        if bad:
            ok = False
            lines.append(f'  {FAIL} Vehicle {vid+1}: class decreases at seq '
                         + ', '.join(f'{s}({a}->{b})' for s, a, b in bad))
        else:
            lines.append(f'  {PASS} Vehicle {vid+1}: class sequence '
                         f'non-decreasing {cs}')
    return ok, lines


def check_deadhead(adj_arc, node_dms, req):
    n    = len(req) + 1          # arc 0 = depot sentinel
    tails = np.array([0] + [int(req[i, 0]) for i in range(len(req))], dtype=np.int64)
    heads = np.array([0] + [int(req[i, 1]) for i in range(len(req))], dtype=np.int64)
    max_err, n_bad = 0.0, 0
    for i in range(n):
        for j in range(n):
            exp = float(node_dms[heads[i], tails[j]])
            got = float(adj_arc[i, j])
            err = abs(exp - got)
            if err > EPS:
                n_bad  += 1
                max_err = max(max_err, err)
    if n_bad == 0:
        return True, (f'{PASS} Deadhead adj: all {n*n} entries match '
                      f'Floyd-Warshall node distances (max_err=0)')
    return False, (f'{FAIL} Deadhead adj: {n_bad}/{n*n} entries wrong, '
                   f'max_err={max_err:.2e}')


def check_service_times(svc, req):
    got  = svc[1:].astype(np.float64)
    exp  = req[:, 4].astype(np.float64)
    merr = float(np.abs(got - exp).max())
    if merr <= EPS:
        return True, (f'{PASS} Service times: all {len(req)} arcs match '
                      f'req[:,4] (max_err={merr:.2e})')
    n_bad = int((np.abs(got - exp) > EPS).sum())
    return False, (f'{FAIL} Service times: {n_bad} mismatches, '
                   f'max_err={merr:.2e}')


def timing_table(rows, header):
    lines = [header,
             f'  {"veh":>3}  {"trp":>3}  {"seq":>3}  {"arc":>3}  '
             f'{"cls"}  {"dead":>7}  {"service":>7}  {"t_finish":>9}',
             '  ' + '-' * 56]
    for r in rows:
        lines.append(r)
    return lines


def check_timing(routes, adj, svc, clss, demands_, T_saved, variant, M, pos_val=(1, 2, 3)):
    lines, ok = [], True

    # ── A: manual on saved routes ────────────────────────────────────────────
    Ts_ls = manual_Ts_routes(routes, adj, svc, clss, pos_val)
    lines.append('  A. T from saved routes (manual):')
    lines.append(f'  {"":4} {"Manual":>12}  {"Saved":>12}  {"diff":>10}')
    for k, (tm, ts) in enumerate(zip(Ts_ls, T_saved)):
        diff = abs(tm - ts)
        # routes T and Scheduler T can legitimately differ (re-partition)
        lines.append(f'  {INFO} T{k+1}: {tm:12.6f}  {ts:12.6f}  {diff:10.4f}')

    rows = []
    for vid, route in enumerate(routes):
        trip = route[1:-1]
        comps, dur = trip_times(trip, adj, svc)
        for pos, (arc, t) in enumerate(zip(trip, comps)):
            prev = trip[pos - 1] if pos > 0 else 0
            rows.append(
                f'  {vid+1:>3}  {"1":>3}  {pos+1:>3}  {arc:>3}  '
                f'  {int(clss[arc])}  {adj[prev,arc]:>7.4f}  '
                f'{svc[arc]:>7.4f}  {t:>9.4f}')
        ret = float(adj[trip[-1], 0]) if len(trip) else 0.0
        rows.append(f'  {"":3}  {"":3}  {"":3}  {"":3}  '
                    f'     dead_ret={ret:.4f}  dur={dur:.4f}')
        rows.append('')
    lines.extend(timing_table(rows, '  Per-arc timing (saved routes):'))
    lines.append('')

    # ── B: Scheduler re-partition ────────────────────────────────────────────
    flat = np.array([a for r in routes for a in r], dtype=np.int64)
    td   = {'adj': adj, 'service_times': svc, 'clss': clss, 'demand': demands_}
    sched = Scheduler(variant=variant, pos_val=tuple(pos_val))
    vehicles, _ = sched(flat, td, M=M)
    Ts_sched    = manual_Ts_vehicles(vehicles, adj, svc, clss, pos_val)

    lines.append('  B. T from Scheduler re-partitioned routes (manual re-check):')
    lines.append(f'  {"":4} {"Saved":>12}  {"Sched(manual)":>14}  {"diff":>10}')
    for k, (ts, tm) in enumerate(zip(T_saved, Ts_sched)):
        diff = abs(ts - tm)
        tag  = PASS if diff <= EPS else FAIL
        ok  &= diff <= EPS
        lines.append(f'  {tag} T{k+1}: {ts:12.6f}  {tm:14.6f}  {diff:10.2e}')

    lines.append('')
    lines.append(f'  {INFO} Scheduler produced {len(vehicles)} vehicle(s):')
    for vid, veh_trips in enumerate(vehicles):
        n_arcs = sum(len(t) for t in veh_trips)
        lines.append(f'       Vehicle {vid+1}: {len(veh_trips)} trip(s), '
                     f'{n_arcs} arcs')
        for tid, trip in enumerate(veh_trips):
            lines.append(f'         trip {tid+1}: {list(trip)}')

    rows2 = []
    for vid, veh_trips in enumerate(vehicles):
        offset = 0.0
        for tid, trip in enumerate(veh_trips):
            comps, dur = trip_times(trip, adj, svc)
            for pos, (arc, t) in enumerate(zip(trip, comps)):
                prev = trip[pos - 1] if pos > 0 else 0
                rows2.append(
                    f'  {vid+1:>3}  {tid+1:>3}  {pos+1:>3}  {arc:>3}  '
                    f'  {int(clss[arc])}  {adj[prev,arc]:>7.4f}  '
                    f'{svc[arc]:>7.4f}  {offset+t:>9.4f}')
            ret = float(adj[trip[-1], 0]) if len(trip) else 0.0
            rows2.append(f'  {"":3}  {"":3}  {"":3}  {"":3}  '
                         f'     dead_ret={ret:.4f}  dur={dur:.4f}'
                         f'  offset->{offset+dur:.4f}')
            offset += dur
        rows2.append('')
    lines.extend(timing_table(rows2, '  Per-arc timing (Scheduler vehicles):'))

    diff_max = float(np.abs(Ts_ls - T_saved).max())
    if diff_max > EPS:
        lines.append(
            f'  {INFO} Saved routes T vs Scheduler T differ by up to '
            f'{diff_max:.4f}. Scheduler._build_P re-partitions arcs by '
            f'class-balanced chunks, ignoring saved vehicle boundaries.')

    return ok, lines


# ─────────────────────────────────────────────────────────────────────────────
# main report
# ─────────────────────────────────────────────────────────────────────────────

def run(instance_path, solution_path):
    es, adj, svc, clss, demands = load_instance(instance_path)
    routes, T_saved, variant, orig_instance, M = load_solution(solution_path)
    req  = es['req']

    node_dms = build_node_dms(es)

    lines = []
    lines.append('=' * 64)
    lines.append('  SOLUTION VALIDATION')
    lines.append(f'  Instance : {instance_path}')
    lines.append(f'  Solution : {solution_path}')
    lines.append(f'  (originally solved from: {orig_instance})')
    lines.append(f'  P={es["P"]} classes | M={M} vehicles | '
                 f'{len(req)} required arcs | variant={variant}')
    lines.append(f'  Saved T = ({T_saved[0]:.4f}, {T_saved[1]:.4f}, {T_saved[2]:.4f})')
    lines.append('=' * 64)
    lines.append('')

    all_ok = True

    def section(title, ok, msgs):
        nonlocal all_ok
        all_ok &= ok
        lines.append(title)
        if isinstance(msgs, list):
            lines.extend(msgs)
        else:
            lines.append('  ' + msgs)
        lines.append('')

    ok, msg  = check_coverage(routes, len(req))
    section('1. Coverage:', ok, msg)

    ok, msgs = check_capacity(routes, demands)
    section('2. Capacity:', ok, msgs)

    ok, msgs = check_sentinels(routes)
    section('3. Depot sentinels:', ok, msgs)

    ok, msgs = check_class_order(routes, clss, variant)
    section(f'4. Class ordering (variant={variant}):', ok, msgs)

    ok, msg  = check_deadhead(adj, node_dms, req)
    section('5. Deadhead adj matrix:', ok, msg)

    ok, msg  = check_service_times(svc, req)
    section('6. Service times:', ok, msg)

    ok, msgs = check_timing(routes, adj, svc, clss, demands, T_saved, variant, M)
    section('7. Timing:', ok, msgs)

    lines.append('=' * 64)
    tag = PASS if all_ok else FAIL
    lines.append(f'  {tag} Overall: '
                 f'{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    lines.append('=' * 64)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True,
                        help='Problem instance .npz file')
    parser.add_argument('--solution', required=True,
                        help='Solution .npz file saved by ils_log.py')
    parser.add_argument('--log',      default=None,
                        help='Write report to this file (default: stdout only)')
    args = parser.parse_args()

    report = run(args.instance, args.solution)
    print(report)

    if args.log:
        os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
        with open(args.log, 'w') as fh:
            fh.write(report + '\n')
        print(f'\nSaved -> {args.log}')


if __name__ == '__main__':
    main()
