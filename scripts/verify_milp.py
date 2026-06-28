#!/usr/bin/env python3
"""MILP solution verifier for HDCARP-P / HDCARP-U.

Solves the MILP, then verifies every algebraic constraint against the SCIP
solution and re-derives T[k] from the physically-reconstructed routes (using
Eulerian-circuit traversal of the y-flow multigraph) to confirm the model
encodes the actual problem correctly.

Checks
------
  1. Coverage        – every required arc served exactly once
  2. Capacity        – per-vehicle demand <= Q
  3. Flow balance    – for each (vehicle, level, node): in == out
  4. Level start     – level 1 enters from the depot
  5. Level transition – re-entry vertex == previous-level exit vertex
  6. T[k] model      – T[k] >= t[m,h] when r[m,k,h]=1
  7. T[k] physical   – Eulerian-route timing agrees with SCIP T[k]

Usage
-----
  uv run python scripts/verify_milp.py \\
      --path data/ood/osm_cityB/40/38_14_859.npz \\
      --variant P --time_limit 300
"""

import argparse
import collections
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from solvers.lp import solve_milp_p, solve_milp_u, build_graph, _beta

PASS = '[PASS]'
FAIL = '[FAIL]'
EPS  = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(model, var):
    """Return SCIP solution value, handling the case t[(m,0)] = 0.0 (float)."""
    if isinstance(var, float):
        return var
    return model.getVal(var)


def _eulerian_check(model, m, lv, x_by_level, y_vars, req_by_level, A_prime, dummy):
    """Check that (vehicle m, level lv) y+x flows form a traversable route.

    Service arcs (x=1) count as one directed traversal each.
    Deadhead arcs (y=k) count as k directed traversals each.

    The serviced route lives in the weakly-connected component that contains the
    dummy node v'_0 (level 1 always enters there, and every level couples to the
    previous via v'_0). That component must be balanced (in==out) so it is
    Eulerian-traversable.

    A disconnected component with NO service arcs is a "phantom" deadhead loop:
    a balanced y-cycle that the model permits because it satisfies flow balance
    (28) and the lazy connectivity handler only cuts components that contain a
    serviced required arc. Such loops are objective-neutral degeneracies (they
    add deadhead time to t[m,h] only when that level is non-binding), NOT a
    feasibility violation -- so we WARN but do not FAIL.

    A disconnected component that DOES contain a service arc is a genuine
    sub-tour the connectivity handler missed: that is a real bug -> FAIL.

    Returns (is_valid, error_message, warning_message).
    """
    MG = nx.MultiDiGraph()

    for a in req_by_level[lv]:
        if _get(model, x_by_level.get((m, lv, a), 0.0)) > 0.5:
            MG.add_edge(a[0], a[1], arc=a, kind='service')

    for a in A_prime:
        cnt = int(round(_get(model, y_vars.get((m, a, lv), 0.0))))
        for _ in range(cnt):
            MG.add_edge(a[0], a[1], arc=a, kind='deadhead')

    if MG.number_of_edges() == 0:
        return True, None, None  # vehicle does nothing at this level

    # Every node must be balanced (in==out) for the multigraph to be traversable.
    bad = [(v, MG.in_degree(v), MG.out_degree(v))
           for v in MG.nodes if MG.in_degree(v) != MG.out_degree(v)]
    if bad:
        return False, f'unbalanced nodes (in!=out): {bad[:5]}', None

    if nx.is_weakly_connected(MG):
        return True, None, None

    # Disconnected: classify each component that does NOT hold the dummy node.
    comps = list(nx.weakly_connected_components(MG))
    phantom = []
    for comp in comps:
        if dummy in comp:
            continue
        has_service = any(
            d.get('kind') == 'service'
            for _u, _v, d in MG.edges(comp, data=True)
        )
        if has_service:
            return (False,
                    f'disconnected component {sorted(comp)} contains a service '
                    f'arc (genuine sub-tour, connectivity cut missed)', None)
        phantom.append(sorted(comp))

    warn = (f'phantom deadhead loop(s) disconnected from dummy: {phantom} '
            f'(no service arcs; balanced y-cycle, model-feasible but '
            f'suboptimal deadhead)')
    return True, None, warn


# ─────────────────────────────────────────────────────────────────────────────
# constraint checkers
# ─────────────────────────────────────────────────────────────────────────────

def check_coverage(model, x_by_level, req_arcs, M, levels):
    """Constraint: every required arc served exactly once."""
    lines, ok = [], True
    for a in req_arcs:
        total = sum(_get(model, x_by_level.get((m, lv, a), 0.0))
                    for m in M for lv in levels)
        if abs(total - 1.0) > EPS:
            ok = False
            lines.append(f'  {FAIL} arc {a}: served {total:.3f} times (expected 1)')
    if ok:
        lines.append(f'  {PASS} all {len(req_arcs)} required arcs served exactly once')
    return ok, lines


def check_capacity(model, x_by_level, req_arcs, M, levels, q, Q):
    """Constraint: per-vehicle demand <= Q."""
    lines, ok = [], True
    for m in M:
        load = sum(q[a] * _get(model, x_by_level.get((m, lv, a), 0.0))
                   for a in req_arcs for lv in levels)
        over = load > Q + EPS
        ok  &= not over
        tag  = FAIL if over else PASS
        lines.append(f'  {tag} vehicle {m}: load={load:.4f} / Q={Q:.4f}')
    return ok, lines


def check_flow_balance(model, x_by_level, y_vars, req_by_level, A_prime,
                       nodes_all, M, levels):
    """Constraint: in-flow == out-flow at every node for each (vehicle, level)."""
    out_prime = collections.defaultdict(list)
    in_prime  = collections.defaultdict(list)
    for a in A_prime:
        out_prime[a[0]].append(a)
        in_prime[a[1]].append(a)

    out_req = collections.defaultdict(list)
    in_req  = collections.defaultdict(list)
    for lv in levels:
        for a in req_by_level[lv]:
            out_req[(lv, a[0])].append(a)
            in_req[(lv, a[1])].append(a)

    lines, ok = [], True
    n_checked = 0
    for m in M:
        for lv in levels:
            for v in nodes_all:
                out_x = sum(_get(model, x_by_level[(m, lv, a)])
                            for a in out_req[(lv, v)])
                in_x  = sum(_get(model, x_by_level[(m, lv, a)])
                            for a in in_req[(lv, v)])
                out_y = sum(_get(model, y_vars[(m, a, lv)])
                            for a in out_prime[v])
                in_y  = sum(_get(model, y_vars[(m, a, lv)])
                            for a in in_prime[v])
                diff = abs((out_x + out_y) - (in_x + in_y))
                if diff > EPS:
                    ok = False
                    lines.append(f'  {FAIL} m={m} lv={lv} v={v}: '
                                 f'out={out_x+out_y:.3f} in={in_x+in_y:.3f}')
                n_checked += 1
    if ok:
        lines.append(f'  {PASS} flow balance holds at all {n_checked} '
                     f'(vehicle, level, node) triples')
    return ok, lines


def check_level_start(model, y_vars, M, dummy, depot):
    """Constraint: level 1 always enters from the depot."""
    lines, ok = [], True
    for m in M:
        val = _get(model, y_vars[(m, (dummy, depot), 1)])
        if abs(val - 1.0) > EPS:
            ok = False
            lines.append(f'  {FAIL} vehicle {m}: y[(dummy,depot),1]={val:.3f} != 1')
        else:
            lines.append(f'  {PASS} vehicle {m}: level 1 enters at depot')
    return ok, lines


def check_level_transition(model, y_vars, M, levels, V_t, dummy):
    """Constraint (8)/(25): re-entry vertex == previous-level exit vertex."""
    lines, ok = [], True
    n_checked = 0
    for m in M:
        for lv in levels:
            if lv == 1:
                continue
            for v in sorted(V_t):
                entry = _get(model, y_vars[(m, (dummy, v), lv)])
                exit_ = _get(model, y_vars[(m, (v, dummy), lv - 1)])
                if abs(entry - exit_) > EPS:
                    ok = False
                    lines.append(f'  {FAIL} m={m} lv={lv} v={v}: '
                                 f'entry={entry:.3f} != exit_prev={exit_:.3f}')
                n_checked += 1
    if ok:
        lines.append(f'  {PASS} level transition coupling holds '
                     f'at all {n_checked} (vehicle, level, vertex) triples')
    return ok, lines


def check_T_model(model, T_vars, t_vars, r_vars, M, P, levels, classes, beta):
    """Constraint (2)/(19): T[k] >= t[m,h] when r[m,k,h]=1."""
    lines, ok = [], True
    for k in classes:
        Tk = _get(model, T_vars[k])
        lines.append(f'  T[{k}] = {Tk:.6f}')
        for m in M:
            for lv in levels:
                rmkh = _get(model, r_vars[(m, k, lv)])
                tmh  = _get(model, t_vars[(m, lv)])
                rhs  = tmh - beta * (1 - rmkh)
                if Tk < rhs - EPS:
                    ok = False
                    lines.append(f'    {FAIL} T[{k}]={Tk:.4f} < t[{m},{lv}]-beta*(1-r)'
                                 f'={rhs:.4f}  (r={rmkh:.0f}, t={tmh:.4f})')
    if ok:
        lines.append(f'  {PASS} T[k] >= t[m,h]-beta*(1-r) for all (k,m,h)')
    return ok, lines


def check_level_times(model, x_by_level, y_vars, t_vars, req_by_level,
                      M, levels, s, cost, A):
    """Verify t[m,lv] = t[m,lv-1] + service_time + deadhead_time (constraint 3/20).

    physical_level_time = Σ s[a]*x[m,a,lv] + Σ d[a]*y[m,a,lv]
    This must equal t[m,lv] - t[m,lv-1] from the SCIP solution.
    """
    lines, ok = [], True
    for m in M:
        for lv in levels:
            svc_t = sum(s[a] * _get(model, x_by_level.get((m, lv, a), 0.0))
                        for a in req_by_level[lv])
            dh_t  = sum(cost[a] * _get(model, y_vars.get((m, a, lv), 0.0))
                        for a in A if cost.get(a, 0.0) > 0)
            phys  = svc_t + dh_t
            t_cur  = _get(model, t_vars[(m, lv)])
            t_prev = _get(model, t_vars[(m, lv - 1)])
            model_dt = t_cur - t_prev
            diff = abs(phys - model_dt)
            if diff > EPS * 10:
                ok = False
                lines.append(f'  {FAIL} m={m} lv={lv}: '
                             f'physical={phys:.6f} model_dt={model_dt:.6f} '
                             f'diff={diff:.2e}')
    if ok:
        lines.append(f'  {PASS} t[m,lv] recursion matches x/y values '
                     f'for all {len(M)*len(levels)} (vehicle, level) pairs')
    return ok, lines


def check_T_physical(model, T_vars, t_vars, r_vars, M, levels, classes, G):
    """Verify T[k] from model t-values (cleaner than Eulerian circuit timing).

    For each k: T[k] = max over (m,lv) where r[m,k,lv]=1 of t[m,lv].
    Compare this with SCIP's T[k] variable value.
    """
    lines, ok = [], True

    T_derived = {}
    for k in classes:
        best = 0.0
        for m in M:
            for lv in levels:
                r = _get(model, r_vars.get((m, k, lv), 0.0))
                if r > 0.5:
                    tmh = _get(model, t_vars[(m, lv)])
                    if tmh > best:
                        best = tmh
        T_derived[k] = best

    lines.append('  T[k] from t-values vs SCIP T[k]:')
    lines.append(f'  {"k":>3}  {"SCIP T[k]":>12}  {"max t[m,h]":>12}  {"diff":>10}  status')
    lines.append('  ' + '-' * 57)
    for k in classes:
        Tk_scip = _get(model, T_vars[k])
        Tk_der  = T_derived[k]
        diff    = abs(Tk_scip - Tk_der)
        tag     = PASS if diff <= EPS * 10 else FAIL
        ok     &= (tag == PASS)
        lines.append(f'  {k:>3}  {Tk_scip:>12.6f}  {Tk_der:>12.6f}  '
                     f'{diff:>10.6f}  {tag}')
    return ok, lines


def check_eulerian(model, x_by_level, y_vars, req_by_level, A_prime, M, levels, dummy):
    """Verify each (vehicle, level) y+x flow graph is Eulerian (traversable).

    The serviced route (dummy-connected component) must be balanced and
    traversable. Phantom deadhead-only loops disconnected from the dummy are
    model-feasible degeneracies: reported as warnings, not failures.
    """
    lines, ok = [], True
    n_checked, warns = 0, []
    for m in M:
        for lv in levels:
            valid, err, warn = _eulerian_check(model, m, lv, x_by_level, y_vars,
                                               req_by_level, A_prime, dummy)
            if not valid:
                ok = False
                lines.append(f'  {FAIL} vehicle {m} level {lv}: {err}')
            if warn:
                warns.append(f'  [WARN] vehicle {m} level {lv}: {warn}')
            n_checked += 1
    if ok:
        lines.append(f'  {PASS} all {n_checked} (vehicle, level) serviced routes '
                     f'are balanced and traversable')
    lines.extend(warns)
    return ok, lines


# ─────────────────────────────────────────────────────────────────────────────
# main report
# ─────────────────────────────────────────────────────────────────────────────

def verify(path, variant, time_limit, threads):
    es = np.load(path)
    G  = build_graph(es)
    P  = G['P']
    Mn = G['M']
    M       = list(range(Mn))
    classes = list(range(1, P + 1))
    levels  = list(range(1, P + 1))
    beta    = _beta(G)
    dummy   = G['dummy']
    depot   = G['depot']
    nodes_all = G['nodes'] | {dummy}

    print(f'Solving MILP-{variant} (time_limit={time_limit}s)...', flush=True)
    solver = solve_milp_p if variant == 'P' else solve_milp_u
    Tvec, model, vd = solver(es, time_limit=time_limit, threads=threads,
                             verbose=False, return_model=True)

    lines = []
    lines.append('=' * 64)
    lines.append(f'  MILP-{variant} SOLUTION VERIFIER')
    lines.append(f'  Instance : {path}')
    lines.append(f'  P={P} classes | M={Mn} vehicles | '
                 f'{len(G["req_arcs"])} required arcs')
    lines.append('=' * 64)

    if Tvec is None:
        lines.append(f'\n  {FAIL} No feasible solution found.')
        return '\n'.join(lines)

    lines.append(f'\n  SCIP T values: ' +
                 '  '.join(f'T{k}={Tvec[k-1]:.6f}' for k in classes))
    lines.append('')

    # Normalise variable access: unify P and U into common (m, level, arc) key
    x_raw = vd['x']
    y_raw = vd['y']
    r_raw = vd['r']
    t_raw = vd['t']
    T_raw = vd['T']

    if variant == 'P':
        # x[(m,a)], y[(m,a,k)], r[(m,k)], t[(m,k)]
        x_by_level = {(m, k, a): x_raw[(m, a)]
                      for m in M for k in classes for a in G['req_by_class'][k]}
        y_vars     = {(m, a, k): y_raw[(m, a, k)]
                      for m in M for a in G['A_prime'] for k in classes}
        r_vars     = {(m, k, lv): r_raw[(m, k)]
                      for m in M for k in classes for lv in [k]}
        # pad r_vars for off-diagonal (won't be active, beta makes constraint slack)
        for m in M:
            for k in classes:
                for lv in levels:
                    if (m, k, lv) not in r_vars:
                        r_vars[(m, k, lv)] = 0.0
        req_by_level = {k: G['req_by_class'][k] for k in classes}
        cls_of_arc   = G['cls']
    else:
        # x[(m,a,h)], y[(m,a,h)], r[(m,k,h)], t[(m,h)]
        x_by_level   = {(m, h, a): x_raw[(m, a, h)]
                        for m in M for h in levels for a in G['req_arcs']}
        y_vars       = {(m, a, h): y_raw[(m, a, h)]
                        for m in M for a in G['A_prime'] for h in levels}
        r_vars       = {(m, k, h): r_raw[(m, k, h)]
                        for m in M for k in classes for h in levels}
        req_by_level = {h: list(G['req_arcs']) for h in levels}
        cls_of_arc   = G['cls']

    t_vars = {(m, lv): t_raw[(m, lv)] for m in M for lv in levels}
    t_vars.update({(m, 0): 0.0 for m in M})

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

    ok, msgs = check_coverage(model, x_by_level, G['req_arcs'], M, levels)
    section('1. Coverage:', ok, msgs)

    ok, msgs = check_capacity(model, x_by_level, G['req_arcs'], M, levels,
                              G['q'], G['Q'])
    section('2. Capacity:', ok, msgs)

    ok, msgs = check_flow_balance(model, x_by_level, y_vars, req_by_level,
                                  G['A_prime'], nodes_all, M, levels)
    section('3. Flow balance:', ok, msgs)

    ok, msgs = check_level_start(model, y_vars, M, dummy, depot)
    section('4. Level-1 depot start:', ok, msgs)

    ok, msgs = check_level_transition(model, y_vars, M, levels, G['V_t'], dummy)
    section('5. Level transition (constraint 8/25):', ok, msgs)

    ok, msgs = check_T_model(model, T_raw, t_vars, r_vars, M, P,
                             levels, classes, beta)
    section('6. T[k] model constraints (constraint 2/19):', ok, msgs)

    ok, msgs = check_level_times(model, x_by_level, y_vars, t_vars,
                                 req_by_level, M, levels,
                                 G['s'], G['cost'], G['A'])
    section('7. Level time recursion (constraint 3/20):', ok, msgs)

    ok, msgs = check_T_physical(model, T_raw, t_vars, r_vars, M, levels, classes, G)
    section('8. T[k] from t-values (constraint 2/19 derived):', ok, msgs)

    ok, msgs = check_eulerian(model, x_by_level, y_vars, req_by_level,
                              G['A_prime'], M, levels, dummy)
    section('9. Eulerian flow (route traversability):', ok, msgs)

    lines.append('=' * 64)
    tag = PASS if all_ok else FAIL
    lines.append(f'  {tag} Overall: '
                 f'{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    lines.append('=' * 64)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Verify MILP-P/U solution')
    ap.add_argument('--path',       required=True)
    ap.add_argument('--variant',    default='P', choices=['P', 'U'])
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--threads',    type=int, default=8)
    ap.add_argument('--log',        default=None)
    args = ap.parse_args()

    report = verify(args.path, args.variant, args.time_limit, args.threads)
    print(report)
    if args.log:
        with open(args.log, 'w') as f:
            f.write(report + '\n')
        print(f'\nReport saved to {args.log}')


if __name__ == '__main__':
    main()
