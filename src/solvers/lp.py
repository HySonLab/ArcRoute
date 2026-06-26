"""MILP solver for HDCARP-P and HDCARP-U.

Paper: Ha et al. (2024) "On the Hierarchical Directed Capacitated Arc Routing
Problem" (arXiv:2501.00852).
Solver: PySCIPOpt 6.2.1 (SCIP 9.2).

Two exact models are implemented:
  * MILP-P (precedence variant, Section 3.3): the class processed at hierarchy
    level k is fixed to class k -> variables x[m,a], y[m,a,k] indexed by class.
  * MILP-U (upgrading variant, Section 3.4): the class processed at a hierarchy
    level is a decision -> variables x[m,a,h], y[m,a,h] indexed by level, with
    r[m,k,h] linking levels to classes.

Both share:
  * the graph transformation of Section 3.2 (a dummy node v'_0 lets a vehicle
    thread between hierarchy levels at zero cost while staying physically put),
  * a lexicographic objective (T_1 >> T_2 >> ... >> T_p) encoded as a weighted
    sum with geometric weights,
  * connectivity / sub-tour elimination added lazily through a custom
    ``Conshdlr`` (constraints (12) / (29)).

The model works in the raw (un-normalised) time/demand units stored in the
``.npz`` file: demand q_a and capacity Q come straight from ``es['req']`` /
``es['C']``; service s_a and traversal d_a from the last two columns.
"""

import argparse
from time import time

import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT


# --------------------------------------------------------------------------- #
# Graph transformation (Section 3.2)
# --------------------------------------------------------------------------- #
def build_graph(es):
    """Build every component of the transformed multigraph G' = (V', A').

    ``es`` is a path to a ``.npz`` instance or an already-loaded npz mapping.

    Returns a dict with:
      depot          int, the depot v_0 (always node 0)
      dummy          int, the dummy node v'_0 (= max real node + 1)
      nodes          set of real node ids (excludes the dummy)
      req_arcs       list of (tail, head) tuples for required arcs A_r
      q/s/cls        dict arc -> demand / service / class   (required arcs)
      req_by_class   dict class -> list of required arcs of that class
      A              list of (tail, head) for all original arcs (req + nonreq)
      A_f            list of (dummy, v) arcs, traversal 0
      A_t            list of (v, dummy) arcs, traversal 0
      A_prime        list of (tail, head) for A union A_f union A_t
      cost           dict arc -> traversal time over A_prime (A_f/A_t -> 0)
      V_t_k          dict class -> set of tail vertices of that class
      V_t            set, union of all V_t_k
      node_sp        dict (u, v) -> node-to-node shortest path time (Floyd-Warshall)
      P              int, number of priority classes
      M              int, number of vehicles
      Q              float, vehicle capacity
    """
    if isinstance(es, str):
        es = np.load(es)
    req = np.asarray(es['req'], dtype=np.float64)
    nonreq = np.asarray(es['nonreq'], dtype=np.float64)
    P = int(es['P'])
    M = int(es['M'])
    Q = float(es['C'])

    depot = 0
    # ----- required arcs -------------------------------------------------- #
    req_arcs = []
    q, s, cls = {}, {}, {}
    req_by_class = {k: [] for k in range(1, P + 1)}
    V_t_k = {k: set() for k in range(1, P + 1)}
    for tail, head, demand, clss, service, traversal in req:
        a = (int(tail), int(head))
        req_arcs.append(a)
        q[a] = float(demand)
        s[a] = float(service)
        cls[a] = int(clss)
        req_by_class[int(clss)].append(a)
        V_t_k[int(clss)].add(int(tail))
    V_t = set().union(*V_t_k.values()) if V_t_k else set()
    V_t = V_t | {depot}  # depot is a service-relevant vertex too

    # ----- all original arcs A ------------------------------------------- #
    cost = {}
    A = []
    nodes = set()
    for tail, head, _d, _c, _s, traversal in np.vstack([req, nonreq]):
        a = (int(tail), int(head))
        A.append(a)
        cost[a] = float(traversal)
        nodes.add(int(tail))
        nodes.add(int(head))

    # ----- dummy node + transition arcs A_f / A_t ------------------------ #
    dummy = max(nodes) + 1
    Vt_plus = sorted(V_t)  # V_t already includes the depot
    A_f = [(dummy, v) for v in Vt_plus]
    A_t = [(v, dummy) for v in Vt_plus]
    for a in A_f + A_t:
        cost[a] = 0.0
    A_prime = list(A) + A_f + A_t

    # ----- node-to-node shortest paths (informational / bounds) ---------- #
    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    INF = np.inf
    D = np.full((n, n), INF)
    np.fill_diagonal(D, 0.0)
    for (t, h), c in zip(A, [cost[a] for a in A]):
        D[idx[t], idx[h]] = min(D[idx[t], idx[h]], c)
    for k in range(n):
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    node_sp = {}
    for u in node_list:
        for v in node_list:
            d = D[idx[u], idx[v]]
            if np.isfinite(d):
                node_sp[(u, v)] = float(d)

    return {
        'depot': depot, 'dummy': dummy, 'nodes': nodes,
        'req_arcs': req_arcs, 'q': q, 's': s, 'cls': cls,
        'req_by_class': req_by_class,
        'A': A, 'A_f': A_f, 'A_t': A_t, 'A_prime': A_prime, 'cost': cost,
        'V_t_k': V_t_k, 'V_t': V_t, 'node_sp': node_sp,
        'P': P, 'M': M, 'Q': Q,
    }


# --------------------------------------------------------------------------- #
# Lazy connectivity / sub-tour elimination (constraints (12) / (29))
# --------------------------------------------------------------------------- #
class ConnectivityHandler(Conshdlr):
    """Adds connectivity cuts lazily.

    For every (vehicle m, hierarchy level L) the support graph of the current
    solution (serviced required arcs + active deadheads) must be connected to
    the dummy node v'_0. Any weakly-connected component S that does not contain
    v'_0 but contains a serviced required arc b is a sub-tour; we add

        sum_{a in delta^+(S)} x/y  >=  x[m, b]

    forcing flow to leave S (constraint (12)/(29)).
    """

    def __init__(self, M, levels, x_lvl, y, req_by_level, A_prime, dummy):
        self.M = M                      # list of vehicle ids
        self.levels = levels            # list of hierarchy levels (1..p)
        self.x_lvl = x_lvl              # dict (m, level, arc) -> var
        self.y = y                      # dict (m, level, arc) -> var
        self.req_by_level = req_by_level  # dict level -> list of req arcs
        self.A_prime = A_prime          # list of arcs in A'
        self.dummy = dummy

    # -- detection ---------------------------------------------------------- #
    def _violated(self, sol):
        """Return list of (m, level, S, b) sub-tour violations for solution sol."""
        model = self.model
        viol = []
        for m in self.M:
            for L in self.levels:
                serviced = [a for a in self.req_by_level[L]
                            if model.getSolVal(sol, self.x_lvl[(m, L, a)]) > 0.5]
                if not serviced:
                    continue
                deadheads = [a for a in self.A_prime
                             if model.getSolVal(sol, self.y[(m, L, a)]) > 0.5]
                g = nx.DiGraph()
                g.add_edges_from(serviced)
                g.add_edges_from(deadheads)
                for comp in nx.weakly_connected_components(g):
                    if self.dummy in comp:
                        continue
                    inside = [a for a in serviced
                              if a[0] in comp and a[1] in comp]
                    if inside:
                        viol.append((m, L, frozenset(comp), inside))
        return viol

    def _add_cut(self, m, L, S, b):
        leaving = [self.x_lvl[(m, L, a)] for a in self.req_by_level[L]
                   if a[0] in S and a[1] not in S]
        leaving += [self.y[(m, L, a)] for a in self.A_prime
                    if a[0] in S and a[1] not in S]
        self.model.addCons(quicksum(leaving) >= self.x_lvl[(m, L, b)])

    def _enforce(self, sol):
        viol = self._violated(sol)
        if not viol:
            return {"result": SCIP_RESULT.FEASIBLE}
        for m, L, S, inside in viol:
            for b in inside:
                self._add_cut(m, L, S, b)
        return {"result": SCIP_RESULT.CONSADDED}

    # -- Conshdlr callbacks ------------------------------------------------- #
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        return self._enforce(None)

    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        return self._enforce(None)

    def conscheck(self, constraints, solution, checkintegrality, checklprows,
                  printreason, completely):
        if self._violated(solution):
            return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


# --------------------------------------------------------------------------- #
# Shared model scaffolding
# --------------------------------------------------------------------------- #
def _beta(G):
    """Valid upper bound on a single route's duration."""
    return sum(G['s'].values()) + sum(G['cost'][a] for a in G['A']) + 1.0


def _obj_weights(P):
    """Geometric weights encoding the lexicographic order T_1 >> ... >> T_p."""
    return {k: 10.0 ** (3 * (P - k)) for k in range(1, P + 1)}


def _arc_groups(G):
    """Precompute out/in adjacency over A' and the required-arc adjacency."""
    nodes_all = G['nodes'] | {G['dummy']}
    out_prime = {v: [] for v in nodes_all}
    in_prime = {v: [] for v in nodes_all}
    for a in G['A_prime']:
        out_prime[a[0]].append(a)
        in_prime[a[1]].append(a)
    return nodes_all, out_prime, in_prime


def _configure(model, time_limit, threads, verbose):
    if not verbose:
        model.hideOutput()
    model.setParam('limits/time', time_limit)
    try:
        model.setParam('parallel/maxnthreads', threads)
    except Exception:
        pass


def _extract_T(model, T, P):
    """Return np.array([T_1, ..., T_p]) if a solution exists, else None."""
    if model.getNSols() == 0:
        return None
    return np.array([model.getVal(T[k]) for k in range(1, P + 1)],
                    dtype=np.float64)


# --------------------------------------------------------------------------- #
# MILP-P (Section 3.3)
# --------------------------------------------------------------------------- #
def solve_milp_p(es, time_limit=3600, threads=8, verbose=False,
                 return_model=False):
    G = build_graph(es)
    P, Mn, Q = G['P'], G['M'], G['Q']
    M = list(range(Mn))
    classes = list(range(1, P + 1))
    beta = _beta(G)
    w = _obj_weights(P)
    nodes_all, out_prime, in_prime = _arc_groups(G)
    depot, dummy = G['depot'], G['dummy']

    # required-arc adjacency, by class
    out_reqk = {(k, v): [] for k in classes for v in nodes_all}
    in_reqk = {(k, v): [] for k in classes for v in nodes_all}
    for k in classes:
        for a in G['req_by_class'][k]:
            out_reqk[(k, a[0])].append(a)
            in_reqk[(k, a[1])].append(a)

    model = Model('HDCARP-P')
    _configure(model, time_limit, threads, verbose)

    # --- variables -------------------------------------------------------- #
    x = {(m, a): model.addVar(f'x_{m}_{a}', vtype='B')
         for m in M for a in G['req_arcs']}
    y = {}
    for m in M:
        for k in classes:
            for a in G['A_prime']:
                ub = 1 if (a[0] == dummy or a[1] == dummy) else len(G['nodes'])
                y[(m, a, k)] = model.addVar(f'y_{m}_{a}_{k}', vtype='I',
                                            lb=0, ub=ub)
    t = {(m, 0): 0.0 for m in M}
    for m in M:
        for k in classes:
            t[(m, k)] = model.addVar(f't_{m}_{k}', vtype='C', lb=0)
    r = {(m, k): model.addVar(f'r_{m}_{k}', vtype='B') for m in M for k in classes}
    T = {k: model.addVar(f'T_{k}', vtype='C', lb=0) for k in classes}

    # --- objective -------------------------------------------------------- #
    model.setObjective(quicksum(w[k] * T[k] for k in classes), 'minimize')

    # --- constraints ------------------------------------------------------ #
    for m in M:
        for k in classes:
            # (2) T_k >= t_{m,k} - beta (1 - r_{m,k})
            model.addCons(T[k] >= t[(m, k)] - beta * (1 - r[(m, k)]))
            # (3) completion-time recursion
            svc = quicksum(G['s'][a] * x[(m, a)] for a in G['req_by_class'][k])
            dh = quicksum(G['cost'][a] * y[(m, a, k)] for a in G['A']
                          if G['cost'][a] > 0)
            model.addCons(t[(m, k)] == t[(m, k - 1)] + svc + dh)
            # (5) r links service
            nk = len(G['req_by_class'][k])
            model.addCons(quicksum(x[(m, a)] for a in G['req_by_class'][k])
                          <= nk * r[(m, k)])
            # (7) exactly one A_f arc per (m, k)
            model.addCons(quicksum(y[(m, a, k)] for a in G['A_f']) == 1)
        # (6) class 1 starts at the depot
        model.addCons(y[(m, (dummy, depot), 1)] == 1)
        # (8) link level entry to previous-level exit
        for k in classes:
            if k == 1:
                continue
            prev_tails = set().union(*[G['V_t_k'][kp] for kp in range(1, k)])
            prev_tails = prev_tails | {depot}
            for v in prev_tails:
                model.addCons(y[(m, (dummy, v), k)] == y[(m, (v, dummy), k - 1)])
        # (10) capacity
        model.addCons(quicksum(G['q'][a] * x[(m, a)] for a in G['req_arcs'])
                      <= Q)
        # (11) flow balance per class, per node (incl. dummy)
        for k in classes:
            for v in nodes_all:
                out_x = quicksum(x[(m, a)] for a in out_reqk[(k, v)])
                out_y = quicksum(y[(m, a, k)] for a in out_prime[v])
                in_x = quicksum(x[(m, a)] for a in in_reqk[(k, v)])
                in_y = quicksum(y[(m, a, k)] for a in in_prime[v])
                model.addCons(out_x + out_y == in_x + in_y)

    # (9) every required arc served exactly once
    for a in G['req_arcs']:
        model.addCons(quicksum(x[(m, a)] for m in M) == 1)

    # --- lazy connectivity ------------------------------------------------ #
    x_lvl = {(m, k, a): x[(m, a)]
             for m in M for k in classes for a in G['req_by_class'][k]}
    y_lvl = {(m, k, a): y[(m, a, k)]
             for m in M for k in classes for a in G['A_prime']}
    handler = ConnectivityHandler(M, classes, x_lvl, y_lvl,
                                  G['req_by_class'], G['A_prime'], dummy)
    model.includeConshdlr(handler, 'connectivity_P',
                          'lazy sub-tour elimination (MILP-P)',
                          sepapriority=-1, enfopriority=-1, chckpriority=-1,
                          sepafreq=-1, propfreq=-1, eagerfreq=-1,
                          needscons=False)

    model.optimize()
    Tvec = _extract_T(model, T, P)
    if return_model:
        return Tvec, model, {'x': x, 'y': y, 'r': r, 't': t, 'T': T, 'G': G,
                             'M': M, 'levels': classes}
    return Tvec


# --------------------------------------------------------------------------- #
# MILP-U (Section 3.4)
# --------------------------------------------------------------------------- #
def solve_milp_u(es, time_limit=3600, threads=8, verbose=False,
                 return_model=False):
    G = build_graph(es)
    P, Mn, Q = G['P'], G['M'], G['Q']
    M = list(range(Mn))
    classes = list(range(1, P + 1))
    levels = list(range(1, P + 1))
    beta = _beta(G)
    w = _obj_weights(P)
    nodes_all, out_prime, in_prime = _arc_groups(G)
    depot, dummy = G['depot'], G['dummy']

    # required-arc adjacency (all classes -> level h services any class)
    out_req = {v: [] for v in nodes_all}
    in_req = {v: [] for v in nodes_all}
    for a in G['req_arcs']:
        out_req[a[0]].append(a)
        in_req[a[1]].append(a)

    model = Model('HDCARP-U')
    _configure(model, time_limit, threads, verbose)

    # --- variables -------------------------------------------------------- #
    x = {(m, a, h): model.addVar(f'x_{m}_{a}_{h}', vtype='B')
         for m in M for a in G['req_arcs'] for h in levels}
    y = {}
    for m in M:
        for h in levels:
            for a in G['A_prime']:
                ub = 1 if (a[0] == dummy or a[1] == dummy) else len(G['nodes'])
                y[(m, a, h)] = model.addVar(f'y_{m}_{a}_{h}', vtype='I',
                                            lb=0, ub=ub)
    t = {(m, 0): 0.0 for m in M}
    for m in M:
        for h in levels:
            t[(m, h)] = model.addVar(f't_{m}_{h}', vtype='C', lb=0)
    r = {(m, k, h): model.addVar(f'r_{m}_{k}_{h}', vtype='B')
         for m in M for k in classes for h in levels}
    T = {k: model.addVar(f'T_{k}', vtype='C', lb=0) for k in classes}

    # --- objective -------------------------------------------------------- #
    model.setObjective(quicksum(w[k] * T[k] for k in classes), 'minimize')

    # --- constraints ------------------------------------------------------ #
    for m in M:
        for h in levels:
            # (20) completion-time recursion
            svc = quicksum(G['s'][a] * x[(m, a, h)] for a in G['req_arcs'])
            dh = quicksum(G['cost'][a] * y[(m, a, h)] for a in G['A']
                          if G['cost'][a] > 0)
            model.addCons(t[(m, h)] == t[(m, h - 1)] + svc + dh)
            # (24) exactly one A_f arc per (m, h)
            model.addCons(quicksum(y[(m, a, h)] for a in G['A_f']) == 1)
            for k in classes:
                # (19) T_k >= t_{m,h} - beta (1 - r_{m,k,h})
                model.addCons(T[k] >= t[(m, h)] - beta * (1 - r[(m, k, h)]))
                # (22) r links service of class k at level h
                nk = len(G['req_by_class'][k])
                model.addCons(
                    quicksum(x[(m, a, h)] for a in G['req_by_class'][k])
                    <= nk * r[(m, k, h)])
        # (23) level 1 starts at the depot
        model.addCons(y[(m, (dummy, depot), 1)] == 1)
        # (25) link level entry to previous-level exit (over all V_t U {v_0})
        for h in levels:
            if h == 1:
                continue
            for v in sorted(G['V_t']):
                model.addCons(y[(m, (dummy, v), h)] == y[(m, (v, dummy), h - 1)])
        # (27) capacity
        model.addCons(
            quicksum(G['q'][a] * x[(m, a, h)] for a in G['req_arcs']
                     for h in levels) <= Q)
        # (28) flow balance per level, per node (incl. dummy)
        for h in levels:
            for v in nodes_all:
                out_x = quicksum(x[(m, a, h)] for a in out_req[v])
                out_y = quicksum(y[(m, a, h)] for a in out_prime[v])
                in_x = quicksum(x[(m, a, h)] for a in in_req[v])
                in_y = quicksum(y[(m, a, h)] for a in in_prime[v])
                model.addCons(out_x + out_y == in_x + in_y)

    # (26) every required arc served exactly once over all (m, h)
    for a in G['req_arcs']:
        model.addCons(quicksum(x[(m, a, h)] for m in M for h in levels) == 1)

    # --- lazy connectivity ------------------------------------------------ #
    req_by_level = {h: list(G['req_arcs']) for h in levels}
    x_lvl = {(m, h, a): x[(m, a, h)]
             for m in M for h in levels for a in G['req_arcs']}
    y_lvl = {(m, h, a): y[(m, a, h)]
             for m in M for h in levels for a in G['A_prime']}
    handler = ConnectivityHandler(M, levels, x_lvl, y_lvl,
                                  req_by_level, G['A_prime'], dummy)
    model.includeConshdlr(handler, 'connectivity_U',
                          'lazy sub-tour elimination (MILP-U)',
                          sepapriority=-1, enfopriority=-1, chckpriority=-1,
                          sepafreq=-1, propfreq=-1, eagerfreq=-1,
                          needscons=False)

    model.optimize()
    Tvec = _extract_T(model, T, P)
    if return_model:
        return Tvec, model, {'x': x, 'y': y, 'r': r, 't': t, 'T': T, 'G': G,
                             'M': M, 'levels': levels}
    return Tvec


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description='MILP solver for HDCARP-P / HDCARP-U')
    p.add_argument('--variant', type=str, default='P', choices=['P', 'U'])
    p.add_argument('--path', type=str, required=True, help='path to a .npz instance')
    p.add_argument('--time_limit', type=float, default=3600)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    solve = solve_milp_p if args.variant == 'P' else solve_milp_u
    t0 = time()
    T = solve(args.path, time_limit=args.time_limit, threads=args.threads,
              verbose=args.verbose)
    dt = time() - t0
    print(f'{args.path} ::: variant={args.variant} ::: T={T} ::: {dt:.2f}s')
