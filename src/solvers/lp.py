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
import contextlib
import io
import os
from time import time

import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT

from utils.ops import import_instance, print_route_log


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
      V_t_k          dict class -> set of head vertices of that class's arcs
      V_t            set, union of all V_t_k (plus the depot)
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
        # V_t^k must be the HEAD vertices of class-k required arcs: a vehicle's
        # physical position at a hierarchy-level boundary is always either the
        # depot or the head of the last arc it serviced, so the dummy node's
        # transition arcs A_f/A_t must attach exactly there. Using tails
        # instead would force a spurious post-service deadhead (from the head
        # of the last serviced arc to some tail vertex) before the vehicle
        # could exit the level -- charged to t_{m,k}, inflating T_k and making
        # the "exact" optimum worse than feasible heuristic solutions.
        V_t_k[int(clss)].add(int(head))
    V_t = set().union(*V_t_k.values()) if V_t_k else set()
    V_t = V_t | {depot}  # depot is a service-relevant vertex too

    # ----- all original arcs A (parallel arcs collapsed) ----------------- #
    # Parallel arcs sharing the same (tail, head) are merged into a single
    # deadhead arc keeping the CHEAPEST traversal time. This is exact for
    # deadheading (a walk never gains from a costlier parallel copy) and is
    # REQUIRED for correctness: the deadhead variables y are indexed by the
    # (tail, head) pair, so if A kept both copies the flow-balance (11) and
    # deadhead-cost (3) sums -- which iterate A / A' -- would reference the
    # SAME y variable twice. That double-charges the arc's cost AND injects a
    # phantom second unit of flow through its endpoints, which the balance
    # constraints then have to cancel with spurious extra deadheading. The
    # net effect inflates t_{m,k}: SCIP proves an "optimum" strictly worse
    # than a solution the true model admits (reproduced on small.npz
    # instances 18/21/52/63, whose shortest depot->service deadheads run
    # through such a duplicated arc). Deduplicating restores the intended
    # deadhead network.
    cost = {}
    nodes = set()
    for tail, head, _d, _c, _s, traversal in np.vstack([req, nonreq]):
        a = (int(tail), int(head))
        c = float(traversal)
        if a not in cost or c < cost[a]:
            cost[a] = c
        nodes.add(int(tail))
        nodes.add(int(head))
    A = list(cost.keys())  # unique (tail, head) arcs, insertion order

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
    # NOTE: `threads` is effectively a NO-OP. Community SCIP's branch-and-bound
    # under model.optimize() is single-threaded; 'parallel/maxnthreads' only
    # applies to the concurrent-solve mode (model.solveConcurrent()), which is
    # UNUSABLE here: the concurrent sub-solvers do not inherit the Python
    # ConnectivityHandler, silently accept sub-tour solutions, and return an
    # infeasible "optimum" (verified empirically on data/bm5/small/inst_00).
    # The parameter is still set so the intent is preserved should a compiled
    # (copyable) constraint handler ever replace the Python one.
    try:
        model.setParam('parallel/maxnthreads', threads)
    except Exception:
        pass


def _heuristic_ub(es, w, P, variant='P'):
    """Weighted-sum objective value of a fast feasible heuristic solution.

    Runs the InsertCheapest constructive heuristic (~30 ms on the bm5
    instances) and returns sum_k w_k * T_k^heur. Any feasible schedule is an
    upper bound on the MILP optimum (the MILP models the same physics, cf. the
    evaluator cross-check), so the value is safe to use as an objective
    cutoff. Returns None if the heuristic cannot be run on ``es``.

    ``variant`` selects the local-search neighbourhood ('P' or 'U'); the
    returned T is always computed by the evaluator's class-sorted P-builder
    (``SolutionEvaluator`` default), i.e. it is the value of a *P-feasible*
    schedule. That makes it a valid upper bound for MILP-P and, since every
    P-feasible schedule is U-feasible (serve class-k arcs at level k, level
    completion = last class-k arc completion), for MILP-U as well.

    The global NumPy RNG state is saved/restored so callers see no side
    effects; a fixed internal seed makes the bound reproducible.
    """
    try:
        from solvers.meta import InsertCheapestHCARP
        state = np.random.get_state()
        try:
            np.random.seed(0xA5C1)
            h = InsertCheapestHCARP()
            h.import_instance(es)
            T = np.asarray(h(variant=variant, num_sample=20),
                           dtype=np.float64).ravel()
        finally:
            np.random.set_state(state)
        if T.shape[0] != P or not np.all(np.isfinite(T)):
            return None
        return float(sum(w[k] * T[k - 1] for k in range(1, P + 1)))
    except Exception:
        return None


def _extract_T(model, T, P):
    """Return np.array([T_1, ..., T_p]) if a solution exists, else None."""
    if model.getNSols() == 0:
        return None
    return np.array([model.getVal(T[k]) for k in range(1, P + 1)],
                    dtype=np.float64)


def _heuristic_routes(es):
    """Per-vehicle serviced-arc sequences from the InsertCheapest heuristic.

    Returns a list of lists of (tail, head) arcs in service order (one list per
    used vehicle, class-sorted), or None if the constructive heuristic fails.
    Mirrors ``_heuristic_ub``'s call convention (fixed seed, RNG save/restore)
    so the routing matches the schedule whose value ``_heuristic_ub`` reports.
    """
    try:
        from solvers.meta import InsertCheapestHCARP, lex_argbest
        from utils.ops import gen_tours
        state = np.random.get_state()
        try:
            np.random.seed(0xA5C1)
            h = InsertCheapestHCARP()
            h.import_instance(es)
            acts = [h.get_once(h.M, h.clss) for _ in range(20)]
            acts = [a for a in acts if a is not None]
            if not acts:
                return None
            arms, Ts = h.ls_lex_guarded(acts)
            best = arms[lex_argbest(Ts)]
        finally:
            np.random.set_state(state)
        # 1-based arc indices; gen_tours strips depot sentinels.
        return [[int(a) for a in tour[1:-1]] for tour in gen_tours(best)]
    except Exception:
        return None


def _warmstart_u(G, es, w, routes=None):
    """Build a MILP-U-feasible starting incumbent + its objective.

    Takes a P-feasible routing (class-sorted per vehicle) and translates it
    into MILP-U variable values: class-k arcs are serviced at hierarchy
    level = (rank of class k among the classes that vehicle actually serves),
    i.e. the vehicle's occupied levels form a prefix (matching the
    level-compaction symmetry break), and the inter-service deadheads are
    realised as shortest paths over the deadhead network A, so the flow-balance
    (28) / level-linking (25) / capacity (27) constraints all hold by
    construction. A P-feasible schedule is always U-feasible, so this is both a
    valid incumbent AND a valid objective cutoff.

    ``routes`` is an optional per-vehicle list of 1-based required-arc-index
    sequences in service order (depot sentinels are tolerated/stripped). When
    supplied — e.g. the ACTUAL MILP-P optimal routing via
    ``p_service_order_routes`` — the translated incumbent has the P optimum's
    completion times, so MILP-U starts from (and, since SCIP only keeps
    improving incumbents, never regresses below) T_1^P. When ``routes`` is None
    the fast InsertCheapest constructive heuristic's routing is used instead
    (the standalone default / fallback).

    Returns (vals, obj) with ``vals`` = {'x','y','r','t','T'} keyed exactly as
    ``solve_milp_u``'s variables, or (None, None) on any failure (the caller
    then falls back to the plain heuristic bound with no warm start).
    """
    if routes is None:
        routes = _heuristic_routes(es)
    if routes is None:
        return None, None
    # Tolerate depot sentinels (0) so callers may pass reconstruct_routes_p's
    # raw per-vehicle arrays ([0, a_i, ..., 0]) directly.
    routes = [[int(a) for a in seq if int(a) != 0] for seq in routes]
    P, Mn = G['P'], G['M']
    M = list(range(Mn))
    levels = list(range(1, P + 1))
    classes = levels
    depot, dummy = G['depot'], G['dummy']
    req = G['req_arcs']

    # deadhead network (deduplicated A with cheapest traversal costs)
    DG = nx.DiGraph()
    for a in G['A']:
        DG.add_edge(a[0], a[1], weight=G['cost'][a])

    def path_arcs(u, v):
        if u == v:
            return []
        try:
            seq = nx.shortest_path(DG, u, v, weight='weight')
        except Exception:
            return None
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    xval = {(m, a, h): 0 for m in M for a in req for h in levels}
    yval = {(m, a, h): 0 for m in M for a in G['A_prime'] for h in levels}
    rval = {(m, k, h): 0 for m in M for k in classes for h in levels}
    tval = {(m, 0): 0.0 for m in M}
    Tval = {k: 0.0 for k in classes}

    for m in M:
        seq = routes[m] if m < len(routes) else []
        arcs = [req[a - 1] for a in seq]  # heuristic order, class-sorted
        by_class = {}
        for a in arcs:
            by_class.setdefault(G['cls'][a], []).append(a)
        prev_exit = depot
        h = 0
        for k in sorted(by_class):        # occupied levels form a prefix
            h += 1
            cur = prev_exit
            dh = sv = 0.0
            for a in by_class[k]:
                pa = path_arcs(cur, a[0])
                if pa is None:
                    return None, None
                for e in pa:
                    yval[(m, e, h)] += 1
                    dh += G['cost'][e]
                xval[(m, a, h)] = 1
                sv += G['s'][a]
                cur = a[1]
            yval[(m, (dummy, prev_exit), h)] = 1   # A_f: enter level h
            yval[(m, (cur, dummy), h)] = 1          # A_t: exit level h
            rval[(m, k, h)] = 1
            tval[(m, h)] = tval[(m, h - 1)] + sv + dh
            Tval[k] = max(Tval[k], tval[(m, h)])
            prev_exit = cur
        for hh in range(h + 1, P + 1):    # trailing empty levels stay put
            yval[(m, (dummy, prev_exit), hh)] = 1
            yval[(m, (prev_exit, dummy), hh)] = 1
            tval[(m, hh)] = tval[(m, hh - 1)]

    obj = float(sum(w[k] * Tval[k] for k in classes))
    return {'x': xval, 'y': yval, 'r': rval, 't': tval, 'T': Tval}, obj


def _inject_u(model, vals, x, y, r, t, T):
    """Register ``vals`` as a starting incumbent; return True if SCIP keeps it.

    Built in the ORIGINAL problem space (createSol before optimize/transform);
    addSol transfers it into the solution store, so SCIP starts branch-and-
    bound from this incumbent instead of having to discover one.
    """
    sol = model.createSol()
    for key, var in x.items():
        model.setSolVal(sol, var, vals['x'][key])
    for key, var in y.items():
        model.setSolVal(sol, var, vals['y'][key])
    for key, var in r.items():
        model.setSolVal(sol, var, vals['r'][key])
    for key, var in t.items():
        if not isinstance(var, float):     # (m, 0) entries are plain floats
            model.setSolVal(sol, var, vals['t'][key])
    for key, var in T.items():
        model.setSolVal(sol, var, vals['T'][key])
    return model.addSol(sol)


# --------------------------------------------------------------------------- #
# MILP-P (Section 3.3)
# --------------------------------------------------------------------------- #
def solve_milp_p(es, time_limit=3600, threads=8, verbose=False,
                 return_model=False, heur_ub=True):
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

    # Objective cutoff from a fast feasible heuristic: gives SCIP a primal
    # bound to prune against from node 1 (the default run spends ~15% of its
    # time before finding any usable incumbent). Safe: the heuristic schedule
    # is feasible, so the true optimum is <= its objective; the 1e-3 margin
    # (1e-3 of one T_P unit) covers float-arithmetic differences between the
    # heuristic evaluator and SCIP. If the cutoff ever proves too tight (no
    # solution found), we re-solve without it below — exactness is never lost.
    ub_obj = _heuristic_ub(es, w, P) if heur_ub else None
    if ub_obj is not None:
        model.setObjlimit(ub_obj + 1e-3)

    # --- variables -------------------------------------------------------- #
    x = {(m, a): model.addVar(f'x_{m}_{a}', vtype='B')
         for m in M for a in G['req_arcs']}
    y = {}
    for m in M:
        for k in classes:
            # Tighter valid ub for deadhead counts: some optimal solution has
            # every deadhead segment (between consecutive services / the level
            # entry/exit) replaced by a shortest path, which is simple; a level
            # with n_k serviced arcs has at most n_k + 1 such segments, so arc
            # a is deadheaded at most n_k + 1 times in that optimum. Bounds
            # only need to preserve one optimum, so this is exact.
            nk_cls = len(G['req_by_class'][k])
            for a in G['A_prime']:
                ub = (1 if (a[0] == dummy or a[1] == dummy)
                      else min(len(G['nodes']), nk_cls + 1))
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
            # (5) r links service — disaggregated form x <= r (equivalent to
            # sum_a x <= n_k r over binaries, but a strictly tighter LP
            # relaxation: r >= max_a x instead of r >= avg_a x).
            for a in G['req_by_class'][k]:
                model.addCons(x[(m, a)] <= r[(m, k)])
            # (7) exactly one A_f arc per (m, k)
            model.addCons(quicksum(y[(m, a, k)] for a in G['A_f']) == 1)
        # (6) class 1 starts at the depot
        model.addCons(y[(m, (dummy, depot), 1)] == 1)
        # (8) link level entry to previous-level exit.
        #
        # The paper writes (8) over ∪_{k'<k} V_t^{k'} ∪ {v_0}, but that set
        # omits vertices in V_t^k that are not tails of lower-class arcs. A
        # vehicle can exit level k-1 at such a vertex (via a deadhead traversal)
        # and re-enter level k at a different vertex with zero deadhead cost —
        # a free repositioning that under-counts deadheading. Using the full V_t
        # (as MILP-U's analogous constraint (25) does) closes this gap and
        # ensures every inter-level transition is physically accounted for.
        for k in classes:
            if k == 1:
                continue
            for v in sorted(G['V_t']):  # V_t already includes the depot
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

    # Branch on the big-M indicators r first, then on the assignments x,
    # before the deadhead counts y. Pure search-order hint (feasible region
    # and objective untouched); resolving the T_k >= t - beta(1-r)
    # disjunctions early shrinks the tree by ~10x on the bm5 instances.
    for var in r.values():
        model.chgVarBranchPriority(var, 10)
    for var in x.values():
        model.chgVarBranchPriority(var, 5)

    model.optimize()
    if (ub_obj is not None and model.getNSols() == 0
            and model.getStatus() in ('infeasible', 'inforunbd')):
        # The whole search was cut off by the objective cutoff (a heuristic
        # bound below the model optimum would indicate a semantics mismatch)
        # — re-solve without it so exactness can never be lost to the cutoff.
        # Deliberately NOT triggered on 'timelimit' with no solution: that is
        # the same outcome the cutoff-free model would report.
        model.freeTransform()
        model.setObjlimit(model.infinity())
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
                 return_model=False, heur_ub=True, p_solution=None):
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

    # Warm start (constructed below the variables) + objective cutoff. The
    # MILP-U model is much larger than MILP-P (x and r carry an extra level
    # index), and SCIP's default heuristics fail to find ANY incumbent within
    # the time budget on several small-tier instances -- reported as
    # "infeasible / no solution" even though every one is feasible (any
    # P-feasible schedule is U-feasible). We therefore translate the
    # InsertCheapest heuristic's P-feasible routing into a concrete U incumbent
    # and inject it after the model is built (see _warmstart_u / _inject_u).
    #
    # The cutoff is taken from THAT SAME schedule's objective rather than from
    # a separate _heuristic_ub() call: the two agree to float noise, but using
    # the injected solution's own value guarantees the incumbent survives the
    # objlimit (a value from an independent evaluator can round a hair below
    # the reconstruction and reject the very solution we seed). If the warm
    # start cannot be built we fall back to the plain heuristic bound.
    #
    # ``p_solution`` (optional): the ACTUAL MILP-P optimal routing for this
    # instance (per-vehicle 1-based arc-index sequences in true service order,
    # from ``p_service_order_routes``). Because HDCARP-U relaxes HDCARP-P's
    # precedence constraint, every P-feasible schedule is U-feasible; seeding SCIP with
    # the *proven-optimal* P routing (instead of the fast-but-suboptimal
    # InsertCheapest heuristic) makes MILP-U's initial incumbent equal to the P
    # optimum, so the reported T_1 starts at T_1^P and can only stay equal or
    # improve — it can never regress above T_1^P. When p_solution is None the
    # heuristic routing is used (standalone default / fallback for instances
    # where MILP-P is not available or not proven).
    warm_vals, ub_obj = (_warmstart_u(G, es, w, routes=p_solution)
                         if heur_ub else (None, None))
    if warm_vals is None and p_solution is not None and heur_ub:
        # A supplied P routing failed to translate (e.g. an arc pair with no
        # deadhead path in the deduplicated network) -- fall back to the
        # constructive heuristic warm start so feasibility is still guaranteed.
        warm_vals, ub_obj = _warmstart_u(G, es, w, routes=None)
    if ub_obj is None and heur_ub:
        ub_obj = _heuristic_ub(es, w, P, variant='U')
    if ub_obj is not None:
        model.setObjlimit(ub_obj + 1e-3)

    # Tighter valid ub for deadhead counts, MILP-U analog of the MILP-P
    # n_k + 1 bound. In some optimum every deadhead stretch of a level
    # (entry -> first service, between consecutive services, last service ->
    # exit) is a simple shortest path, so with n services at level h an arc is
    # deadheaded at most n + 1 times there. Unlike MILP-P a level may serve
    # arcs of ANY class, so n is not bounded per class -- but capacity (27)
    # bounds it: vehicle m serves at most n_cap arcs over ALL its levels,
    # where n_cap = max #required arcs whose total demand fits in Q
    # (smallest-demands-first count; 1e-6 slack matches SCIP's feastol).
    # Hence y[(m,a,h)] <= n_cap + 1 in that optimum.
    q_sorted = np.sort(np.array([G['q'][a] for a in G['req_arcs']],
                                dtype=np.float64))
    n_cap = int((np.cumsum(q_sorted) <= Q + 1e-6).sum()) if len(q_sorted) else 0

    # --- variables -------------------------------------------------------- #
    x = {(m, a, h): model.addVar(f'x_{m}_{a}_{h}', vtype='B')
         for m in M for a in G['req_arcs'] for h in levels}
    y = {}
    for m in M:
        for h in levels:
            for a in G['A_prime']:
                ub = (1 if (a[0] == dummy or a[1] == dummy)
                      else min(len(G['nodes']), n_cap + 1))
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
                # (22) r links service of class k at level h — disaggregated
                # form x <= r (equivalent to sum_a x <= n_k r over binaries,
                # but a strictly tighter LP relaxation: r >= max_a x instead
                # of r >= avg_a x), mirroring the MILP-P (5) disaggregation.
                for a in G['req_by_class'][k]:
                    model.addCons(x[(m, a, h)] <= r[(m, k, h)])
        # (23) level 1 starts at the depot
        model.addCons(y[(m, (dummy, depot), 1)] == 1)
        # (25) link level entry to previous-level exit.
        #
        # The paper's literal (25) couples every v in V_t U {v_0}: the vertex
        # at which a route enters hierarchy level h (through v'_0) must equal
        # the vertex at which it left level h-1. This encodes the physical
        # requirement (paper, Section 3.2) that "the node visited immediately
        # after v'_0 must be the same as the node visited right before v'_0" --
        # i.e. v'_0 is a zero-cost *marker* between levels, not a teleporter.
        #
        # Because A_f / A_t connect v'_0 to *every* vertex in V_t U {v_0}, the
        # entry and exit vertices of any level can be any element of that set,
        # so the coupling MUST cover all of V_t to guarantee continuity. A
        # previous restriction to only the tails of lower-index classes left
        # the coupling missing at every other vertex, letting a vehicle exit
        # level h-1 at one vertex and re-enter level h at a different one for
        # free (e.g. exit at v8, re-enter at v7 with zero deadhead). That free
        # repositioning under-counts deadheading and produced impossibly small
        # completion times (T_1 below the true service+deadhead of its arcs).
        for h in levels:
            if h == 1:
                continue
            for v in sorted(G['V_t']):  # V_t already includes the depot
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

    # --- valid strengthening (no optimum lost) ------------------------------ #
    n_req = len(G['req_arcs'])
    for m in M:
        # Capacity-cardinality cut: n_cap is the max number of required arcs
        # whose total demand fits in Q (greedy smallest-first), so any
        # capacity-feasible service set of vehicle m has <= n_cap arcs.
        # Implied by (27) + integrality but NOT by the LP relaxation.
        model.addCons(quicksum(x[(m, a, h)] for a in G['req_arcs']
                               for h in levels) <= n_cap)
        # Level-compaction symmetry break: an x-empty level h with a non-empty
        # level h+1 is a pure symmetry — (i) any repositioning deadheads of the
        # empty level can be prepended to level h+1's walk and its spurious
        # r's dropped (all charged completion times unchanged or relaxed),
        # then (ii) level h+1's entire content (x, y, r) shifts down into
        # level h: t_{m,h}^new = t_{m,h+1}^old and every T_k charge keeps its
        # numeric value, so the objective is identical. Hence some optimum
        # has each vehicle's service-carrying levels forming a prefix, and
        # forcing "level h empty => level h+1 empty" loses no optimum while
        # collapsing the P-fold level-shift symmetry of the search tree.
        for h in levels[:-1]:
            model.addCons(
                quicksum(x[(m, a, h + 1)] for a in G['req_arcs'])
                <= min(n_req, n_cap)
                * quicksum(x[(m, a, h)] for a in G['req_arcs']))

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

    # NOTE: deliberately NO r/x branch-priority hints here, unlike MILP-P.
    # Measured on bm5/small/inst_01 (600 s, identical model otherwise):
    # with r>x priorities the gap stalls at 27% and SCIP finds no incumbent
    # on some instances; with SCIP's default (reliability pseudocost)
    # branching the gap is 0.15%. In MILP-U the r[m,k,h] indicators are a
    # P x P matrix per vehicle whose values are mostly implied by x, so
    # forcing them to the top of the tree wastes the search.

    # Inject the P-feasible warm start as the initial incumbent (see above).
    # Guarantees NSols >= 1 whenever the instance is feasible, so SCIP reports
    # a solution instead of a spurious "no solution", and gives the search a
    # good primal bound to improve on rather than to rediscover.
    if warm_vals is not None:
        _inject_u(model, warm_vals, x, y, r, t, T)

    model.optimize()
    if (ub_obj is not None and model.getNSols() == 0
            and model.getStatus() in ('infeasible', 'inforunbd')):
        # The whole search was cut off by the objective cutoff (a heuristic
        # bound below the model optimum would indicate a semantics mismatch)
        # — re-solve without it so exactness can never be lost to the cutoff.
        # Deliberately NOT triggered on 'timelimit' with no solution: that is
        # the same outcome the cutoff-free model would report.
        model.freeTransform()
        model.setObjlimit(model.infinity())
        model.optimize()
    Tvec = _extract_T(model, T, P)
    if return_model:
        return Tvec, model, {'x': x, 'y': y, 'r': r, 't': t, 'T': T, 'G': G,
                             'M': M, 'levels': levels}
    return Tvec


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
    """Per-vehicle arc sequences for MILP-P (class order, greedy deadhead)."""
    a2i = _arc_to_idx_map(req_raw)
    routes = []
    for m in range(G['M']):
        route_arcs, prev = [], 0
        for k in range(1, G['P'] + 1):
            arcs_k = [a2i[a] for a in G['req_arcs']
                      if G['cls'][a] == k and model.getVal(x_vars[(m, a)]) > 0.5]
            if arcs_k:
                ordered = _greedy_order(arcs_k, adj, prev)
                route_arcs.extend(ordered)
                prev = route_arcs[-1]
        routes.append(np.array([0] + route_arcs + [0], dtype=np.int32))
    return routes


def p_service_order_routes(model, x_vars, y_vars, G, req_raw):
    """Per-vehicle serviced-arc sequences in the P solution's TRUE order.

    Unlike ``reconstruct_routes_p`` (which re-orders each class's arcs by a
    greedy nearest-neighbour heuristic — fine for a human-readable route log
    but lossy w.r.t. the optimal deadhead structure), this follows the P
    solution's own flow: for each (vehicle, class) it walks an Euler circuit
    of the serviced (x) and deadhead (y) arcs the solver actually used,
    starting/ending at the dummy node, and records the serviced arcs in the
    order they are traversed. Feeding the result to ``_warmstart_u`` (which
    realises inter-service deadheads as shortest paths) reproduces the P
    optimum's completion times EXACTLY, because in the P optimum every
    consecutive-service deadhead is already a shortest path.

    Returns a list (one per vehicle) of 1-based required-arc-index sequences
    concatenated across classes 1..P — the format ``_warmstart_u``/
    ``solve_milp_u(p_solution=...)`` expect.
    """
    a2i = _arc_to_idx_map(req_raw)
    dummy = G['dummy']
    routes = []
    for m in range(G['M']):
        seq = []
        for k in range(1, G['P'] + 1):
            # Collect the arcs (tail, head, arc_index_or_None) this vehicle
            # uses at class/level k: serviced required arcs and deadheads (with
            # integer multiplicity), plus the dummy transition arcs A_f / A_t.
            used = []
            for a in G['req_by_class'][k]:
                if model.getVal(x_vars[(m, a)]) > 0.5:
                    used.append((a[0], a[1], a2i[a]))
            for a in G['A_prime']:
                cnt = int(round(model.getVal(y_vars[(m, a, k)])))
                used.extend([(a[0], a[1], None)] * cnt)
            if not used:
                continue
            # Hierholzer Euler circuit from the dummy node (the flow is
            # balanced per node by constraint (11), so a circuit exists).
            adj = {}
            for eidx, (tail, _h, _ai) in enumerate(used):
                adj.setdefault(tail, []).append(eidx)
            circuit = []
            stack = [(dummy, None)]
            while stack:
                v, ein = stack[-1]
                bucket = adj.get(v)
                if bucket:
                    eidx = bucket.pop()
                    stack.append((used[eidx][1], eidx))
                else:
                    _, e = stack.pop()
                    if e is not None:
                        circuit.append(e)
            circuit.reverse()
            seq.extend(used[e][2] for e in circuit if used[e][2] is not None)
        routes.append(seq)
    return routes


def reconstruct_routes_u(model, x_vars, G, req_raw, adj):
    """Per-vehicle arc sequences for MILP-U (level order, greedy deadhead)."""
    a2i = _arc_to_idx_map(req_raw)
    routes = []
    for m in range(G['M']):
        route_arcs, prev = [], 0
        for h in range(1, G['P'] + 1):
            arcs_h = [a2i[a] for a in G['req_arcs']
                      if model.getVal(x_vars[(m, a, h)]) > 0.5]
            if arcs_h:
                ordered = _greedy_order(arcs_h, adj, prev)
                route_arcs.extend(ordered)
                prev = route_arcs[-1]
        routes.append(np.array([0] + route_arcs + [0], dtype=np.int32))
    return routes


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='MILP solver for HDCARP-P / HDCARP-U')
    p.add_argument('--path', required=True, help='path to a .npz instance')
    p.add_argument('--variant', default='P', choices=['P', 'U'])
    p.add_argument('--time_limit', type=float, default=3600)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--log', default=None, help='write route log to this file (in addition to stdout)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log:
        # ── single-instance route log mode ────────────────────────────────
        es = np.load(args.path)
        req_raw = np.asarray(es['req'])
        M_n, P_n = int(es['M']), int(es['P'])
        dms, _, _, demands, _, _, _, _ = import_instance(args.path)

        print(f"MILP-{args.variant} solver")
        print(f"Instance : {args.path}")
        print(f"P={P_n} classes | M={M_n} vehicles | {len(req_raw)} required arcs")
        print(f"Time limit: {args.time_limit}s | Threads: {args.threads}")
        print("Solving...", flush=True)

        solver_fn = solve_milp_p if args.variant == "P" else solve_milp_u
        t0 = time()
        Tvec, model, vars_dict = solver_fn(
            es, time_limit=args.time_limit, threads=args.threads,
            verbose=False, return_model=True,
        )
        elapsed = time() - t0

        if Tvec is None:
            print(f"\nNo feasible solution found within {elapsed:.1f}s.")
            raise SystemExit(1)

        print(f"Solved in {elapsed:.3f}s\n")
        G = vars_dict['G']
        x = vars_dict['x']
        routes = (reconstruct_routes_p if args.variant == "P" else reconstruct_routes_u)(
            model, x, G, req_raw, dms
        )
        header = (f"MILP-{args.variant} solver\nInstance : {args.path}\n"
                  f"P={P_n} classes | M={M_n} vehicles | {len(req_raw)} required arcs"
                  f" | solved in {elapsed:.3f}s\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(header)
            print_route_log(routes, req_raw, demands, dms, Tvec)
        out = buf.getvalue()
        print(out)
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as f:
            f.write(out)
        print(f"Log written to {args.log}")
    else:
        # ── fast T-only mode ──────────────────────────────────────────────
        solve = solve_milp_p if args.variant == 'P' else solve_milp_u
        t0 = time()
        T = solve(args.path, time_limit=args.time_limit, threads=args.threads,
                  verbose=args.verbose)
        dt = time() - t0
        print(f'{args.path} ::: variant={args.variant} ::: T={T} ::: {dt:.2f}s')
