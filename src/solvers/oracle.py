"""Exact T1 oracle for HDCARP variant P (argmin recovery).

Under variant P's global precedence, every vehicle must serve all of its
class-1 arcs before any higher class, so a vehicle's class-1 workload is an
*opening route prefix*. T1 therefore only depends on how the class-1 arcs are
partitioned across the <= M vehicles and ordered within each vehicle -- a
problem small enough (n1 class-1 arcs, typically ~8-10 on the OSM benchmarks)
to solve exactly:

  1. Held-Karp DP over the class-1 arcs:
       dp[S][last] = min completion time of serving subset S ending at arc
       `last`, where completion follows the evaluator/env semantics
         completion(a_i) = svc[0] + adj[0,a_1] + svc[a_1] + adj[a_1,a_2]
                           + ... + svc[a_i].
     Subsets whose total demand exceeds the capacity are discarded.
  2. Partition DP across m = 1..min(M, n1) vehicles:
       gm[m][S] = min over submask partitions of S into m parts of the max
       per-part Held-Karp path cost.  T1* = gm[min(M,n1)][full].
  3. Argmin recovery walks the partition DP back, extracting one certifying
     submask per vehicle, then the Held-Karp parent pointers give each
     vehicle's optimal service order.

Optional exact lexicographic tiebreak (`tiebreak=True`): among ALL
T1-optimal solutions (parts whose path cost is <= T1* + tol, orderings with
completion <= T1* + tol), a secondary sum-DP minimizes

    sum over vehicles of (prefix finish time + dist(final position, nearest
                          class-2 arc))

i.e. a "how soon can each vehicle start on class-2" readiness proxy. Empty
vehicles contribute dist(depot, nearest class-2 arc). The primary T1 stays
provably optimal (every part is constrained to the T1* tie set); the
secondary is exactly minimized within that tie set.

Complexity is O(2^n1 * n1^2) time and O(2^n1 * n1) memory, so the oracle is
only *applicable* for small n1. Callers must treat `T1 = inf` (returned when
`n1 > max_n1`, or when no capacity-feasible partition exists) as "oracle not
applicable" and fall back to their non-oracle pipeline.
"""

import time

import numpy as np

INF = float("inf")

#: Default guard on the class-1 arc count. The DP is O(2^n1 * n1^2); n1 = 16
#: means ~1.7e7 (subset, last) states -- around a second of pure-python work.
#: Benchmark `data/osm_test_small` peaks at n1 = 10, but the guard must hold
#: for arbitrary future data, hence it is enforced, not assumed.
MAX_N1_DEFAULT = 16


def _held_karp(adj, svc, c1):
    """Held-Karp over the class-1 arcs `c1` (global indices).

    Returns (dp, par):
      dp[S][last]  min completion of serving subset S ending at c1[last]
      par[S][last] previous local index on that optimal path (-1 = path start)
    """
    n = len(c1)
    svc0 = float(svc[0])
    d0 = adj[0, c1]
    dd = adj[np.ix_(c1, c1)]
    sv = svc[c1]
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    par = [[-1] * n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = svc0 + float(d0[i]) + float(sv[i])
    for S in range(1, size):
        row = dp[S]
        for last in range(n):
            c = row[last]
            if c == INF or not (S >> last) & 1:
                continue
            base = dd[last]
            for nxt in range(n):
                if (S >> nxt) & 1:
                    continue
                S2 = S | (1 << nxt)
                c2 = c + float(base[nxt]) + float(sv[nxt])
                if c2 < dp[S2][nxt]:
                    dp[S2][nxt] = c2
                    par[S2][nxt] = last
    return dp, par


def _path_from(par, S, last):
    """Recover the local-index service order ending at `last` covering mask S."""
    seq = []
    while last != -1:
        seq.append(last)
        p = par[S][last]
        S ^= 1 << last
        last = p
    return seq[::-1]


def _nearest_c2(adj, clss, from_idx):
    """Shortest-path distance from arc/depot index `from_idx` to the nearest
    class-2 arc (fallback: class-3; 0.0 if neither class exists)."""
    for k in (2, 3):
        ck = np.where(clss == k)[0]
        if len(ck):
            return float(adj[from_idx, ck].min())
    return 0.0


def t1_oracle_argmin(adj, svc, clss, demands, M, cap=1.0, tiebreak=False,
                     max_n1=MAX_N1_DEFAULT):
    """Exact T1 optimum + a witnessing class-1 partition/ordering.

    Parameters
    ----------
    adj      : (N1, N1) arc-to-arc shortest-path matrix, index 0 = depot.
    svc      : (N1,) service times (svc[0] is the depot's, usually 0).
    clss     : (N1,) arc classes (0 = depot row, 1..P = priority classes).
    demands  : (N1,) normalized demands.
    M        : fleet size (upper bound on the number of parts).
    cap      : vehicle capacity in `demands` units (default 1.0).
    tiebreak : exact secondary DP over the T1-optimal tie set (see module doc).
    max_n1   : applicability guard; if the class-1 arc count exceeds it, the
               oracle declines (T1 = inf) instead of running an exponential DP.

    Returns dict with
      T1        : optimal T1 (float). `inf` means NOT APPLICABLE -- either
                  n1 > max_n1 or no capacity-feasible partition exists; the
                  caller must then skip the oracle path entirely.
      parts     : list of exactly M lists of GLOBAL arc indices in service
                  order (vehicle v's forced opening prefix; possibly empty),
                  or None when T1 is inf.
      secondary : secondary objective value of the returned solution.
      sec_plain : secondary value of the plain (first-found) argmin.
      tie_gain  : sec_plain - secondary (> 0 iff tiebreak changed the answer).
    """
    c1 = np.where(clss == 1)[0]
    n = len(c1)
    if n == 0:
        return dict(T1=0.0, parts=[[] for _ in range(int(M))],
                    secondary=0.0, sec_plain=0.0, tie_gain=0.0)
    if max_n1 is not None and n > max_n1:
        return dict(T1=INF, parts=None, secondary=INF, sec_plain=INF,
                    tie_gain=0.0)

    dp, par = _held_karp(adj, svc, c1)
    size = 1 << n
    FULL = size - 1
    dem = demands[c1]

    # total demand per subset (incremental over the lowest set bit)
    demS = np.zeros(size)
    for S in range(1, size):
        low = S & (-S)
        demS[S] = demS[S ^ low] + dem[low.bit_length() - 1]

    # single-vehicle cost per subset (best Held-Karp end), capacity-filtered
    cost1 = np.empty(size)
    arglast = np.full(size, -1, dtype=np.int64)
    cost1[0] = 0.0
    for S in range(1, size):
        row = dp[S]
        best, bi = INF, -1
        for i in range(n):
            if row[i] < best:
                best, bi = row[i], i
        cost1[S] = best
        arglast[S] = bi
    feas = demS <= cap + 1e-9
    g1 = np.where(feas, cost1, INF)
    g1[0] = 0.0

    K = min(int(M), n)

    # ---- primary DP: gm[m][S] = min over partitions of S into m parts of the
    # max part cost (empty parts allowed via the T=0 / prow[S] branch).
    gm = np.full((K + 1, size), INF)
    gm[0, 0] = 0.0
    gm[1, :] = g1
    for m in range(2, K + 1):
        grow = gm[m]
        prow = gm[m - 1]
        for S in range(size):
            best = prow[S]                      # T = 0 (empty part)
            T = S
            while T:
                a = g1[T]
                if a < best:
                    b = prow[S ^ T]
                    v = a if a > b else b
                    if v < best:
                        best = v
                T = (T - 1) & S
            grow[S] = best
    T1 = float(gm[K, FULL])
    if not np.isfinite(T1):
        return dict(T1=INF, parts=None, secondary=INF, sec_plain=INF,
                    tie_gain=0.0)
    thr = T1 + max(1e-9, 1e-9 * abs(T1))

    # ---- plain argmin recovery (first submask that certifies the DP value)
    def recover_plain():
        masks = []
        S = FULL
        for m in range(K, 1, -1):
            found = None
            T = S
            while True:
                a = g1[T]
                b = gm[m - 1, S ^ T]
                if max(a, b) <= thr:
                    found = T
                    break
                if T == 0:
                    break
                T = (T - 1) & S
            masks.append(found)
            S ^= found
        masks.append(S)
        return masks

    # per-arc distance to the nearest class-2 arc (secondary objective terms)
    d2 = np.array([_nearest_c2(adj, clss, int(g)) for g in c1])
    d2_depot = _nearest_c2(adj, clss, 0)

    def sec_of_masks(masks, lasts=None):
        """Secondary value of a list of part masks (`lasts`: chosen Held-Karp
        end index per mask; None -> the cost-argmin end)."""
        tot = 0.0
        for i, T in enumerate(masks):
            if T == 0:
                tot += d2_depot
            else:
                la = arglast[T] if lasts is None else lasts[i]
                tot += dp[T][la] + d2[la]
        return tot

    plain_masks = recover_plain()
    plain_lasts = [arglast[T] if T else -1 for T in plain_masks]
    sec_plain = sec_of_masks(plain_masks)

    out_masks, out_lasts, secondary = plain_masks, plain_lasts, sec_plain

    if tiebreak:
        # ---- secondary DP restricted to the T1-optimal tie set:
        # sec_part[S] = min over ends `last` with dp[S][last] <= thr of
        # dp + d2 (INF if the part cannot appear in any T1 optimum),
        # argsec[S] = that end.
        sec_part = np.full(size, INF)
        argsec = np.full(size, -1, dtype=np.int64)
        sec_part[0] = d2_depot
        for S in range(1, size):
            if g1[S] > thr:                     # part can't appear in any optimum
                continue
            row = dp[S]
            best, bi = INF, -1
            for i in range(n):
                c = row[i]
                if c <= thr:
                    v = c + d2[i]
                    if v < best:
                        best, bi = v, i
            sec_part[S] = best
            argsec[S] = bi

        g2 = np.full((K + 1, size), INF)
        g2[0, 0] = 0.0
        g2[1, :] = sec_part
        for m in range(2, K + 1):
            grow = g2[m]
            prow = g2[m - 1]
            for S in range(size):
                best = sec_part[0] + prow[S]     # empty part for this vehicle
                T = S
                while T:
                    a = sec_part[T]
                    if a < INF:
                        b = prow[S ^ T]
                        if a + b < best:
                            best = a + b
                    T = (T - 1) & S
                grow[S] = best
        # standard sum-DP recovery: at (m, S) find T certifying
        # sec_part[T] + g2[m-1][S^T] == g2[m][S] (within tol), recurse on S^T.
        tb_masks = []
        S = FULL
        ok = True
        for m in range(K, 1, -1):
            target = g2[m, S]
            tol = max(1e-9, 1e-9 * abs(target))
            found = None
            T = S
            while True:
                if sec_part[T] + g2[m - 1, S ^ T] <= target + tol:
                    found = T
                    break
                if T == 0:
                    break
                T = (T - 1) & S
            if found is None:
                ok = False
                break
            tb_masks.append(found)
            S ^= found
        if ok:
            tb_masks.append(S)
            tb_lasts = [int(argsec[T]) if T else -1 for T in tb_masks]
            secondary = sec_of_masks(tb_masks, tb_lasts)
            out_masks, out_lasts = tb_masks, tb_lasts

    # ---- materialize parts as global arc-index sequences, padded to M
    parts = []
    for T, la in zip(out_masks, out_lasts):
        if T == 0:
            parts.append([])
        else:
            order = _path_from(par, T, int(la))
            parts.append([int(c1[i]) for i in order])
    while len(parts) < int(M):
        parts.append([])

    return dict(T1=T1, parts=parts, secondary=float(secondary),
                sec_plain=float(sec_plain),
                tie_gain=float(sec_plain - secondary))


#: Default class-1 arc count above which the local-search heuristic switches
#: from "full" (2-opt/Or-opt intra-route polish + relocate/swap) to a lighter
#: relocate/swap-only variant. Measured (`docs/grpo.md` "heuristic" section):
#: at n1 >= ~25-30, the O(n^2)-O(n^3) intra-route sweeps crowd out ILS
#: restarts within a fixed time budget, and skipping them roughly halves the
#: mean optimality gap at n1=30 (3.88% -> 1.98%). Below that, "full" wins.
HEURISTIC_INTRA_CUTOFF_DEFAULT = 25

#: Default per-instance time budget for the heuristic. Measured tradeoff
#: (real `data/osm_train` instances, cross-validated against the exact DP
#: where tractable): mean gap 0.39%/0.10%/0.04% at 0.03s/0.15s/0.5s budgets
#: (n1 <= 19); at n1 up to 35 vs a long best-known reference, 0.5s gives
#: 0.18%/1.7%/2.7%/5.6% mean gap at n1~19/25/31/35 respectively, at well
#: under 1s wall-clock throughout.
HEURISTIC_TIME_BUDGET_DEFAULT = 0.5


def _t1_route_cost(seq, d0, dd, sv, svc0):
    if not seq:
        return 0.0
    c = svc0 + d0[seq[0]] + sv[seq[0]]
    for a, b in zip(seq, seq[1:]):
        c += dd[a, b] + sv[b]
    return c


def _t1_best_insertion(seq, x, d0, dd, sv):
    """(delta, position) of the cheapest insertion of local-index `x` into `seq`."""
    if not seq:
        return d0[x] + sv[x], 0
    best, pos = d0[x] + dd[x, seq[0]] - d0[seq[0]] + sv[x], 0
    for i in range(1, len(seq)):
        u, w = seq[i - 1], seq[i]
        dlt = dd[u, x] + dd[x, w] - dd[u, w] + sv[x]
        if dlt < best:
            best, pos = dlt, i
    dlt = dd[seq[-1], x] + sv[x]
    if dlt < best:
        best, pos = dlt, len(seq)
    return best, pos


def _t1_improve_route(seq, d0, dd, sv, svc0):
    """2-opt + Or-opt(1,2) to local optimum on one route. Returns (seq, cost)."""
    cost = _t1_route_cost(seq, d0, dd, sv, svc0)
    n = len(seq)
    if n < 2:
        return seq, cost
    improved = True
    while improved:
        improved = False
        n = len(seq)
        for i in range(n - 1):                    # 2-opt (asymmetric -> full recompute)
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = _t1_route_cost(cand, d0, dd, sv, svc0)
                if c < cost - 1e-12:
                    seq, cost, improved = cand, c, True
        n = len(seq)
        for L in (1, 2):                           # Or-opt: relocate a 1-2 segment
            if n <= L:
                continue
            for i in range(n - L + 1):
                segf = seq[i:i + L]
                rest = seq[:i] + seq[i + L:]
                for segment in (segf, segf[::-1]) if L > 1 else (segf,):
                    for p in range(len(rest) + 1):
                        if p == i and segment is segf:
                            continue
                        cand = rest[:p] + segment + rest[p:]
                        c = _t1_route_cost(cand, d0, dd, sv, svc0)
                        if c < cost - 1e-12:
                            seq, cost, improved = cand, c, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return seq, cost


def _t1_construct(n, K, d0, dd, sv, svc0, dem, cap, rng, randomize):
    """Seeded greedy best-insertion partition. Returns (routes, costs, loads)
    in local index space, or None if no capacity-feasible partition exists."""
    sym = np.minimum(dd[:n, :n], dd[:n, :n].T)
    if randomize:
        first = int(rng.choice(n, p=(d0[:n] + 1e-9) / (d0[:n] + 1e-9).sum()))
    else:
        first = int(np.argmax(d0[:n]))
    seeds = [first]
    while len(seeds) < K:
        mind = sym[:, seeds].min(axis=1)
        mind[seeds] = -1.0
        if randomize:
            top = np.argsort(-mind)[:3]
            seeds.append(int(rng.choice(top)))
        else:
            seeds.append(int(np.argmax(mind)))
    routes = [[s] for s in seeds]
    loads = [dem[s] for s in seeds]
    costs = [_t1_route_cost(r, d0, dd, sv, svc0) for r in routes]

    rest = [i for i in range(n) if i not in seeds]
    rest.sort(key=lambda i: -d0[i])
    if randomize and len(rest) > 1:
        for _ in range(max(1, len(rest) // 3)):
            a, b = rng.integers(0, len(rest), 2)
            rest[a], rest[b] = rest[b], rest[a]

    for x in rest:
        best_key, best_v, best_p, best_c = None, -1, -1, 0.0
        for v in range(K):
            if loads[v] + dem[x] > cap + 1e-9:
                continue
            dlt, pos = _t1_best_insertion(routes[v], x, d0, dd, sv)
            newc = costs[v] + dlt
            key = tuple(sorted([newc if u == v else costs[u]
                                for u in range(K)], reverse=True))
            if best_key is None or key < best_key:
                best_key, best_v, best_p, best_c = key, v, pos, newc
        if best_v < 0:
            return None                                    # capacity-infeasible
        routes[best_v].insert(best_p, x)
        costs[best_v] = best_c
        loads[best_v] += dem[x]
    costs = [_t1_route_cost(r, d0, dd, sv, svc0) for r in routes]
    return routes, costs, loads


def _t1_key(costs):
    return tuple(sorted(costs, reverse=True))


def _t1_local_search(routes, costs, loads, d0, dd, sv, svc0, dem, cap,
                     deadline, use_swap=True, use_intra=True):
    """First-improvement relocate/swap (+ optional intra-route 2-opt/Or-opt),
    accepting only moves that strictly reduce the descending-sorted route-cost
    tuple (min-max first, balance second -- escapes max-plateaus)."""
    K = len(routes)
    if use_intra:
        for v in range(K):
            routes[v], costs[v] = _t1_improve_route(routes[v], d0, dd, sv, svc0)
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        cur = _t1_key(costs)
        cmax = cur[0]
        maxvs = [v for v in range(K) if costs[v] >= cmax - 1e-12]
        for v in maxvs:                            # relocate out of a maximal route
            for xi in range(len(routes[v])):
                x = routes[v][xi]
                src = routes[v][:xi] + routes[v][xi + 1:]
                csrc = _t1_route_cost(src, d0, dd, sv, svc0)
                for u in range(K):
                    if u == v or loads[u] + dem[x] > cap + 1e-9:
                        continue
                    dlt, pos = _t1_best_insertion(routes[u], x, d0, dd, sv)
                    nc = list(costs)
                    nc[v], nc[u] = csrc, costs[u] + dlt
                    if _t1_key(nc) < cur:
                        routes[u].insert(pos, x)
                        routes[v] = src
                        loads[v] -= dem[x]
                        loads[u] += dem[x]
                        if use_intra:
                            routes[v], csrc = _t1_improve_route(routes[v], d0, dd, sv, svc0)
                            routes[u], ncu = _t1_improve_route(routes[u], d0, dd, sv, svc0)
                        else:
                            ncu = costs[u] + dlt
                        costs[v], costs[u] = csrc, ncu
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        if use_swap:                               # swap between two routes
            for v in maxvs:
                for xi in range(len(routes[v])):
                    x = routes[v][xi]
                    for u in range(K):
                        if u == v:
                            continue
                        for yi in range(len(routes[u])):
                            y = routes[u][yi]
                            if (loads[v] - dem[x] + dem[y] > cap + 1e-9 or
                                    loads[u] - dem[y] + dem[x] > cap + 1e-9):
                                continue
                            rv = routes[v][:xi] + routes[v][xi + 1:]
                            ru = routes[u][:yi] + routes[u][yi + 1:]
                            dv, pv = _t1_best_insertion(rv, y, d0, dd, sv)
                            du, pu = _t1_best_insertion(ru, x, d0, dd, sv)
                            cv = _t1_route_cost(rv, d0, dd, sv, svc0) + dv
                            cu = _t1_route_cost(ru, d0, dd, sv, svc0) + du
                            nc = list(costs)
                            nc[v], nc[u] = cv, cu
                            if _t1_key(nc) < cur:
                                rv.insert(pv, y)
                                ru.insert(pu, x)
                                routes[v], routes[u] = rv, ru
                                loads[v] += dem[y] - dem[x]
                                loads[u] += dem[x] - dem[y]
                                if use_intra:
                                    routes[v], cv = _t1_improve_route(routes[v], d0, dd, sv, svc0)
                                    routes[u], cu = _t1_improve_route(routes[u], d0, dd, sv, svc0)
                                costs[v], costs[u] = cv, cu
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    return routes, costs, loads


def _t1_perturb(routes, loads, dem, cap, rng, kmoves=3):
    """Random capacity-safe relocations of `kmoves` arcs, for ILS restarts."""
    K = len(routes)
    routes = [list(r) for r in routes]
    loads = list(loads)
    for _ in range(kmoves):
        vs = [v for v in range(K) if routes[v]]
        if not vs:
            break
        v = int(rng.choice(vs))
        xi = int(rng.integers(0, len(routes[v])))
        x = routes[v][xi]
        us = [u for u in range(K)
              if u != v and loads[u] + dem[x] <= cap + 1e-9]
        if not us:
            continue
        u = int(rng.choice(us))
        routes[v].pop(xi)
        routes[u].insert(int(rng.integers(0, len(routes[u]) + 1)), x)
        loads[v] -= dem[x]
        loads[u] += dem[x]
    return routes, loads


def t1_heuristic_argmin(adj, svc, clss, demands, M, cap=1.0,
                        time_budget=HEURISTIC_TIME_BUDGET_DEFAULT,
                        restarts=4, hk_polish_max=12,
                        intra_cutoff=HEURISTIC_INTRA_CUTOFF_DEFAULT,
                        ils_iters=40, seed=0):
    """Fast heuristic drop-in for `t1_oracle_argmin` -- same return contract
    (T1, parts, secondary/sec_plain/tie_gain -- the latter three always 0.0
    here, this solver has no exact tiebreak DP), but scales to any n1 in
    roughly constant wall-clock (bounded by `time_budget`) instead of
    `O(2^n1 * n1^2)`. Trades the exact guarantee for an empirically-measured
    optimality gap (see module-level constants' docstrings) -- intended for
    n1 beyond `t1_oracle_argmin`'s `max_n1` guard, where the exact DP is not
    an option at all, not as a faster substitute where the DP is already fast.

    Pipeline: multi-restart farthest-point-seeded greedy best-insertion
    construction, first-improvement local search (relocate/swap, +2-opt/Or-opt
    intra-route polish below `intra_cutoff` class-1 arcs -- above it, skipping
    the O(n^2)-O(n^3) intra-route sweeps and spending the same time budget on
    more relocate/swap/ILS restarts measured better on average, see
    `HEURISTIC_INTRA_CUTOFF_DEFAULT`), then ILS (random capacity-safe
    relocation kicks + re-descent) for the remaining budget. A final exact
    Held-Karp re-ordering polishes any route with <= `hk_polish_max` arcs
    (provably order-optimal for its final assignment, independent of budget).

    `time_budget` is soft, not hard: the deadline is only checked between
    local-search sweeps/restarts, not inside a single sweep, so a large
    instance's sweep can overshoot it somewhat (measured up to ~2x at n1=35).
    """
    t_start = time.perf_counter()
    deadline = t_start + time_budget
    c1 = np.where(clss == 1)[0]
    n = len(c1)
    Mi = int(M)
    if n == 0:
        return dict(T1=0.0, parts=[[] for _ in range(Mi)], secondary=0.0,
                    sec_plain=0.0, tie_gain=0.0)
    K = min(Mi, n)
    use_intra = n < intra_cutoff

    svc0 = float(svc[0])
    d0 = adj[0, c1].astype(np.float64)
    dd = adj[np.ix_(c1, c1)].astype(np.float64)
    sv = svc[c1].astype(np.float64)
    dem = demands[c1].astype(np.float64)

    # Necessary feasibility check the construction step doesn't otherwise
    # make (it always seeds K routes unconditionally): if any single arc's
    # own demand exceeds capacity, no partition can ever place it -- matches
    # `t1_oracle_argmin`'s per-singleton-subset capacity gate.
    if (dem > cap + 1e-9).any():
        return dict(T1=INF, parts=None, secondary=INF, sec_plain=INF, tie_gain=0.0)

    rng = np.random.default_rng(seed)
    best_key, best_routes = None, None
    for r in range(max(1, restarts)):
        if r > 0 and time.perf_counter() > deadline:
            break
        built = _t1_construct(n, K, d0, dd, sv, svc0, dem, cap, rng,
                              randomize=(r > 0))
        if built is None:
            continue
        routes, costs, loads = built
        routes, costs, loads = _t1_local_search(
            routes, costs, loads, d0, dd, sv, svc0, dem, cap, deadline,
            use_swap=True, use_intra=use_intra)
        k = _t1_key(costs)
        if best_key is None or k < best_key:
            best_key, best_routes = k, [list(x) for x in routes]

    if best_routes is not None and ils_iters:
        best_loads = [float(sum(dem[i] for i in rt)) for rt in best_routes]
        for _ in range(ils_iters):
            if time.perf_counter() > deadline:
                break
            routes, loads = _t1_perturb(best_routes, best_loads, dem, cap, rng)
            costs = [_t1_route_cost(r, d0, dd, sv, svc0) for r in routes]
            routes, costs, loads = _t1_local_search(
                routes, costs, loads, d0, dd, sv, svc0, dem, cap, deadline,
                use_swap=True, use_intra=use_intra)
            k = _t1_key(costs)
            if k < best_key:
                best_key, best_routes, best_loads = k, [list(x) for x in routes], list(loads)

    if best_routes is None:
        return dict(T1=INF, parts=None, secondary=INF, sec_plain=INF, tie_gain=0.0)

    if hk_polish_max and hk_polish_max > 0:
        for v in range(len(best_routes)):
            if 2 <= len(best_routes[v]) <= hk_polish_max:
                order, _ = _hk_path_reorder(best_routes[v], d0, dd, sv, svc0)
                best_routes[v] = order

    costs = [_t1_route_cost(rt, d0, dd, sv, svc0) for rt in best_routes]
    parts = [[int(c1[i]) for i in rt] for rt in best_routes]
    while len(parts) < Mi:
        parts.append([])
    return dict(T1=float(max(costs)), parts=parts, secondary=0.0,
                sec_plain=0.0, tie_gain=0.0)


def _hk_path_reorder(idxs, d0, dd, sv, svc0):
    """Exact Held-Karp re-ordering of a small (local-index) arc set `idxs`.
    Returns (order, cost) with `order` a permutation of `idxs`."""
    n = len(idxs)
    if n <= 1:
        return list(idxs), _t1_route_cost(list(idxs), d0, dd, sv, svc0)
    d0s = np.asarray([d0[i] for i in idxs])
    dds = np.asarray([[dd[a, b] for b in idxs] for a in idxs])
    svs = np.asarray([sv[i] for i in idxs])
    size = 1 << n
    dp = np.full((size, n), INF)
    par = np.full((size, n), -1, dtype=np.int64)
    for i in range(n):
        dp[1 << i, i] = svc0 + d0s[i] + svs[i]
    for S in range(1, size):
        row = dp[S]
        for last in range(n):
            c = row[last]
            if c == INF or not (S >> last) & 1:
                continue
            cand = c + dds[last] + svs
            for nxt in range(n):
                if (S >> nxt) & 1:
                    continue
                S2 = S | (1 << nxt)
                if cand[nxt] < dp[S2, nxt]:
                    dp[S2, nxt] = cand[nxt]
                    par[S2, nxt] = last
    FULL = size - 1
    last = int(np.argmin(dp[FULL]))
    order, S = [], FULL
    while last != -1:
        order.append(idxs[last])
        p = int(par[S, last])
        S ^= 1 << last
        last = p
    return order[::-1], float(dp[FULL, int(np.argmin(dp[FULL]))])


def forced_flat_actions(parts, N1):
    """Flatten oracle `parts` into Joint M x N flat decode actions.

    Vehicle v serves its whole prefix, then v+1, ... -- per-vehicle clocks are
    independent under the Joint M x N env, so cross-vehicle interleaving is
    irrelevant to the resulting T-vector.

    parts : list (len <= M) of global arc-index lists, in service order.
    N1    : number of arc columns per vehicle (N + 1 incl. the depot column);
            flat action = vehicle_m * N1 + arc_j.
    """
    acts = []
    for v, part in enumerate(parts):
        for j in part:
            acts.append(v * N1 + int(j))
    return acts
