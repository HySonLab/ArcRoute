"""Heuristic class-1 opening-prefix generator for HDCARP variant P (P3
architecture: heuristic-only, K-candidate hedging -- see `docs/grpo.md`).

Under variant P's global precedence, every vehicle must serve all of its
class-1 arcs before any higher class, so a vehicle's class-1 workload is an
*opening route prefix* -- T1 therefore only depends on how the class-1 arcs
are partitioned across the <= M vehicles and ordered within each vehicle.

This module used to also carry an exact Held-Karp + partition DP
(`t1_oracle_argmin`, O(2^n1 * n1^2) time, applicable only below a class-1
arc-count guard). It was removed in the P3 refactor in favor of a single,
unconditional heuristic path (`t1_heuristic_argmin`, used at every n1) plus
K-candidate hedging (`t1_heuristic_candidates`): call the heuristic under K
different seeds and keep the distinct partitions found, instead of trusting
any single call's local optimum. Rationale/due-diligence (P3 report):
measured on 30 real `data/test_bm/medium.npz` instances with n1 <= 16 (the
former DP's applicable range), a single heuristic call matched the exact
optimum EXACTLY on 30/30 (0.000% mean/median/max gap); K-seed hedging (K=3,
5, 10) found nothing further to close. This is an empirical finding on one
benchmark, not a formal guarantee -- the heuristic carries no exactness
proof -- but dropping the DP measured zero T1 regression in the regime it
used to cover. Above n1=16, where `t1_heuristic_argmin` was already the only
option pre-refactor, its documented degradation curve (see
`HEURISTIC_TIME_BUDGET_DEFAULT`) is unchanged by this refactor.
"""

import time

import numpy as np

INF = float("inf")


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
    """Class-1 opening-prefix heuristic: the only class-1 solve path (P3 --
    see module docstring for why the former exact DP was removed). Returns
    T1, parts, and secondary/sec_plain/tie_gain (always 0.0 -- this solver
    has no exact tiebreak DP, unlike the removed `t1_oracle_argmin`). Runs in
    roughly constant wall-clock (bounded by `time_budget`) regardless of n1.
    Trades an exactness guarantee for an empirically-measured optimality gap
    (see module-level constants' docstrings and the module docstring's
    due-diligence numbers) -- callers wanting more than one shot at a good
    partition should use `t1_heuristic_candidates` (K-seed hedging) instead
    of relying on a single call's local optimum.

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
    # own demand exceeds capacity, no partition can ever place it.
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

    # Whole-route budget guard (reload-bug fix, see docs/root-cause-allow.md).
    # The class-1 arcs are only the PREFIX of each vehicle's single trip;
    # classes >=2 will later share the same capacity budget (one trip, no
    # reload). A class-1 partition that fits on its own can still be impossible
    # to complete without a reload -- e.g. it fills every vehicle so full that
    # the largest higher-class arc no longer fits in ANY vehicle's residual.
    # Reject such prefixes. Only NECESSARY conditions are checked (largest
    # higher-class arc must fit the most-empty vehicle; total higher demand
    # must fit the total residual), so no genuinely completable prefix is ever
    # discarded.
    higher = np.where(np.asarray(clss) >= 2)[0]
    if len(higher):
        loads = np.array([float(np.asarray(demands)[p].sum()) if p else 0.0
                          for p in parts], dtype=np.float64)
        residual = cap - loads
        hd = np.asarray(demands)[higher].astype(np.float64)
        if hd.sum() > residual.sum() + 1e-9 or hd.max() > residual.max() + 1e-9:
            return dict(T1=INF, parts=None, secondary=INF, sec_plain=INF,
                        tie_gain=0.0)

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


def t1_heuristic_candidates(adj, svc, clss, demands, M, K, cap=1.0,
                            seed_base=0, **kwargs):
    """K distinct class-1 opening-prefix candidates (P3 architecture).

    Deliberately the simplest possible generator: `K` independent
    `t1_heuristic_argmin` calls under seeds `seed_base .. seed_base+K-1`
    (different seeds diversify the construction's randomized restarts and
    the ILS perturbation trajectory -- see `t1_heuristic_argmin`), deduped by
    their ACTUAL resulting partition (not by T1 value alone -- two distinct
    partitions can tie on T1, and a T1-only dedup would silently collapse
    them, undercutting exactly the diversity K-candidate hedging is for).
    `t1_heuristic_argmin` itself is untouched -- this is a thin wrapper, no
    new search logic.

    Parameters
    ----------
    K        : number of seeds to try. Fewer than K distinct partitions may
               come back (dedup, or repeated capacity-infeasible seeds).
    seed_base: first seed tried; seeds `seed_base .. seed_base+K-1` in order.
    **kwargs : forwarded to every `t1_heuristic_argmin` call (`cap` is also
               forwarded; `time_budget`, `restarts`, `intra_cutoff`,
               `ils_iters`, `hk_polish_max` are the typical ones to tune when
               spreading a fixed total compute budget across K candidates).

    Returns
    -------
    List of `t1_heuristic_argmin` result dicts, sorted best-T1-first, always
    at least 1 entry (even when every seed hits an identical capacity-
    infeasible result -- callers check `np.isfinite(res["T1"])` per entry
    exactly as they would for a single call).
    """
    seen = set()
    out = []
    last_res = None
    for k in range(max(int(K), 1)):
        res = t1_heuristic_argmin(adj, svc, clss, demands, M, cap=cap,
                                  seed=seed_base + k, **kwargs)
        last_res = res
        if res["parts"] is None:
            continue
        key = tuple(tuple(p) for p in res["parts"])
        if key in seen:
            continue
        seen.add(key)
        out.append(res)
    if not out:
        out = [last_res]
    out.sort(key=lambda r: r["T1"])
    return out
