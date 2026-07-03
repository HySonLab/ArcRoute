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
