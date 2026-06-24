"""HDCARP inter/intra-route local search operators.

Routes are 1-D integer arrays of arc indices with sentinel depot 0s at the
first and last positions, e.g. ``[0, a1, a2, a3, 0]``. ``adj`` is the
arc-to-arc deadheading cost matrix (Floyd-Warshall shortest paths), where
``adj[i, j]`` is the cost of going from arc ``i`` to arc ``j``. Arc 0 is the
depot sentinel (service_time = 0, demand = 0, clss = 0).

Operators are *swap*-based: an intra-route swap exchanges the positions of two
arcs within one route; an inter-route swap exchanges one arc of route ``r1``
with one arc of route ``r2``. Service times never change under a swap (the same
arcs are still served), so only deadheading cost changes -- this lets every
delta be computed in O(1).

``variant``:
  - ``'P'`` (HDCARP-P): only arcs of the *same* priority class may be swapped.
  - ``'U'`` (HDCARP-U): any two arcs may be swapped regardless of class.
"""

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def calc_length(adj, service, sub):
    """Total cost (service + deadheading) of a single route ``sub``.

    Works for both numpy arrays and torch tensors (used by the RL path).
    """
    if hasattr(sub, "clone"):  # torch tensor
        s = service[sub].clone()
        a = adj[sub[:-1], sub[1:]].clone()
        s[1:] += a
        return s.sum()
    sub = np.asarray(sub)
    s = service[sub].astype(np.float64).copy()
    a = adj[sub[:-1], sub[1:]]
    s[1:] += a
    return s.sum()


def get_subtour_p(route, clss, p):
    """Interior indices of ``route`` whose arc has priority class ``p``."""
    route = np.asarray(route)
    idx = np.arange(1, len(route) - 1)
    return idx[clss[route[idx]] == p]


def get_subtour_u(route):
    """All interior (non-sentinel) indices of ``route``."""
    return np.arange(1, len(np.asarray(route)) - 1)


# --------------------------------------------------------------------------- #
# intra-route
# --------------------------------------------------------------------------- #
def calc_swap_delta_intra(route, i, j, adj):
    """O(1) cost delta of swapping positions ``i`` and ``j`` in ``route``.

    Positive delta means the swap is worse. ``route`` carries sentinels at 0
    and -1 so the neighbour lookups ``i-1`` and ``j+1`` are always valid for
    interior i, j. Requires ``i < j`` (caller guarantees this); for ``i == j``
    the delta is 0.
    """
    if i == j:
        return 0.0
    if i > j:
        i, j = j, i
    a = int(route[i - 1])
    bi = int(route[i])
    a2 = int(route[i + 1])
    c = int(route[j - 1])
    bj = int(route[j])
    d = int(route[j + 1])

    if j == i + 1:
        # adjacent: ... a, bi, bj, d ...  ->  ... a, bj, bi, d ...
        old = adj[a, bi] + adj[bi, bj] + adj[bj, d]
        new = adj[a, bj] + adj[bj, bi] + adj[bi, d]
    else:
        old = adj[a, bi] + adj[bi, a2] + adj[c, bj] + adj[bj, d]
        new = adj[a, bj] + adj[bj, a2] + adj[c, bi] + adj[bi, d]
    return float(new - old)


def _intra_delta_matrix(route, adj):
    """Full (m, m) swap-delta matrix over interior positions ``1..n-2``.

    ``D[a, b]`` (with ``a < b``) is ``calc_swap_delta_intra(route, pos[a], pos[b])``.
    Only the strict upper triangle is meaningful; the rest is filled with +inf.
    Returns ``(D, pos)`` where ``pos`` maps matrix rows/cols to route positions.
    """
    n = len(route)
    pos = np.arange(1, n - 1)
    m = len(pos)
    prev = route[pos - 1]
    cur = route[pos]
    nxt = route[pos + 1]

    # general (non-adjacent j > i+1) case, fully broadcast:
    #   new = adj[prev_i,cur_j] + adj[cur_j,nxt_i] + adj[prev_j,cur_i] + adj[cur_i,nxt_j]
    A1 = adj[prev[:, None], cur[None, :]]   # adj[prev_i, cur_j]
    A2 = adj[cur[None, :], nxt[:, None]]    # adj[cur_j, nxt_i]
    B1 = adj[prev[None, :], cur[:, None]]   # adj[prev_j, cur_i]
    B2 = adj[cur[:, None], nxt[None, :]]    # adj[cur_i, nxt_j]
    old_i = adj[prev, cur] + adj[cur, nxt]
    D = (A1 + A2 + B1 + B2) - (old_i[:, None] + old_i[None, :])

    # adjacent pairs (j == i+1) need the chained-edge formula, not the general one
    if m >= 2:
        ai = prev[:-1]
        bi = cur[:-1]
        bj = cur[1:]
        dd = route[pos[1:] + 1]
        adj_delta = (adj[ai, bj] + adj[bj, bi] + adj[bi, dd]) - \
                    (adj[ai, bi] + adj[bi, bj] + adj[bj, dd])
        ii = np.arange(m - 1)
        D[ii, ii + 1] = adj_delta

    # keep only the strict upper triangle (i < j); rest is invalid
    r = np.arange(m)
    D[r[:, None] >= r[None, :]] = np.inf
    return D, pos


def best_swap_intra(route, adj, clss, variant='P', p=None):
    """Best improving intra-route swap.

    Returns ``(best_delta, best_i, best_j)``; ``(0.0, -1, -1)`` if none improves.
    For variant ``'P'`` only same-class pairs are considered (restricted to
    class ``p`` when given); for ``'U'`` any interior pair is considered.
    """
    route = np.asarray(route)
    n = len(route)
    if n <= 3:  # [0, x, 0] or shorter -> nothing to swap
        return 0.0, -1, -1

    D, pos = _intra_delta_matrix(route, adj)

    cl = clss[route[pos]]
    if variant == 'P' and p is not None:
        # restrict to interior positions whose arc is class p
        keep = cl == p
        if keep.sum() < 2:
            return 0.0, -1, -1
        D[~keep, :] = np.inf
        D[:, ~keep] = np.inf
    elif variant == 'P':
        # only same-class pairs allowed
        D[cl[:, None] != cl[None, :]] = np.inf

    flat = int(np.argmin(D))
    best_delta = float(D.flat[flat])
    if best_delta < -EPS:
        m = len(pos)
        a, b = divmod(flat, m)
        return best_delta, int(pos[a]), int(pos[b])
    return 0.0, -1, -1


def intra_route_opt(route, adj, clss, variant='P', p=None, max_iter=100):
    """Repeatedly apply the best improving intra-route swap until convergence."""
    route = np.asarray(route).copy()
    for _ in range(max_iter):
        delta, i, j = best_swap_intra(route, adj, clss, variant=variant, p=p)
        if i < 0 or delta >= -EPS:
            break
        route[i], route[j] = route[j], route[i]
    return route


# --------------------------------------------------------------------------- #
# inter-route
# --------------------------------------------------------------------------- #
def calc_swap_delta_inter(r1, i, r2, j, adj):
    """O(1) delta of swapping arc at position ``i`` in ``r1`` with position ``j`` in ``r2``.

    The two routes are independent, so cost changes are additive across them.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    a1, x, a2 = int(r1[i - 1]), int(r1[i]), int(r1[i + 1])
    b1, y, b2 = int(r2[j - 1]), int(r2[j]), int(r2[j + 1])

    old = adj[a1, x] + adj[x, a2] + adj[b1, y] + adj[y, b2]
    new = adj[a1, y] + adj[y, a2] + adj[b1, x] + adj[x, b2]
    return float(new - old)


def best_swap_inter(r1, r2, adj, demands, capacity, clss, variant='P', p=None):
    """Best improving inter-route swap between ``r1`` and ``r2``.

    Capacity feasibility is checked *first* for each candidate (a swap that
    would push either route over ``capacity`` is rejected before computing the
    delta). Returns ``(best_delta, best_i, best_j)`` or ``(0.0, -1, -1)``.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    if len(r1) <= 2 or len(r2) <= 2:
        return 0.0, -1, -1

    load1 = demands[r1].sum()
    load2 = demands[r2].sum()

    idx1 = get_subtour_u(r1)
    idx2 = get_subtour_u(r2)

    a1 = r1[idx1 - 1]
    x = r1[idx1]
    a2 = r1[idx1 + 1]
    b1 = r2[idx2 - 1]
    y = r2[idx2]
    b2 = r2[idx2 + 1]

    dx = demands[x]
    dy = demands[y]

    #   new = adj[a1_i,y_j]+adj[y_j,a2_i] + adj[b1_j,x_i]+adj[x_i,b2_j]
    new = (adj[a1[:, None], y[None, :]] + adj[y[None, :], a2[:, None]]
           + adj[b1[None, :], x[:, None]] + adj[x[:, None], b2[None, :]])
    old_i = adj[a1, x] + adj[x, a2]
    old_j = adj[b1, y] + adj[y, b2]
    D = new - (old_i[:, None] + old_j[None, :])

    feas = (load1 - dx[:, None] + dy[None, :] <= capacity + EPS) & \
           (load2 - dy[None, :] + dx[:, None] <= capacity + EPS)
    if variant == 'P':
        feas &= clss[x][:, None] == clss[y][None, :]
    D[~feas] = np.inf

    flat = int(np.argmin(D))
    best_delta = float(D.flat[flat])
    if best_delta < -EPS:
        a, b = divmod(flat, len(idx2))
        return best_delta, int(idx1[a]), int(idx2[b])
    return 0.0, -1, -1


def _route_cost(route, adj):
    route = np.asarray(route)
    if len(route) < 2:
        return 0.0
    return float(adj[route[:-1], route[1:]].sum())


def inter_route_opt(routes, adj, demands, clss, capacity=1.0, variant='P', max_iter=100):
    """Apply the best improving inter-route swap across all route pairs.

    Greedily picks, over every ordered-unique pair of routes, the single best
    feasible improving swap and applies it; repeats until no pair yields an
    improvement (or ``max_iter`` reached). The objective minimised here is the
    total deadheading cost; capacity is never violated.
    """
    routes = [np.asarray(r).copy() for r in routes]
    for _ in range(max_iter):
        best = (0.0, -1, -1, -1, -1)  # delta, ri, rj, i, j
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                d, i, j = best_swap_inter(
                    routes[ri], routes[rj], adj, demands, capacity, clss,
                    variant=variant,
                )
                if i >= 0 and d < best[0] - EPS:
                    best = (d, ri, rj, i, j)
        if best[1] < 0:
            break
        _, ri, rj, i, j = best
        routes[ri][i], routes[rj][j] = routes[rj][j], routes[ri][i]
    return routes


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def ls(vars, variant='P', actions=None):
    """Main entry point used by ``meta.py``.

    ``actions``: list of flat numpy arrays (one per population member), each of
    shape ``(nseq + nroutes - 1,)`` with 0-separators between routes.

    Returns a list of route-lists (one per member); each route-list is what
    ``gen_tours`` produces (routes with sentinel 0s). ``meta.py`` feeds the
    result to ``deserialize_tours_batch`` / ``deserialize_tours``.

    Pipeline per member:
      1. parse flat actions into routes (gen_tours)
      2. intra-route optimisation per priority class (P) / globally (U)
      3. inter-route optimisation across all routes
    """
    from utils.nb_utils import gen_tours

    adj = vars['adj']
    adj = adj.numpy() if hasattr(adj, 'numpy') else np.asarray(adj)
    adj = adj.astype(np.float64)
    clss = vars['clss']
    clss = clss.numpy() if hasattr(clss, 'numpy') else np.asarray(clss)
    demands = vars['demand']
    demands = demands.numpy() if hasattr(demands, 'numpy') else np.asarray(demands)

    classes = [c for c in np.unique(clss) if c != 0]

    out = []
    for action in actions:
        if action is None:
            out.append([])
            continue
        action = np.asarray(action)
        routes = gen_tours(action)

        # intra-route
        new_routes = []
        for route in routes:
            if variant == 'P':
                for p in classes:
                    route = intra_route_opt(route, adj, clss, variant='P', p=p)
            else:
                route = intra_route_opt(route, adj, clss, variant='U')
            new_routes.append(route)
        routes = new_routes

        # inter-route
        routes = inter_route_opt(routes, adj, demands, clss,
                                 capacity=1.0, variant=variant)
        out.append(routes)
    return out


def lsRL(td, tours):
    """Legacy entry point for the RL trainer / HRDA inference.

    ``tours``: 2-D int tensor/array of shape ``(M, max_len)``; each row is a
    route of arc indices, zero-padded. Applies intra-route swap optimisation
    (U-variant: the RL policy already partitions by class via the Scheduler) to
    each non-trivial row in place and returns ``tours`` in its original type.
    """
    is_tensor = hasattr(tours, 'numpy')

    adj = td['adj']
    adj = adj.numpy() if hasattr(adj, 'numpy') else np.asarray(adj)
    adj = adj.astype(np.float64)
    # service time key has varied between revisions
    svc = td.get('service_times', td.get('service_time'))
    if hasattr(svc, 'numpy'):
        svc = svc.numpy()
    svc = np.asarray(svc)
    clss = td['clss']
    clss = clss.numpy() if hasattr(clss, 'numpy') else np.asarray(clss)

    # adj/clss may carry a leading batch dim of 1 (from RLHCARP); squeeze it.
    if adj.ndim == 3 and adj.shape[0] == 1:
        adj = adj[0]
    if clss.ndim == 2 and clss.shape[0] == 1:
        clss = clss[0]

    tours_np = tours.numpy() if is_tensor else np.asarray(tours)
    tours_np = tours_np.copy()

    for r in range(tours_np.shape[0]):
        row = tours_np[r]
        # treat the row as a route; strip trailing padding but keep the leading
        # depot. A row is [0, a1, a2, ..., 0, 0, ...]; the meaningful part runs
        # up to the last non-zero arc, plus a closing sentinel.
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            continue
        last = int(nz[-1])
        # route with closing sentinel
        route = np.concatenate([row[:last + 1], np.array([0], dtype=row.dtype)])
        if len(route) <= 3:
            continue
        route = intra_route_opt(route, adj, clss, variant='U')
        # write back the interior (drop the closing sentinel we added)
        tours_np[r, :last + 1] = route[:last + 1]

    if is_tensor:
        import torch
        return torch.as_tensor(tours_np, dtype=tours.dtype)
    return tours_np
