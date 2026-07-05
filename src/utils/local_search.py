"""HDCARP inter/intra-route local search operators.

Routes are 1-D integer arrays of arc indices with sentinel depot 0s at the
first and last positions, e.g. ``[0, a1, a2, a3, 0]``. ``adj`` is the
arc-to-arc deadheading cost matrix (Floyd-Warshall shortest paths), where
``adj[i, j]`` is the cost of going from arc ``i`` to arc ``j``. Arc 0 is the
depot sentinel (service_time = 0, demand = 0, clss = 0).

Two move families are provided, both with O(1) deltas (service times never
change under either move -- the same arcs are still served -- so only
deadheading cost changes):

  - *swap*: exchange the positions of two arcs (within one route, or one arc
    of route ``r1`` with one arc of route ``r2``).
  - *relocate* (Or-opt-1): remove one arc and re-insert it immediately after
    another position (within the same route, or into another route). This is
    the move that lets variant U actually exploit cross-class reordering --
    e.g. pulling an arc from position 5 to position 1 in one step instead of a
    chain of swaps.

``variant``:
  - ``'P'`` (HDCARP-P): only arcs of the *same* priority class may be swapped;
    a relocate must preserve the route's class sequence (intra: the whole span
    between the removal and insertion points is one class; inter: the arc is
    inserted right after an arc of its own class).
  - ``'U'`` (HDCARP-U): any move regardless of class.
"""

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def _best_intra_moves(route, adj, clss, variant='P', p=None):
    """Fused intra-route move search: best swap AND best relocate in one call.

    Returns ``(sd, si, sj, rd, ri, rj)`` — exactly what ``best_swap_intra``
    and ``best_relocate_intra`` return, computed from shared gathers. All
    deltas are the same expressions evaluated in the same elementwise order
    (bit-identical values), and both selections keep the row-major-argmin
    tie-breaking of the standalone searches.

    Variant ``'P'``: swaps only between same-class pairs; relocates only
    within the arc's contiguous same-class run. With ``p`` given, only
    class-``p`` arcs are considered as moving arcs.
    """
    no_move = (0.0, -1, -1)
    route = np.asarray(route)
    n = len(route)
    if n <= 3:  # [0, x, 0] or shorter -> nothing to move
        return no_move + no_move

    cl = clss[route]
    pos = np.arange(1, n - 1)
    if variant == 'P' and p is not None:
        pos = pos[cl[pos] == p]
        if len(pos) == 0:
            return no_move + no_move
        x = route[pos]
        prev = route[pos - 1]
        nxt = route[pos + 1]
        contiguous = False
    else:
        # unrestricted rows: use slice views instead of fancy-index copies
        x = route[1:-1]
        prev = route[:-2]
        nxt = route[2:]
        contiguous = True
    m = len(pos)

    base = adj[route[:-1], route[1:]]      # edge costs adj[route[j], route[j+1]]
    e_in = base[pos - 1] if not contiguous else base[:-1]    # adj[prev, x]
    e_out = base[pos] if not contiguous else base[1:]        # adj[x, nxt]

    # ---- swap (cf. best_swap_intra) ----
    swap = no_move
    if m >= 2:
        A1 = adj[prev[:, None], x[None, :]]
        A2 = adj[x[None, :], nxt[:, None]]
        B1 = adj[prev[None, :], x[:, None]]
        B2 = adj[x[:, None], nxt[None, :]]
        old_i = e_in + e_out
        D = (A1 + A2 + B1 + B2) - (old_i[:, None] + old_i[None, :])
        # route-adjacent pairs need the chained-edge formula
        k = np.arange(m - 1) if contiguous else \
            np.flatnonzero(pos[1:] == pos[:-1] + 1)
        if len(k) > 0:
            ai = prev[k]
            bi = x[k]
            bj = x[k + 1]
            dd = nxt[k + 1]
            D[k, k + 1] = (adj[ai, bj] + adj[bj, bi] + adj[bi, dd]) - \
                          (adj[ai, bi] + adj[bi, bj] + adj[bj, dd])
        r = np.arange(m)
        D[r[:, None] >= r[None, :]] = np.inf
        if variant == 'P' and p is None:
            clp = cl[1:-1]
            D[clp[:, None] != clp[None, :]] = np.inf
        flat = int(D.argmin())
        bd = float(D.flat[flat])
        if bd < -EPS:
            a, b = divmod(flat, m)
            swap = (bd, int(pos[a]), int(pos[b]))

    # ---- relocate (cf. best_relocate_intra) ----
    rel = no_move
    rem = adj[prev, nxt] - e_in - e_out                     # (m,)
    u = route[:-1]                                          # slots j = 0..n-2
    v = route[1:]
    D = rem[:, None] + adj[u[None, :], x[:, None]] \
        + adj[x[:, None], v[None, :]] - base[None, :]

    allowed = np.ones((m, n - 1), dtype=bool)
    ar = np.arange(m)
    allowed[ar, pos] = False                                # no-op targets
    allowed[ar, pos - 1] = False
    if variant == 'P':
        # target slot must stay inside the arc's same-class run
        change = np.flatnonzero(cl[1:] != cl[:-1]) + 1      # run start indices
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change - 1, [n - 1]))
        block = np.zeros(n, dtype=np.int64)
        block[change] = 1
        block = np.cumsum(block)                            # run id per position
        lo = starts[block[pos]] - 1                         # min allowed j
        hi = np.minimum(ends[block[pos]], n - 2)            # max allowed j
        jpos = np.arange(0, n - 1)
        allowed &= (jpos[None, :] >= lo[:, None]) & (jpos[None, :] <= hi[:, None])

    D = np.where(allowed, D, np.inf)
    flat = int(D.argmin())
    bd = float(D.flat[flat])
    if bd < -EPS:
        a, b = divmod(flat, n - 1)
        rel = (bd, int(pos[a]), int(b))

    return swap + rel


def best_swap_intra(route, adj, clss, variant='P', p=None):
    """Best improving intra-route swap.

    Returns ``(best_delta, best_i, best_j)``; ``(0.0, -1, -1)`` if none improves.
    For variant ``'P'`` only same-class pairs are considered (restricted to
    class ``p`` when given); for ``'U'`` any interior pair is considered.
    """
    sd, si, sj, _, _, _ = _best_intra_moves(route, adj, clss, variant=variant, p=p)
    return sd, si, sj


def calc_relocate_delta_intra(route, i, j, adj):
    """O(1) cost delta of removing the arc at position ``i`` and re-inserting
    it immediately after position ``j`` (both in ORIGINAL route indexing).

    Removal reconnects ``route[i-1] -> route[i+1]``; insertion splits the edge
    ``route[j] -> route[j+1]``. The two edge deltas are additive and remain
    exact even for the adjacent case ``j == i + 1`` (the insertion edge
    ``(route[i+1], route[i+2])`` is untouched by the removal). The only
    degenerate targets are ``j in (i-1, i)`` -- re-inserting where the arc
    already sits -- which are no-ops with delta 0.

    ``i`` must be interior (``1 <= i <= n-2``); ``j`` ranges ``0..n-2``
    (``j == 0`` inserts right after the leading depot sentinel).
    """
    if j == i or j == i - 1:
        return 0.0
    prv, x, nxt = int(route[i - 1]), int(route[i]), int(route[i + 1])
    u, v = int(route[j]), int(route[j + 1])
    rem = adj[prv, nxt] - adj[prv, x] - adj[x, nxt]
    ins = adj[u, x] + adj[x, v] - adj[u, v]
    return float(rem + ins)


def apply_relocate_intra(route, i, j):
    """Return a new route with the arc at position ``i`` moved to sit
    immediately after (original) position ``j``."""
    route = np.asarray(route)
    x = route[i]
    r = np.delete(route, i)
    if j < i:
        return np.insert(r, j + 1, x)
    # deletion shifted original position j to j-1; inserting at index j puts
    # x right after it.
    return np.insert(r, j, x)


def best_relocate_intra(route, adj, clss, variant='P', p=None):
    """Best improving intra-route relocate (Or-opt-1).

    Returns ``(best_delta, best_i, best_j)`` -- move arc at position ``i`` to
    just after position ``j`` -- or ``(0.0, -1, -1)`` if none improves.

    Variant ``'P'``: a relocate is only allowed when it preserves the route's
    class sequence, i.e. every interior position spanned by the move (``i..j``
    when ``j > i``, ``j+1..i`` when ``j < i-1``) carries the moving arc's own
    class. On a class-sorted route this is exactly "reorder freely within your
    class block (incl. jumping to the block front)". With ``p`` given, only
    class-``p`` arcs are considered. Variant ``'U'``: any ``(i, j)``.
    """
    _, _, _, rd, ri, rj = _best_intra_moves(route, adj, clss, variant=variant, p=p)
    return rd, ri, rj


def intra_route_opt(route, adj, clss, variant='P', p=None, max_iter=100):
    """Repeatedly apply the best improving intra-route move -- swap or
    relocate, whichever improves more -- until neither improves."""
    route = np.asarray(route).copy()
    for _ in range(max_iter):
        sd, si, sj, rd, ri, rj = _best_intra_moves(
            route, adj, clss, variant=variant, p=p)
        if si < 0 and ri < 0:
            break
        if si >= 0 and (ri < 0 or sd <= rd):
            route[si], route[sj] = route[sj], route[si]
        else:
            route = apply_relocate_intra(route, ri, rj)
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

    flat = int(D.argmin())
    best_delta = float(D.flat[flat])
    if best_delta < -EPS:
        a, b = divmod(flat, len(idx2))
        return best_delta, int(idx1[a]), int(idx2[b])
    return 0.0, -1, -1


def calc_relocate_delta_inter(r1, i, r2, j, adj):
    """O(1) delta of moving the arc at position ``i`` of ``r1`` into ``r2``,
    inserted immediately after position ``j``. The routes are independent, so
    the removal delta (in ``r1``) and insertion delta (in ``r2``) are additive.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    prv, x, nxt = int(r1[i - 1]), int(r1[i]), int(r1[i + 1])
    u, v = int(r2[j]), int(r2[j + 1])
    rem = adj[prv, nxt] - adj[prv, x] - adj[x, nxt]
    ins = adj[u, x] + adj[x, v] - adj[u, v]
    return float(rem + ins)


def best_relocate_inter(r1, r2, adj, demands, capacity, clss, variant='P', p=None):
    """Best improving relocate of one arc out of ``r1`` into ``r2``.

    Capacity feasibility on the receiving route is checked first (``r2``'s
    load plus the moved arc's demand must stay within ``capacity``; ``r1``
    only loses load, so it stays feasible by construction). For variant
    ``'P'`` the arc may only be inserted immediately after an arc of its own
    class (so a class-sorted ``r2`` stays class-sorted); for ``'U'`` any slot
    (including right after the leading depot) is allowed.

    Returns ``(best_delta, best_i, best_j)`` -- arc at position ``i`` of
    ``r1`` goes right after position ``j`` of ``r2`` -- or ``(0.0, -1, -1)``.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    if len(r1) <= 2 or len(r2) < 2:
        return 0.0, -1, -1

    idx1 = get_subtour_u(r1)                 # donor positions
    jpos = np.arange(0, len(r2) - 1)         # insert-after slots in r2
    x = r1[idx1]
    rem = adj[r1[idx1 - 1], r1[idx1 + 1]] - adj[r1[idx1 - 1], x] \
        - adj[x, r1[idx1 + 1]]
    u = r2[jpos]
    v = r2[jpos + 1]
    ins = (adj[u[None, :], x[:, None]] + adj[x[:, None], v[None, :]]
           - adj[u, v][None, :])
    D = rem[:, None] + ins

    load2 = demands[r2].sum()
    cap_ok = load2 + demands[x] <= capacity + EPS       # (len(idx1),)
    D[~cap_ok, :] = np.inf
    if variant == 'P':
        same = clss[x][:, None] == clss[u][None, :]
        if p is not None:
            same &= (clss[x] == p)[:, None]
        D[~same] = np.inf

    flat = int(D.argmin())
    best_delta = float(D.flat[flat])
    if best_delta < -EPS:
        a, b = divmod(flat, len(jpos))
        return best_delta, int(idx1[a]), int(jpos[b])
    return 0.0, -1, -1


def _pair_moves(r1, r2, adj, demands, capacity, clss, variant, load1, load2):
    """Fused per-pair inter-route move search.

    Computes, in one call sharing every gather, the same three results the
    separate searches produce for the pair (r1, r2):

        (best swap r1<->r2,  best relocate r1->r2,  best relocate r2->r1)

    each as ``(delta, i, j)`` with the exact formulas / argmin tie-breaking of
    ``best_swap_inter`` / ``best_relocate_inter`` (values are bit-identical:
    identical expressions evaluated in the same elementwise order).
    """
    no_move = (0.0, -1, -1)
    swap = rel12 = rel21 = no_move
    n1, n2 = len(r1), len(r2)
    have1 = n1 > 2                     # r1 has at least one interior arc
    have2 = n2 > 2

    # Slice views (no fancy-index copies): interior arcs and their neighbours.
    # base_k[j] = adj[r_k[j], r_k[j+1]] is the route's edge-cost vector; the
    # per-arc in/out edge costs are exactly its slices (identical adj entries,
    # so all deltas below are bit-identical to the standalone searches).
    if have1:
        x = r1[1:-1]
        a1 = r1[:-2]
        a2 = r1[2:]
        dx = demands[x]
        base1 = adj[r1[:-1], r1[1:]]
        adj_a1x = base1[:-1]           # adj[a1, x]
        adj_xa2 = base1[1:]            # adj[x, a2]
        cx = clss[x] if variant == 'P' else None
    if have2:
        y = r2[1:-1]
        b1 = r2[:-2]
        b2 = r2[2:]
        dy = demands[y]
        base2 = adj[r2[:-1], r2[1:]]
        adj_b1y = base2[:-1]           # adj[b1, y]
        adj_yb2 = base2[1:]            # adj[y, b2]
        cy = clss[y] if variant == 'P' else None

    # ---- swap r1 <-> r2 (cf. best_swap_inter) ----
    if have1 and have2:
        new = (adj[a1[:, None], y[None, :]] + adj[y[None, :], a2[:, None]]
               + adj[b1[None, :], x[:, None]] + adj[x[:, None], b2[None, :]])
        old_i = adj_a1x + adj_xa2
        old_j = adj_b1y + adj_yb2
        D = new - (old_i[:, None] + old_j[None, :])
        feas = (load1 - dx[:, None] + dy[None, :] <= capacity + EPS) & \
               (load2 - dy[None, :] + dx[:, None] <= capacity + EPS)
        if variant == 'P':
            feas &= cx[:, None] == cy[None, :]
        D[~feas] = np.inf
        flat = int(D.argmin())
        bd = float(D.flat[flat])
        if bd < -EPS:
            a, b = divmod(flat, n2 - 2)
            swap = (bd, a + 1, b + 1)  # interior offset -> route position

    # ---- relocate r1 -> r2 (cf. best_relocate_inter) ----
    if have1 and n2 >= 2:
        rem = adj[a1, a2] - adj_a1x - adj_xa2
        u = r2[:-1]
        v = r2[1:]
        ins = (adj[u[None, :], x[:, None]] + adj[x[:, None], v[None, :]]
               - base2[None, :]) if have2 else \
              (adj[u[None, :], x[:, None]] + adj[x[:, None], v[None, :]]
               - adj[u, v][None, :])
        D = rem[:, None] + ins
        cap_ok = load2 + dx <= capacity + EPS
        D[~cap_ok, :] = np.inf
        if variant == 'P':
            D[~(cx[:, None] == clss[u][None, :])] = np.inf
        flat = int(D.argmin())
        bd = float(D.flat[flat])
        if bd < -EPS:
            a, b = divmod(flat, n2 - 1)
            rel12 = (bd, a + 1, int(b))

    # ---- relocate r2 -> r1 (symmetric) ----
    if have2 and n1 >= 2:
        rem = adj[b1, b2] - adj_b1y - adj_yb2
        u = r1[:-1]
        v = r1[1:]
        ins = (adj[u[None, :], y[:, None]] + adj[y[:, None], v[None, :]]
               - base1[None, :]) if have1 else \
              (adj[u[None, :], y[:, None]] + adj[y[:, None], v[None, :]]
               - adj[u, v][None, :])
        D = rem[:, None] + ins
        cap_ok = load1 + dy <= capacity + EPS
        D[~cap_ok, :] = np.inf
        if variant == 'P':
            D[~(cy[:, None] == clss[u][None, :])] = np.inf
        flat = int(D.argmin())
        bd = float(D.flat[flat])
        if bd < -EPS:
            a, b = divmod(flat, n1 - 1)
            rel21 = (bd, a + 1, int(b))

    return swap, rel12, rel21


def inter_route_opt(routes, adj, demands, clss, capacity=1.0, variant='P', max_iter=100):
    """Apply the best improving inter-route move across all route pairs.

    Greedily picks, over every route pair, the single best feasible improving
    move -- a swap (exchange two arcs) or a relocate (move one arc from one
    route into another) -- and applies it; repeats until no pair yields an
    improvement (or ``max_iter`` reached). The objective minimised here is the
    total deadheading cost; capacity is never violated.

    Each pass computes all pair moves once via the fused ``_pair_moves``
    (values identical to the standalone searches), then compares candidates in
    the exact historical order -- all swaps by (ri, rj), then all relocates by
    (ra, rb) row-major -- so tie-breaking between equal-delta moves is
    unchanged.
    """
    routes = [np.asarray(r).copy() for r in routes]
    nr = len(routes)
    for _ in range(max_iter):
        loads = [demands[r].sum() for r in routes]
        swap_res = {}
        rel_res = {}
        for ri in range(nr):
            for rj in range(ri + 1, nr):
                s, r12, r21 = _pair_moves(
                    routes[ri], routes[rj], adj, demands, capacity, clss,
                    variant, loads[ri], loads[rj],
                )
                swap_res[(ri, rj)] = s
                rel_res[(ri, rj)] = r12
                rel_res[(rj, ri)] = r21

        best_delta = 0.0
        best_move = None                     # (kind, ra, rb, i, j)
        for ri in range(nr):
            for rj in range(ri + 1, nr):
                d, i, j = swap_res[(ri, rj)]
                if i >= 0 and d < best_delta - EPS:
                    best_delta, best_move = d, ('swap', ri, rj, i, j)
        for ra in range(nr):
            for rb in range(nr):
                if ra == rb:
                    continue
                d, i, j = rel_res[(ra, rb)]
                if i >= 0 and d < best_delta - EPS:
                    best_delta, best_move = d, ('relocate', ra, rb, i, j)
        if best_move is None:
            break
        kind, ra, rb, i, j = best_move
        if kind == 'swap':
            routes[ra][i], routes[rb][j] = routes[rb][j], routes[ra][i]
        else:
            x = routes[ra][i]
            routes[rb] = np.insert(routes[rb], j + 1, x)
            routes[ra] = np.delete(routes[ra], i)
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
    from utils.ops import gen_tours

    adj = vars['adj']
    adj = adj.numpy() if hasattr(adj, 'numpy') else np.asarray(adj)
    adj = adj.astype(np.float64, copy=False)
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
    adj = adj.astype(np.float64, copy=False)
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
