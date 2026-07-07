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

    each as ``(delta, i, j)`` with the exact per-move delta formulas of
    ``calc_swap_delta_inter`` / ``calc_relocate_delta_inter`` and the
    row-major argmin tie-breaking of ``best_relocate_inter`` (values are
    bit-identical: identical expressions evaluated in the same elementwise
    order).
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

    # ---- swap r1 <-> r2 (cf. calc_swap_delta_inter) ----
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
# lex-safe U search (move-level acceptance)
# --------------------------------------------------------------------------- #
def _u_improving_moves(routes, adj, demands, capacity, loads):
    """Enumerate EVERY structurally-legal, deadhead-improving U move across
    ``routes``, sorted best (most negative) delta first.

    Structural legality is exactly what the deadhead-greedy U kernels allow:
    any-class swap/relocate, intra- and inter-route, with inter-route capacity
    checked up front and no-op relocate targets excluded. The deltas are the
    same edge-cost expressions the greedy kernels use (see
    ``calc_swap_delta_intra`` etc.); here they are only a RANKING -- the
    caller decides acceptance separately.

    Returns a list of ``(delta, kind, ta, tb, i, j)``:
      - ``('iswap', t, t, i, j)``: swap positions ``i < j`` within ``routes[t]``
      - ``('irel',  t, t, i, j)``: move ``routes[t][i]`` to after position ``j``
      - ``('xswap', ta, tb, i, j)``: swap ``routes[ta][i]`` with ``routes[tb][j]``
      - ``('xrel',  ta, tb, i, j)``: move ``routes[ta][i]`` into ``routes[tb]``,
        inserted after position ``j``
    """
    cands = []
    nr = len(routes)

    # ---- intra-route moves ----
    for t in range(nr):
        r = routes[t]
        n = len(r)
        m = n - 2
        if m < 1:
            continue
        x = r[1:-1]
        prev = r[:-2]
        nxt = r[2:]
        base = adj[r[:-1], r[1:]]
        e_in = base[:-1]
        e_out = base[1:]

        if m >= 2:
            A1 = adj[prev[:, None], x[None, :]]
            A2 = adj[x[None, :], nxt[:, None]]
            B1 = adj[prev[None, :], x[:, None]]
            B2 = adj[x[:, None], nxt[None, :]]
            old = e_in + e_out
            D = (A1 + A2 + B1 + B2) - (old[:, None] + old[None, :])
            k = np.arange(m - 1)                      # adjacent pairs
            D[k, k + 1] = (adj[prev[k], x[k + 1]] + adj[x[k + 1], x[k]]
                           + adj[x[k], nxt[k + 1]]) - \
                          (adj[prev[k], x[k]] + adj[x[k], x[k + 1]]
                           + adj[x[k + 1], nxt[k + 1]])
            rr = np.arange(m)
            D[rr[:, None] >= rr[None, :]] = np.inf
            for a, b in np.argwhere(D < -EPS):
                cands.append((float(D[a, b]), 'iswap', t, t,
                              int(a) + 1, int(b) + 1))

        rem = adj[prev, nxt] - e_in - e_out
        u = r[:-1]
        v = r[1:]
        D = rem[:, None] + adj[u[None, :], x[:, None]] \
            + adj[x[:, None], v[None, :]] - base[None, :]
        ar = np.arange(m)
        D[ar, ar + 1] = np.inf                        # j == i     (no-op)
        D[ar, ar] = np.inf                            # j == i - 1 (no-op)
        for a, b in np.argwhere(D < -EPS):
            cands.append((float(D[a, b]), 'irel', t, t, int(a) + 1, int(b)))

    # ---- inter-route moves ----
    for ta in range(nr):
        for tb in range(ta + 1, nr):
            ra, rb = routes[ta], routes[tb]
            ma, mb = len(ra) - 2, len(rb) - 2

            if ma >= 1 and mb >= 1:                   # swap ta <-> tb
                x = ra[1:-1]
                a1 = ra[:-2]
                a2 = ra[2:]
                y = rb[1:-1]
                b1 = rb[:-2]
                b2 = rb[2:]
                new = (adj[a1[:, None], y[None, :]] + adj[y[None, :], a2[:, None]]
                       + adj[b1[None, :], x[:, None]] + adj[x[:, None], b2[None, :]])
                old_i = adj[a1, x] + adj[x, a2]
                old_j = adj[b1, y] + adj[y, b2]
                D = new - (old_i[:, None] + old_j[None, :])
                dx = demands[x]
                dy = demands[y]
                feas = (loads[ta] - dx[:, None] + dy[None, :] <= capacity + EPS) & \
                       (loads[tb] - dy[None, :] + dx[:, None] <= capacity + EPS)
                D[~feas] = np.inf
                for a, b in np.argwhere(D < -EPS):
                    cands.append((float(D[a, b]), 'xswap', ta, tb,
                                  int(a) + 1, int(b) + 1))

            for src, dst in ((ta, tb), (tb, ta)):     # relocate src -> dst
                rs, rd = routes[src], routes[dst]
                if len(rs) <= 2 or len(rd) < 2:
                    continue
                x = rs[1:-1]
                p = rs[:-2]
                q = rs[2:]
                rem = adj[p, q] - adj[p, x] - adj[x, q]
                u = rd[:-1]
                v = rd[1:]
                D = rem[:, None] + adj[u[None, :], x[:, None]] \
                    + adj[x[:, None], v[None, :]] - adj[u, v][None, :]
                cap_ok = loads[dst] + demands[x] <= capacity + EPS
                D[~cap_ok, :] = np.inf
                for a, b in np.argwhere(D < -EPS):
                    cands.append((float(D[a, b]), 'xrel', src, dst,
                                  int(a) + 1, int(b)))

    cands.sort(key=lambda c: c[0])
    return cands


def u_route_opt_lex_safe(routes, owner, n_vehicles, adj, demands, veh_score,
                         capacity=1.0, max_iter=200):
    """Move-level lex-safe unrestricted (variant-U) local search.

    Motivation (measured, data/test_bm/small.npz, production RL-P config):
    running the deadhead-greedy U kernels (``intra_route_opt`` +
    ``inter_route_opt``, variant='U') to convergence and lex-guarding only
    the BUNDLED end result fired 0/100 times -- the pass finds real deadhead
    reductions (mean -11.94/instance) but always includes at least one
    T1-damaging move, so the whole bundle is rejected even though most of its
    moves were T1-neutral T2/T3 wins. This search fixes that by moving the
    lexicographic acceptance check to the INDIVIDUAL move.

    Each iteration:
      1. enumerate all deadhead-improving candidate moves (same move families
         and capacity feasibility as the greedy U kernels -- structural
         legality is unchanged), ranked by deadhead delta;
      2. walk the ranking and apply the FIRST candidate whose full-solution
         objective tuple does not lex-worsen (i.e. the best-deadhead lex-SAFE
         move); the tuple is evaluated exactly, per touched vehicle, via
         ``veh_score`` -- a move only alters the vehicle(s) owning the touched
         route(s), so untouched vehicles' cached contributions are reused and
         the check costs O(touched-vehicle arcs);
      3. stop when no candidate is lex-safe (or nothing improves deadhead).

    Guarantees (by construction):
      - every accepted move is lex-non-worsening, so the objective tuple --
        and in particular T1 -- is monotone non-increasing over the whole
        pass; the final solution is lex <= the input;
      - termination: every accepted move strictly reduces total deadheading
        (bounded below), ``max_iter`` is only a safety cap;
      - feasibility: identical structural legality to the greedy U kernels
        (arcs are only permuted/relocated, single service preserved, inter-
        route capacity checked before ranking).

    Why accept lex-EQUAL moves (not only strict improvements): a deadhead
    reduction that leaves (T1, T2, T3) untouched (e.g. on a route that does
    not carry any class's max completion) is free progress and frequently
    unlocks strictly-improving moves later; cycling is impossible because
    deadhead still strictly decreases.

    Parameters
    ----------
    routes     : list of sentinel routes ``[0, a1, ..., ak, 0]`` (the flat
                 trip list; multiple trips may belong to one vehicle).
    owner      : vehicle index per route (len == len(routes)).
    n_vehicles : total vehicle count M.
    adj        : arc-to-arc deadheading cost matrix (float64).
    demands    : per-arc demand vector.
    veh_score  : callable(list_of_trip_interiors) -> (P,) array: ONE vehicle's
                 per-class max completion times (e.g. a closure over
                 ``SolutionEvaluator._completion_times([veh_trips], td)``).
                 The full-solution objective is the elementwise max over
                 vehicles, which matches scoring all vehicles at once.
    capacity   : per-trip capacity (demands are normalized to it).
    max_iter   : safety cap on accepted moves.

    Returns the optimized routes (same list structure/owners; a route may
    become empty ``[0, 0]`` after relocates -- callers drop empty trips).
    """
    routes = [np.asarray(r).astype(np.int64).copy() for r in routes]
    owner = [int(m) for m in owner]

    def _veh_trips(m, override=None):
        trips = []
        for t, r in enumerate(routes):
            if owner[t] != m:
                continue
            rr = r if override is None or t not in override else override[t]
            if len(rr) > 2:
                trips.append(rr[1:-1])
        return trips

    contrib = np.stack([np.asarray(veh_score(_veh_trips(m)), dtype=np.float64)
                        for m in range(n_vehicles)])
    cur_T = contrib.max(axis=0)

    for _ in range(max_iter):
        loads = [demands[r].sum() for r in routes]
        cands = _u_improving_moves(routes, adj, demands, capacity, loads)
        accepted = False
        for delta, kind, ta, tb, i, j in cands:
            override = {}
            if kind == 'iswap':
                nr_ = routes[ta].copy()
                nr_[i], nr_[j] = nr_[j], nr_[i]
                override[ta] = nr_
            elif kind == 'irel':
                override[ta] = apply_relocate_intra(routes[ta], i, j)
            elif kind == 'xswap':
                na_ = routes[ta].copy()
                nb_ = routes[tb].copy()
                na_[i], nb_[j] = nb_[j], na_[i]
                override[ta] = na_
                override[tb] = nb_
            else:                                     # 'xrel'
                xarc = routes[ta][i]
                override[tb] = np.insert(routes[tb], j + 1, xarc)
                override[ta] = np.delete(routes[ta], i)

            new_contrib = contrib.copy()
            for m in sorted({owner[t] for t in override}):
                new_contrib[m] = np.asarray(
                    veh_score(_veh_trips(m, override)), dtype=np.float64)
            new_T = new_contrib.max(axis=0)
            if tuple(new_T.tolist()) <= tuple(cur_T.tolist()):
                for t, rr in override.items():
                    routes[t] = rr
                contrib = new_contrib
                cur_T = new_T
                accepted = True
                break
        if not accepted:
            break
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
    result to ``deserialize_tours``.

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
