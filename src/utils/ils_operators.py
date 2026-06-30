"""Iterated Local Search (ILS) operators for HDCARP.

Pure, self-contained utilities — they import only from ``numpy`` and
``utils.ops`` (no dependency on ``solvers/``). Three independent pieces:

  * :func:`perturbate` — a feasibility-preserving *random relocate* kick that
    moves ``strength`` arcs between routes (capacity- and, for variant ``'P'``,
    priority-aware).
  * :func:`accept`     — pure acceptance criterion (``'best'`` improvement-only
    or ``'sa'`` simulated annealing with a geometric cooling schedule).
  * :func:`routes_to_action` — flatten a route-list into a 0-separated action
    array (thin, explicit wrapper around ``deserialize_tours``).

Conventions match ``utils/local_search.py``: a route is a 1-D integer array with
sentinel depot 0s at both ends, e.g. ``[0, a1, a2, 0]``; arc 0 is the depot
(demand = 0, clss = 0). ``capacity`` is normalised to 1.0 (demands are divided
by Q in the data).
"""

import numpy as np

from utils.ops import deserialize_tours

EPS = 1e-9


# --------------------------------------------------------------------------- #
# 1a. perturbation
# --------------------------------------------------------------------------- #
def _route_interior(route):
    """Strip sentinels and return the interior arcs as a list of python ints."""
    r = np.asarray(route).astype(np.int64)
    if len(r) <= 2:
        return []
    return [int(a) for a in r[1:-1]]


def _interior_load(interior, demands):
    """Total normalised demand of the arcs in ``interior``."""
    if not interior:
        return 0.0
    return float(np.asarray(demands)[interior].sum())


def _insert_arc(interior, arc, clss, variant, rng):
    """Insert ``arc`` into ``interior`` in place, respecting the variant rule.

    ``'P'``: append then stable-sort the whole interior by class so the route
    stays non-decreasing in priority class (stable => same-class arcs keep their
    relative order). ``'U'``: insert at a random interior position.
    """
    if variant == 'P':
        interior.append(int(arc))
        # stable sort by class -> priority order preserved, ties keep order
        interior.sort(key=lambda a: int(clss[a]))
    else:
        pos = int(rng.integers(0, len(interior) + 1))
        interior.insert(pos, int(arc))


def perturbate(routes, adj, demands, clss, capacity, variant, rng, strength=3):
    """Random-relocate perturbation that preserves feasibility and arc identity.

    Relocates ``strength`` random non-depot arcs out of their source routes into
    feasible positions of *other* routes. Capacity is enforced (a destination is
    only eligible if it can absorb the arc's demand); for variant ``'P'`` the
    destination's interior is re-sorted by class so priority order is kept. If an
    arc has no feasible foreign destination it is reinserted into its source.

    Parameters
    ----------
    routes : list of 1-D int arrays ``[0, ..., 0]``  (NOT mutated)
    adj : arc-to-arc cost matrix (unused here; kept for a uniform operator API)
    demands : 1-D float array, ``demands[arc]`` normalised load (``demands[0]=0``)
    clss : 1-D int array of priority classes (``clss[0]=0``)
    capacity : per-route capacity (1.0)
    variant : ``'P'`` or ``'U'``
    rng : ``numpy.random.Generator``
    strength : number of arcs to relocate (>= 0)

    Returns
    -------
    list of 1-D int32 arrays — a *new* route list (input untouched) holding
    exactly the same multiset of non-depot arcs as the input.
    """
    demands = np.asarray(demands)
    clss = np.asarray(clss)
    interiors = [_route_interior(r) for r in routes]
    n = len(interiors)

    for _ in range(int(strength)):
        src_candidates = [i for i in range(n) if interiors[i]]
        if not src_candidates:
            break
        si = int(rng.choice(src_candidates))
        ai = int(rng.integers(0, len(interiors[si])))
        arc = interiors[si].pop(ai)
        d = float(demands[arc])

        # feasible foreign destinations (capacity-respecting)
        dst_candidates = [
            j for j in range(n)
            if j != si and _interior_load(interiors[j], demands) + d <= capacity + EPS
        ]
        if dst_candidates:
            dj = int(rng.choice(dst_candidates))
            _insert_arc(interiors[dj], arc, clss, variant, rng)
        else:
            # no room elsewhere -> put it back into its own route
            _insert_arc(interiors[si], arc, clss, variant, rng)

    return [np.array([0] + interior + [0], dtype=np.int32) for interior in interiors]


# --------------------------------------------------------------------------- #
# 1b. acceptance criterion
# --------------------------------------------------------------------------- #
def accept(current_obj, candidate_obj, iteration, max_iter,
           mode='best', T_init=0.05, T_final=0.001):
    """Decide whether to move from ``current`` to ``candidate`` (higher = better).

    ``mode='best'`` : accept iff ``candidate_obj >= current_obj`` (ties accepted
    to escape stagnation). ``mode='sa'`` : always accept improvements; accept a
    worsening move with probability ``exp((candidate - current) / T)`` where the
    temperature follows a geometric schedule
    ``T = T_init * (T_final / T_init) ** (iteration / max_iter)``.

    Pure w.r.t. its arguments (the only state touched is the global numpy RNG for
    the SA coin flip).
    """
    if candidate_obj >= current_obj:
        return True
    if mode != 'sa':
        return False

    if max_iter <= 0:
        T = T_final
    else:
        T = T_init * (T_final / T_init) ** (iteration / max_iter)
    if T <= 0:
        return False

    prob = np.exp((candidate_obj - current_obj) / T)
    return bool(np.random.random() < prob)


# --------------------------------------------------------------------------- #
# 1c. serialisation helper
# --------------------------------------------------------------------------- #
def routes_to_action(routes, nseq):
    """Flatten a route-list into a 0-separated action array of length ``nseq``.

    Thin, explicit re-export of ``utils.ops.deserialize_tours`` for use from
    the ILS solver.
    """
    return deserialize_tours(routes, nseq)
