"""OSM instance generation utilities for HDCARP training data.

Defines CITIES, BUCKETS, CURRICULUM constants and gen_city_bucket(),
the single entry-point for generating one (city, bucket) shard.

Output layout (see docs/strategy_data.md §4):
    data/osm_train/<city>/<bucket>.npz   — training
    data/osm_test/<city>/<bucket>.npz    — held-out test
    data/benchmark/<|V|>_<|A|>/*.npz     — Ha et al. (2024) benchmark
"""

import os
import random
import shutil
import tempfile

import numpy as np
import networkx as nx

# --------------------------------------------------------------------------- #
# City catalogue  (N, S, E, W bounding boxes)
# --------------------------------------------------------------------------- #
CITIES = {
    "new_york":      (40.7600, 40.7420, -73.9710, -73.9930),
    "chicago":       (41.8900, 41.8720, -87.6190, -87.6420),
    "boston":        (42.3680, 42.3500, -71.0520, -71.0760),
    "toronto":       (43.6610, 43.6430, -79.3730, -79.3960),
    "mexico_city":   (19.4350, 19.4170, -99.1500, -99.1730),
    "san_francisco": (37.7950, 37.7770,-122.3950,-122.4180),
    "buenos_aires":  (-34.5950,-34.6130, -58.3680, -58.3910),
    "sao_paulo":     (-23.5400,-23.5580, -46.6280, -46.6510),
    "bogota":        ( 4.6080,  4.5900, -74.0650, -74.0870),
    "london":        (51.5200, 51.5050,  -0.0900,  -0.1230),
    "paris":         (48.8650, 48.8500,   2.3640,   2.3380),
    "rome":          (41.9050, 41.8900,  12.4900,  12.4660),
    "barcelona":     (41.3960, 41.3810,   2.1750,   2.1530),
    "amsterdam":     (52.3760, 52.3620,   4.9100,   4.8830),
    "berlin":        (52.5260, 52.5110,  13.4090,  13.3830),
    "prague":        (50.0920, 50.0780,  14.4320,  14.4090),
    "cairo":         (30.0560, 30.0410,  31.2470,  31.2300),
    "lagos":         ( 6.4570,  6.4420,   3.4030,   3.3870),
    "marrakesh":     (31.6350, 31.6210,  -7.9810,  -7.9970),
    "nairobi":       (-1.2790, -1.2930,  36.8290,  36.8130),
    "tokyo":         (35.7000, 35.6850, 139.7110, 139.6920),
    "hanoi":         (21.0400, 21.0260, 105.8580, 105.8410),
    "mumbai":        (18.9300, 18.9150,  72.8390,  72.8230),
    "singapore":     ( 1.2890,  1.2750, 103.8530, 103.8380),
    "shanghai":      (31.2350, 31.2200, 121.4880, 121.4710),
    "chandigarh":    (30.7460, 30.7320,  76.7900,  76.7720),
    "sydney":        (-33.8630,-33.8780, 151.2150, 151.1990),
    "melbourne":     (-37.8080,-37.8230, 144.9740, 144.9560),
}

# --------------------------------------------------------------------------- #
# Bucket catalogue  (±5% arc tolerance, m_nominal=3 aligns with Ha et al.)
# --------------------------------------------------------------------------- #
BUCKETS = {
    # d = 1.5 — organic cities (out-degree < 1.9)
    "20_30":  dict(num_loc=20, min_arc=28,  max_arc=32),   # |A_r|=21
    "30_45":  dict(num_loc=30, min_arc=43,  max_arc=47),   # |A_r|=33
    "40_60":  dict(num_loc=40, min_arc=57,  max_arc=63),   # |A_r|=45
    "50_75":  dict(num_loc=50, min_arc=71,  max_arc=79),   # |A_r|=54
    "80_120": dict(num_loc=80, min_arc=114, max_arc=126),  # |A_r|=90
    # d = 2.0 — grid/mixed cities (out-degree 1.9–2.4)
    "20_40":  dict(num_loc=20, min_arc=38,  max_arc=42),   # |A_r|=30
    "30_60":  dict(num_loc=30, min_arc=57,  max_arc=63),   # |A_r|=45
    "40_80":  dict(num_loc=40, min_arc=76,  max_arc=84),   # |A_r|=60
    "50_100": dict(num_loc=50, min_arc=95,  max_arc=105),  # |A_r|=75
    "80_133": dict(num_loc=80, min_arc=126, max_arc=140),  # |A_r|=99
}

# --------------------------------------------------------------------------- #
# Curriculum phases  (keyed by |A_r| breakpoints)
# --------------------------------------------------------------------------- #
CURRICULUM = {
    "small":  ["20_30", "20_40"],
    "medium": ["30_45", "30_60", "40_60", "50_75", "40_80"],
    "large":  ["50_100", "80_120", "80_133"],
}

# --------------------------------------------------------------------------- #
# Density tier routing  (organic cities can't reach d=2.0 arc densities)
# --------------------------------------------------------------------------- #
# "d15" cities generate only d=1.5 buckets (20_30, 30_45, 40_60, 50_75).
# "d20" cities generate all 8 buckets (d=1.5 + d=2.0).
CITY_TIERS = {
    "d15": {"boston", "sao_paulo", "london", "paris", "rome", "amsterdam",
            "prague", "cairo", "marrakesh", "hanoi", "mumbai", "barcelona"},
    "d20": {"new_york", "chicago", "toronto", "mexico_city", "san_francisco",
            "buenos_aires", "bogota", "berlin", "lagos", "nairobi", "tokyo",
            "singapore", "shanghai", "chandigarh", "sydney", "melbourne"},
}

_D15_BUCKETS = ["20_30", "30_45", "40_60", "50_75", "80_120"]
_D20_BUCKETS = list(BUCKETS.keys())  # all 10


def eligible_buckets(city: str) -> list:
    """Return the bucket keys this city's density tier can feasibly generate."""
    if city in CITY_TIERS["d15"]:
        return _D15_BUCKETS
    if city in CITY_TIERS["d20"]:
        return _D20_BUCKETS
    raise KeyError(f"City '{city}' not found in CITY_TIERS. Add it to data_utils.py.")

M_NOMINAL = 3  # aligns with Ha et al. (2024) benchmark


# --------------------------------------------------------------------------- #
# Instance physics  (paper F2–F5, pure numpy)
# --------------------------------------------------------------------------- #
def required_count(num_arc):
    """Paper F2: |A_r| = 3*floor(|A|/4)."""
    return 3 * (num_arc // 4)


def build_instance(edges, coords, M, rng=np.random):
    """Build an HDCARP instance from a directed graph.

    edges:  (|A|, 2) int array of (tail, head) node indices
    coords: (n, 2)   float array of projected (x, y) positions
    M:      nominal fleet size (sets C = Σq/M + 0.5)
    Returns (req, nonreq, C). Raises ValueError on degenerate input.
    """
    edges  = np.asarray(edges, dtype=int)
    coords = np.asarray(coords, dtype=np.float64)
    num_arc = len(edges)

    d_eucl = np.sqrt(((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2).sum(-1))
    d_max  = d_eucl.max()
    if d_max <= 0:
        raise ValueError("degenerate graph: all arc lengths are zero")
    d = d_eucl / d_max

    n_req = required_count(num_arc)
    if n_req >= num_arc:
        raise ValueError("not enough arcs to split required / non-required")

    idx     = np.arange(num_arc)
    idx_req = rng.choice(idx, size=n_req, replace=False)
    tries   = 0
    while 0 not in edges[idx_req, 0]:
        idx_req = rng.choice(idx, size=n_req, replace=False)
        tries  += 1
        if tries > 1000:
            raise ValueError("could not place depot on a required arc")
    idx_nonreq = np.setdiff1d(idx, idx_req)

    e_req, e_nonreq = edges[idx_req], edges[idx_nonreq]
    d_req, d_nonreq = d[idx_req],     d[idx_nonreq]

    s_req = 2.0 * d_req
    q_req = 0.5 * d_req + 0.5

    per_class = n_req // 3
    clss = np.concatenate([
        np.repeat([1, 2, 3], per_class),
        np.full(n_req - 3 * per_class, 3),
    ])
    rng.shuffle(clss)

    C   = float(q_req.sum() / M + 0.5)
    req = np.column_stack([e_req,    q_req, clss, s_req,         d_req]).astype(np.float64)
    z   = np.zeros(len(e_nonreq))
    nonreq = np.column_stack([e_nonreq, z,     z,    z,    d_nonreq]).astype(np.float64)
    return req, nonreq, C


# --------------------------------------------------------------------------- #
# OSM graph helpers  (require osmnx)
# --------------------------------------------------------------------------- #
def load_osm_graph(north, south, east, west):
    """Download (or load from cache) and project an OSM drive graph."""
    import osmnx as ox
    ox.settings.cache_folder = "data/cache"
    try:
        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west,
                               network_type="drive")
    except TypeError:
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    return ox.project_graph(G)


def _largest_scc(G):
    """Return a copy of the subgraph induced by G's largest strongly-connected
    component, so downstream sampling always works on a connected base graph."""
    return G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()


def _random_connected_subgraph(G, num_nodes, tol=2, max_attempts=500):
    """Return a strongly-connected subgraph with roughly ``num_nodes`` nodes.

    The old implementation took the first ``num_nodes`` BFS nodes and re-rolled
    until ``is_strongly_connected`` — but a truncated BFS frontier is almost
    never strongly connected, and degenerate 1-node subgraphs pass the check
    trivially, so it returned (near-)empty subgraphs that the caller then
    rejected forever. Instead we grow a BFS window from a random start and keep
    the window's largest strongly-connected component, accepting the first one
    whose size lands in ``[num_nodes - tol, num_nodes + tol]``.

    ``G`` is assumed to already be strongly connected (see ``_largest_scc``).
    """
    nodes_list = list(G.nodes)
    fallback = None
    for _ in range(max_attempts):
        start = random.choice(nodes_list)
        bfs = list(nx.bfs_tree(G, start).nodes)
        if len(bfs) < num_nodes - tol:
            continue
        for k in range(max(1, num_nodes - tol), min(len(bfs), num_nodes * 3) + 1):
            scc = max(nx.strongly_connected_components(G.subgraph(bfs[:k])), key=len)
            if len(scc) >= num_nodes - tol:
                if len(scc) <= num_nodes + tol:
                    return G.subgraph(scc).copy()
                fallback = fallback or scc  # overshoot; remember in case nothing better
                break
    if fallback is not None:
        return G.subgraph(fallback).copy()
    raise RuntimeError(
        f"could not sample a strongly-connected subgraph of ~{num_nodes} nodes "
        f"after {max_attempts} attempts (base graph |V|={G.number_of_nodes()})"
    )


def _trim_edges(edges, max_arc, rng=np.random):
    """Randomly drop edges until len(edges) <= max_arc, preserving strong connectivity."""
    if len(edges) <= max_arc:
        return edges
    order = rng.permutation(len(edges)).tolist()
    keep  = set(order)
    for i in order:
        if len(keep) <= max_arc:
            break
        keep.discard(i)
        g = nx.DiGraph()
        g.add_edges_from(edges[sorted(keep)].tolist())
        if not nx.is_strongly_connected(g):
            keep.add(i)
    return edges[sorted(keep)]


def _sample_instance(G_proj, target_nodes, M, save_dir, min_arc, max_arc,
                     rng=np.random, max_tries=2000):
    """Sample one strongly-connected subgraph, trim to arc range, write a .npz.

    ``G_proj`` is assumed strongly connected (restrict via ``_largest_scc`` once
    in the caller). Returns (path, num_arc, num_nodes). Raises RuntimeError if
    no valid instance is found within ``max_tries`` — this signals an infeasible
    city/bucket pairing (e.g. an organic, low-out-degree city against a
    high-arc-density bucket) instead of looping forever.
    """
    last_arc = -1
    for _ in range(max_tries):
        sub      = _random_connected_subgraph(G_proj, target_nodes)
        node_map = {v: i for i, v in enumerate(sub.nodes())}
        coords   = np.array([[sub.nodes[v]["x"], sub.nodes[v]["y"]] for v in node_map],
                            dtype=np.float64)
        edges    = np.array([[node_map[u], node_map[v]] for u, v in sub.edges()], dtype=int)
        if len(edges) == 0:
            continue

        edges = _trim_edges(edges, max_arc, rng=rng)
        last_arc = len(edges)
        if len(edges) < min_arc:
            continue

        g = nx.DiGraph()
        g.add_edges_from(edges.tolist())
        if not nx.is_strongly_connected(g):
            continue

        try:
            req, nonreq, C = build_instance(edges, coords, M, rng=rng)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError(
            f"no valid instance after {max_tries} tries "
            f"(target_nodes={target_nodes}, min_arc={min_arc}, max_arc={max_arc}, "
            f"last trimmed |A|={last_arc}). Likely an infeasible city/bucket pairing."
        )

    num_arc   = len(req) + len(nonreq)
    num_nodes = len(coords)
    tau       = float(req[:, 2].sum() / (M * C))
    d_density = num_arc / num_nodes
    fpath     = f"{save_dir}/{num_arc}_{num_nodes}_{rng.randint(0, 1_000_000):06d}"
    np.savez(fpath, req=req, nonreq=nonreq, P=3, M=M, C=C,
             d=d_density, topology="osm", tau=tau, n_req=len(req))
    return fpath + ".npz", num_arc, num_nodes


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def gen_city_bucket(
    city: str,
    bucket: str,
    per_bucket: int,
    seed: int,
    out_dir: str = "data/osm_train",
) -> str:
    """Generate one (city, bucket) shard and save as a bundled .npz.

    Args:
        city:       Key in CITIES, e.g. "paris".
        bucket:     Key in BUCKETS, e.g. "20_40".
        per_bucket: Number of accepted instances to generate.
        seed:       RNG seed. Recommended: base_seed ^ hash((city, bucket)) & 0xFFFFFFFF
        out_dir:    Root output dir. File written to <out_dir>/<city>/<bucket>.npz

    Returns:
        Absolute path of the written .npz file.

    Raises:
        KeyError:   Unknown city or bucket.
        ValueError: City graph too small for this bucket.
    """
    if city not in CITIES:
        raise KeyError(f"Unknown city '{city}'. Available: {sorted(CITIES)}")
    if bucket not in BUCKETS:
        raise KeyError(f"Unknown bucket '{bucket}'. Available: {sorted(BUCKETS)}")

    np.random.seed(seed)
    random.seed(seed)

    cfg     = BUCKETS[bucket]
    num_loc = cfg["num_loc"]
    min_arc = cfg["min_arc"]
    max_arc = cfg["max_arc"]
    max_req = 3 * (max_arc // 4)

    N, S, E, W = CITIES[city]
    G_proj = load_osm_graph(N, S, E, W)
    G_proj = _largest_scc(G_proj)  # sample only from a strongly-connected base

    if G_proj.number_of_nodes() < num_loc:
        raise ValueError(
            f"{city}: graph has {G_proj.number_of_nodes()} nodes, "
            f"bucket '{bucket}' needs num_loc={num_loc}."
        )

    out_path = os.path.join(out_dir, city, f"{bucket}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp = tempfile.mkdtemp()
    reqs, nonreqs, Cs, Ms, n_reqs, taus = [], [], [], [], [], []
    print(f"[gen] {city}/{bucket}: target={per_bucket} "
          f"|V|={num_loc} |A|∈[{min_arc},{max_arc}]")

    try:
        consecutive_fails = 0
        while len(reqs) < per_bucket:
            try:
                fpath, n_arc, n_loc = _sample_instance(
                    G_proj, min(num_loc, G_proj.number_of_nodes()), M_NOMINAL, tmp,
                    min_arc=min_arc, max_arc=max_arc,
                )
                consecutive_fails = 0
            except RuntimeError as e:
                consecutive_fails += 1
                if consecutive_fails >= 5:
                    raise RuntimeError(
                        f"{city}/{bucket}: {consecutive_fails} consecutive failures — "
                        f"likely infeasible pairing. Got {len(reqs)}/{per_bucket}."
                    ) from e
                np.random.seed(seed ^ len(reqs) ^ consecutive_fails)
                random.seed(seed ^ len(reqs) ^ consecutive_fails)
                print(f"[gen] warning: {e} — re-seeding (attempt {consecutive_fails})")
                continue
            node_ok = abs(n_loc - num_loc) <= 2
            req_ok  = required_count(n_arc) <= max_req

            if not (node_ok and req_ok):
                os.remove(fpath)
                continue

            es = np.load(fpath)
            reqs.append(es["req"])
            nonreqs.append(es["nonreq"])
            Cs.append(float(es["C"]))
            Ms.append(int(es["M"]))
            n_reqs.append(int(es["n_req"]))
            taus.append(float(es["tau"]))
            os.remove(fpath)
            print(f"[gen] {len(reqs)}/{per_bucket}  |V|={n_loc}  |A|={n_arc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    np.savez_compressed(
        out_path,
        req=np.array(reqs,    dtype=object),
        nonreq=np.array(nonreqs, dtype=object),
        C=np.array(Cs),
        M=np.array(Ms,    dtype=np.int32),
        n_req=np.array(n_reqs, dtype=np.int32),
        tau=np.array(taus),
    )
    print(f"[gen] saved {len(reqs)} instances → {out_path}")
    return os.path.abspath(out_path)
