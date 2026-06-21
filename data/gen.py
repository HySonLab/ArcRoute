"""Generate HDCARP benchmark instances following the paper's
"Create HDCARP instances" procedure (arXiv:2501.00852).

Pipeline:
  1. Extract a strongly-connected directed graph G=(V,A) from OpenStreetMap
     via OSMnx (real road topology + real node coordinates).
  2. Vehicle fleet |M| in {2, 5}.
  3. Required arcs A_r: |A_r| ~ U[60,70] if |A| >= 80, else 75% of |A|.
  4. Priority class of each required arc drawn uniformly from {1,2,3}.
  5. Traversal time d_a = d'_a / d'_max (Euclidean, normalized); service = 2*d_a.
  6. Demand q_a = d_a * 0.5 + 0.5.
  7. Vehicle capacity Q = sum_{a in A_r} (q_a / 3 + 0.5).

The per-instance physics live in `build_instance`, which is pure (no OSMnx) and
unit-tested in tests/test_gen.py. OSMnx is only needed to run this script
end to end; install it first, e.g. `uv add osmnx`.

Usage:
    uv run python data/gen.py --vehicles 5            # writes data/5m/<|A|>/*.npz
    uv run python data/gen.py --vehicles 2            # writes data/2m/<|A|>/*.npz

.npz schema (consumed by common/ops.import_instance):
    req:    (|A_r|, 6)   columns [tail, head, demand, clss, service, traversal]
    nonreq: (|A_nr|, 6)  columns [tail, head, 0, 0, 0, traversal]
    P=3, M=<vehicles>, C=<capacity>
"""
import os
import random
import argparse
import shutil

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #
# Instance physics (pure, OSMnx-free, unit-tested)
# --------------------------------------------------------------------------- #
def required_count(num_arc, rng=np.random):
    """Paper F2 (¼-split): |A_r| = 3*floor(|A|/4) — each of the three priority
    classes gets floor(|A|/4) arcs, the rest are non-required. This restores the
    original scale-invariant ~75% ratio (HRDA's old 60-70 rule caused a scaling
    defect where the required ratio collapsed as |A| grew)."""
    return 3 * (num_arc // 4)


def build_instance(edges, coords, M, rng=np.random):
    """Build an HDCARP instance from a directed graph.

    edges:  (|A|, 2) int array of (tail, head) node indices in 0..n-1
    coords: (n, 2)   float array of node (x, y) positions (e.g. projected meters)
    M:       number of vehicles
    Returns (req, nonreq, C). Raises ValueError on a degenerate graph.
    """
    edges = np.asarray(edges, dtype=int)
    coords = np.asarray(coords, dtype=np.float64)
    num_arc = len(edges)

    # Step 5: Euclidean distance per arc, normalized by the max over all arcs.
    d_eucl = np.sqrt(((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2).sum(-1))
    d_max = d_eucl.max()
    if d_max <= 0:
        raise ValueError("degenerate graph: all arc lengths are zero")
    d = d_eucl / d_max                            # traversal time d_a in (0, 1]

    # Step 3: pick the required arcs.
    n_req = required_count(num_arc, rng)
    if n_req >= num_arc:
        raise ValueError("not enough arcs to select the required set")

    idx = np.arange(num_arc)
    idx_req = rng.choice(idx, size=n_req, replace=False)
    # The depot (node 0) must be the tail of at least one required arc.
    tries = 0
    while 0 not in edges[idx_req, 0]:
        idx_req = rng.choice(idx, size=n_req, replace=False)
        tries += 1
        if tries > 1000:
            raise ValueError("could not place the depot on a required arc")
    idx_nonreq = np.setdiff1d(idx, idx_req)

    e_req, e_nonreq = edges[idx_req], edges[idx_nonreq]
    d_req, d_nonreq = d[idx_req], d[idx_nonreq]

    # Steps 5, 6: service = 2*d, demand q = 0.5*d + 0.5 (required arcs only).
    s_req = 2.0 * d_req
    q_req = 0.5 * d_req + 0.5

    # Paper F2: balanced classes — each of {1,2,3} gets the same count (n_req/3).
    per_class = n_req // 3
    clss = np.concatenate([
        np.repeat([1, 2, 3], per_class),
        np.full(n_req - 3 * per_class, 3),        # remainder (n_req not /3) -> class 3
    ])
    rng.shuffle(clss)

    # Paper F5: vehicle capacity Q = (sum over required arcs of q_a) / 3 + 0.5.
    # (add 0.5 ONCE — the old `(q/3 + 0.5).sum()` added 0.5 per arc, ~75x too loose.)
    C = float(q_req.sum() / 3.0 + 0.5)

    req = np.column_stack([e_req, q_req, clss, s_req, d_req]).astype(np.float64)
    z = np.zeros(len(e_nonreq))
    nonreq = np.column_stack([e_nonreq, z, z, z, d_nonreq]).astype(np.float64)
    return req, nonreq, C


# --------------------------------------------------------------------------- #
# OSMnx graph extraction
# --------------------------------------------------------------------------- #
def get_subgraph(G, num_nodes):
    if num_nodes > G.number_of_nodes():
        raise ValueError("num_nodes is greater than the number of nodes in the graph.")
    start_node = random.choice(list(G.nodes))
    bfs_tree = nx.bfs_tree(G, start_node)
    nodes_in_subgraph = list(bfs_tree.nodes)[:num_nodes]
    return G.subgraph(nodes_in_subgraph).copy()


def get_random_connected_subgraph(G, num_nodes):
    subgraph = get_subgraph(G, num_nodes)
    while not nx.is_strongly_connected(subgraph):
        subgraph = get_subgraph(G.copy(), num_nodes)
    return subgraph


def load_osm_graph(north, south, east, west):
    """Download and project an OSM drive graph for the given bounding box.
    Handles both the legacy (north/south/east/west) and osmnx>=2.0 (bbox tuple)
    signatures of graph_from_bbox.
    """
    import osmnx as ox
    try:
        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west,
                               network_type="drive")
    except TypeError:
        # osmnx >= 2.0: a single bbox = (left, bottom, right, top)
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    return ox.project_graph(G)


def gen_graph(G_proj, target_nodes, M, save_dir, rng=np.random):
    """Sample one strongly-connected subgraph and write a paper-faithful instance."""
    while True:
        sub = get_random_connected_subgraph(G_proj.copy(), target_nodes)
        node_map = {vi: i for i, vi in enumerate(sub.nodes())}
        coords = np.array([[sub.nodes[vi]["x"], sub.nodes[vi]["y"]] for vi in node_map],
                          dtype=np.float64)
        edges = np.array([[node_map[u], node_map[v]] for u, v in sub.edges()], dtype=int)
        if len(edges) == 0:
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

    num_arc = len(req) + len(nonreq)
    fpath = f"{save_dir}/{num_arc}_{len(coords)}_{rng.randint(0, 1000):03d}"
    np.savez(fpath, req=req, nonreq=nonreq, P=3, M=M, C=C)
    return fpath + ".npz", num_arc


def main():
    p = argparse.ArgumentParser(description="Generate paper-faithful HDCARP instances.")
    p.add_argument("--vehicles", type=int, default=5, choices=[2, 5],
                   help="fleet size |M| (paper uses {2, 5})")
    p.add_argument("--out", type=str, default=None,
                   help="output dir (default: data/<M>m)")
    p.add_argument("--per_bucket", type=int, default=20,
                   help="instances per |A| bucket (paper: 20)")
    p.add_argument("--tol", type=int, default=2,
                   help="accepted deviation of |A| from the bucket centre")
    p.add_argument("--seed", type=int, default=6868)
    p.add_argument("--bbox", type=float, nargs=4, metavar=("N", "S", "E", "W"),
                   default=[16.0741, 16.0591, 108.2187, 108.1972],
                   help="OSM bounding box: north south east west")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    M = args.vehicles
    # Paper step 2: |A| in {30,...,200} for M=2, {70,...,200} for M=5.
    first = 30 if M == 2 else 70
    buckets = list(range(first, 200 + 1, 10))

    out = args.out or f"data/{M}m"
    os.makedirs(out, exist_ok=True)

    N, S, E, W = args.bbox
    G_proj = load_osm_graph(N, S, E, W)

    tmp = "temp"
    os.makedirs(tmp, exist_ok=True)
    try:
        for B in buckets:
            pdir = os.path.join(out, str(B))
            os.makedirs(pdir, exist_ok=True)
            count = len([f for f in os.listdir(pdir) if f.endswith(".npz")])
            while count < args.per_bucket:
                target_nodes = max(5, int(np.random.randint(B // 3, B // 2 + 2)))
                target_nodes = min(target_nodes, G_proj.number_of_nodes())
                fpath, n_arc = gen_graph(G_proj, target_nodes, M, tmp)
                if abs(n_arc - B) > args.tol or os.path.isfile(
                    os.path.join(pdir, os.path.basename(fpath))
                ):
                    os.remove(fpath)
                    continue
                shutil.move(fpath, os.path.join(pdir, os.path.basename(fpath)))
                count += 1
                print(f"[M={M}] |A|~{B}: {count}/{args.per_bucket}  "
                      f"(arcs={n_arc}, nodes={target_nodes})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
