"""Generate ONE HDCARP case-study instance from a held-out test city,
keeping the node coordinates that the production pipeline discards.

Reuses the exact generation primitives from src/utils/data_utils.py
(load_osm_graph, _largest_scc, _random_connected_subgraph, _trim_edges,
build_instance) -- no reimplementation. Mirrors _sample_instance's retry
loop, but saves a sidecar .npz with coords (N x 2) and edges (|A| x 2)
so routes can be drawn on the road network.

Usage (from /home/quangnm/projects/ArcRoute):
    uv run python <this file> <city> <bucket> <seed> <out_prefix>
Writes <out_prefix>.npz (bm5-style single instance) and <out_prefix>_geo.npz.
"""
import sys, random
sys.path.insert(0, 'src')
import numpy as np
import networkx as nx

from utils.data_utils import (
    CITIES, BUCKETS, M_NOMINAL,
    load_osm_graph, _largest_scc, _random_connected_subgraph,
    _trim_edges, build_instance,
)

city, bucket, seed, out_prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
np.random.seed(seed)
random.seed(seed)

cfg = BUCKETS[bucket]
num_loc, min_arc, max_arc = cfg["num_loc"], cfg["min_arc"], cfg["max_arc"]

N, S, E, W = CITIES[city]
G_proj = load_osm_graph(N, S, E, W)
G_proj = _largest_scc(G_proj)
print(f"[{city}] base SCC |V|={G_proj.number_of_nodes()} |A|={G_proj.number_of_edges()}")

for attempt in range(2000):
    sub = _random_connected_subgraph(G_proj, num_loc)
    node_map = {v: i for i, v in enumerate(sub.nodes())}
    coords = np.array([[sub.nodes[v]["x"], sub.nodes[v]["y"]] for v in node_map],
                      dtype=np.float64)
    edges = np.array([[node_map[u], node_map[v]] for u, v in sub.edges()], dtype=int)
    if len(edges) == 0:
        continue
    edges = _trim_edges(edges, max_arc)
    if len(edges) < min_arc:
        continue
    g = nx.DiGraph(); g.add_edges_from(edges.tolist())
    if not nx.is_strongly_connected(g):
        continue
    try:
        req, nonreq, C = build_instance(edges, coords, M_NOMINAL)
    except ValueError:
        continue
    break
else:
    raise RuntimeError("no valid instance after 2000 tries")

num_arc = len(req) + len(nonreq)
tau = float(req[:, 2].sum() / (M_NOMINAL * C))  # matches _sample_instance
np.savez(out_prefix, req=req, nonreq=nonreq, P=3, M=M_NOMINAL, C=C,
         d=num_arc / len(coords), topology="osm", tau=tau, n_req=len(req))
np.savez(out_prefix + "_geo", coords=coords, edges=edges,
         city=city, bucket=bucket, seed=seed)
print(f"[{city}/{bucket}] attempt={attempt} |V|={len(coords)} |A|={num_arc} "
      f"n_req={len(req)} C={C:.4f} M={M_NOMINAL}")
print(f"Saved -> {out_prefix}.npz and {out_prefix}_geo.npz")
