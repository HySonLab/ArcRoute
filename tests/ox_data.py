import networkx as nx
import osmnx as ox
import os
import torch
import numpy as np
from data import *

def get_subgraph(G, num_nodes):
    if num_nodes > G.number_of_nodes():
        raise ValueError("num_nodes is greater than the number of nodes in the graph.")

    start_node = np.random.choice(list(G.nodes))
    bfs_tree = nx.bfs_tree(G, start_node)
    nodes_in_subgraph = list(bfs_tree.nodes)[:num_nodes]

    subgraph = G.subgraph(nodes_in_subgraph).copy()
    return subgraph

def get_random_connected_subgraph(G, num_nodes):
    num_nodes = num_nodes + 1
    subgraph = get_subgraph(G, num_nodes)
    while not nx.is_strongly_connected(subgraph):
        subgraph = get_subgraph(G.copy(), num_nodes)
    
    map_node = {vi:i for i, vi in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, map_node)
    return subgraph

def sample_traversal_time(num_loc, arcs, coords):
    dists = torch.cdist(coords, coords, p=2)
    traversal_time = dists[arcs[:, 0], arcs[:, 1]].clone()

    for k in range(num_loc):
        dists = torch.min(dists, dists[:, k].unsqueeze(1) + dists[k, :].unsqueeze(0))

    dists_edges = dists[arcs[..., 1].unsqueeze(-1), arcs[..., 0].unsqueeze(0)]
    return traversal_time, dists_edges

def generate(num_loc, num_arc, num_vehicle):
    sub = get_random_connected_subgraph(G_proj, 10)
    coords = torch.tensor([[sub.nodes[node]['x'], sub.nodes[node]['y']] for node in sub.nodes], dtype=torch.float32)
    norm = lambda x : (x - x.min()) / (x.max() - x.min())
    coords[:, 0] = norm(coords[:, 0])
    coords[:, 1] = norm(coords[:, 1])
    arcs = sample_arcs(num_loc, num_arc)
    clss = sample_priority_classes(num_arc)
    traversal_time, dists_edges = sample_traversal_time(num_loc, arcs, coords)
    servicing_time = sample_service_time(traversal_time)
    demands = sample_demand(traversal_time, clss)
    vehicle_capacity = sample_vehicle_capacity(demands, clss)
    num_vehicle = torch.tensor([num_vehicle])
    td = TensorDict(
            {
                'clss': clss,
                "demands": demands / vehicle_capacity,
                "capacity": vehicle_capacity,
                "service_times": servicing_time,
                "traversal_times": traversal_time,
                "adj": dists_edges,
                "num_vehicle": num_vehicle
            },
        )
    td = td.unsqueeze(0)
    td.batch_size=torch.Size([1]) 
    return td

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    G_dump = ox.graph_from_bbox(north=16.0741, south=16.0591, east=108.1972, west=108.2187)
    G_proj = ox.project_graph(G_dump)

    td = generate(10, 10, 3)
    

    print(td.batch_size[0])

    