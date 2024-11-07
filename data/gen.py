import networkx as nx
import osmnx as ox
import os
import random
import torch
import numpy as np
import shutil
from torch.distributions import Uniform

CAPACITIES = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }

def check():
    from glob import glob
    from collections import defaultdict

    files = glob('instances/*/*.npz')
    l = defaultdict(list)
    for file in files:
        l[int(file.split('/')[-2])].append(file)
    for k in l.keys():
        print(k, len(l[k]))

def get_subgraph(G, num_nodes):
    if num_nodes > G.number_of_nodes():
        raise ValueError("num_nodes is greater than the number of nodes in the graph.")

    start_node = random.choice(list(G.nodes))
    bfs_tree = nx.bfs_tree(G, start_node)
    nodes_in_subgraph = list(bfs_tree.nodes)[:num_nodes]

    subgraph = G.subgraph(nodes_in_subgraph).copy()
    return subgraph

def get_random_connected_subgraph(G, num_nodes):
    subgraph = get_subgraph(G, num_nodes)
    while not nx.is_strongly_connected(subgraph):
        subgraph = get_subgraph(G.copy(), num_nodes)
    return subgraph

def gen_graph(num_loc, num_arc):
    while True:
        subgraph = get_random_connected_subgraph(G_proj.copy(), num_loc)
        g = nx.DiGraph()
        map_node = {vi:i for i, vi in enumerate(subgraph.nodes())}
        for u,v,attr in subgraph.edges(data=True):
            u,v = map_node[u],map_node[v]
            g.add_edge(u,v)

        if nx.is_empty(g):
            continue
        if not nx.is_strongly_connected(g):
            continue

        idxs = np.arange(len(g.edges))
        if num_arc >= len(idxs):
            continue
        idxs_req = np.random.choice(idxs, size=num_arc, replace=False)
        e_req = np.array(list(g.edges()))[idxs_req]
        while 0 not in e_req[:, 0]:
            idxs = np.arange(len(g.edges))
            idxs_req = np.random.choice(idxs, size=num_arc, replace=False)
            e_req = np.array(list(g.edges()))[idxs_req]
        e_nonreq = np.array(list(g.edges()))[np.setdiff1d(idxs, idxs_req)]

        min_demand, max_demand = 1, 10
        demand_sampler = Uniform(low=min_demand-1, high=max_demand-1)
        demands = demand_sampler.sample((num_arc, 1))
        demands = (demands.int() + 1).float()

        
        # Capacity
        capacity = CAPACITIES.get(num_arc, None)
        if capacity is None:
            closest_num_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_arc))
            capacity = CAPACITIES[closest_num_loc]
        capacity = capacity * 2

        clss = torch.randint(1, 3+1, size=(num_arc, 1))
        a, b = 1, 2
        s = a + (b - a) * torch.rand(num_arc, 1)

        dms = torch.rand(num_loc, num_loc)
        dms[dms == 0] = float('inf')
        torch.diagonal(dms, dim1=0, dim2=1).fill_(0)

        d_nonreq = dms[e_nonreq[:, 0], e_nonreq[:, 1]][..., None]
        d_req = dms[e_req[:, 0], e_req[:, 1]][..., None]

        demands = demands.numpy()
        clss = clss.numpy()
        s = s.numpy()
        d_req = d_req.numpy()

        req = np.concatenate([e_req,demands,clss,s,d_req], axis=-1)
        nonreq = np.concatenate([e_nonreq, np.zeros_like(d_nonreq),np.zeros_like(d_nonreq),np.zeros_like(d_nonreq),d_nonreq], axis=-1)
        break
    
    fpath = f'{dir}/{len(req)+len(nonreq)}_{num_loc}_{np.random.randint(0,1000):03d}'
    np.savez(fpath, req=req, nonreq=nonreq, P=3, M=5, C=capacity)
    return fpath + '.npz', len(req)+len(nonreq)


if __name__ == "__main__":

    G_dump = ox.graph_from_bbox(north=16.0741, south=16.0591, east=108.1972, west=108.2187)
    G_proj = ox.project_graph(G_dump)


    nums = list(range(10, 90, 4))
    nums = list(zip(nums[:-1], nums[1:]))

    dir = "temp"
    if not os.path.isdir(dir): os.mkdir(dir)

    save = 'instances'
    if not os.path.isdir(save): os.mkdir(save)
    for i in range(6, 20):
        n = 10*i
        pdir = f'{save}/{n+10}/'
        if not os.path.isdir(pdir): os.mkdir(pdir)

        low, high = np.array([8, 12])+n

        count = 0
        while count < 20:
            fpath, n_arc = gen_graph(num_loc=np.random.randint(*nums[min(i-2, len(nums)-1)]), num_arc=60)
            if not ((n_arc <= high) & (n_arc >= low)) or os.path.isfile(pdir + fpath.split('/')[-1]):
                continue
            shutil.move(fpath, pdir)
            count += 1

    shutil.rmtree(dir)