import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from common.ops import import_instance, softmax
from common.local_search import local_search
from common.cal_reward import get_Ts

class InsertCheapestHCARP:
    def __init__(self):
        self.has_instance = False

    def import_instance(self, f):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        self.dms, self.P, self.M = dms, P, M
        self.demands, self.clss, self.s, self.d = demands, clss, s, d
        self.edge_indxs = edge_indxs
        self.has_instance = True
    
    def calc_cost(self, path):
        return self.s[path].sum() + self.dms[path[:-1], path[1:]].sum()
    def calc_demand(self, path):
        return self.demands[path].sum()

    def __call__(self, merge_tour=True, is_local_search=False):
        assert self.has_instance, "please import instance first"

        routes = [[0] for _ in self.M]
        for p in self.P:
            edges = np.where(self.clss==p)[0]
            for e in edges:
                paths = [routes[m] + [e] for m in self.M]
                idxs = [i for i, path in enumerate(paths) if self.calc_demand(path) <= 1]
                costs = [-self.calc_cost(paths[i]) for i in idxs]
                idx = np.random.choice(idxs, size=1, p=softmax(costs))[0]
                routes[idx] = paths[idx]
        
        if merge_tour:
            routes = np.int32([a for tour in routes for a in tour])

        return routes
        

def parse_args():
    parser = argparse.ArgumentParser(description="Run InsertCheapestHCARP with custom inputs")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file (.npz)')

    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(42)

    args = parse_args()
    al = InsertCheapestHCARP()
    al.import_instance(args.data_path)
    routes = al(merge_tour=True)
    vars = {
        'adj': al.dms[None],
        'service_time': al.s[None],
        'clss': al.clss[None],
        'demand': al.demands[None]
    }
    # print(vars['demand'].sum(-1))
    tours = local_search(vars, actions=routes[None])
    r = get_Ts(vars, tours_batch=tours)
    print(r)