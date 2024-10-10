import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import glob
from tqdm import tqdm
from common.nb_utils import deserialize_tours_batch
from common.ops import import_instance, softmax, gather_by_index
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

if __name__ == "__main__":
    np.random.seed(42)

    al = InsertCheapestHCARP()
    files = glob.glob('/usr/local/sra/testing_data/medium/*.npz')
    files = sorted(files, key=lambda x : int(x.split('/')[-1].split('_')[0]))[:100]
    up = []
    for f in tqdm(files):
        # print('---')
        al.import_instance(f)
        routes = al(merge_tour=True)[None]
        vars = {
            'adj': al.dms[None],
            'service_time': al.s[None],
            'clss': al.clss[None],
            'demand': al.demands[None]
        }
        
        tours1 = local_search(vars, actions=routes, variant='P', is_train=False)
        rp = get_Ts(vars, tours_batch=tours1)
        # tb = deserialize_tours_batch(tours)
        # print(tb, np.take_along_axis(vars['clss'], tb, axis=1))

        tours2 = local_search(vars, actions=routes, variant='U', is_train=False)
        ru = get_Ts(vars, tours_batch=tours2)
        # tb = deserialize_tours_batch(tours)
        # print(tb, np.take_along_axis(vars['clss'], tb, axis=1))
        # print(rp, ru)
        # exit()
        up.append((ru-rp)[0])
    up = np.array(up, np.float16)
    print(up)
    # print(up[:, 0].mean(), up[:, 1].mean(), up[:, 2].mean())