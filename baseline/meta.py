import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.random import random, randint, permutation

from common.ops import import_instance
from common.nb_utils import gen_tours, deserialize_tours, deserialize_tours_batch, convert_prob
from common.cal_reward import get_Ts
from common.local_search import ls

class BaseHCARP:
    def __init__(self):
        self.has_instance = False

    def import_instance(self, f):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        self.dms, self.P, self.M = dms, P, M
        self.demands, self.clss, self.s, self.d = demands, clss, s, d
        self.edge_indxs = edge_indxs
        self.has_instance = True
        self.nv = len(M)
        self.nseq = len(self.s) + (self.nv - 2)

        self.vars = {
            'adj': self.dms,
            'service_time': self.s,
            'clss': self.clss,
            'demand': self.demands
        }
    
    def get_idx(self, prob, size=1, strategy='sampling'):
        if strategy == 'greedy':
            return np.argmax(prob)
        
        return np.random.choice(len(prob), size=size, p=prob)[0]
        

    def is_valid_once(self, path):
        if self.demands[path].sum() > 1:
            return False
        return True
    
    def is_valid(self, routes):
        if (self.demands[gen_tours(routes)].sum(-1) > 1).sum() > 0:
            return False
        return True
    
    def calc_obj(self, actions):
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars, actions=actions)
        obj = rewards @ w
        return -obj
    
    def get_best(self, samples):
        obj = self.calc_obj(samples)
        idx = obj.argmax()
        return obj[idx], samples[idx]

class InsertCheapestHCARP(BaseHCARP):
    def get_once(self):
        routes = [[0] for _ in self.M]
        for p in self.P:
            edges = np.where(self.clss==p)[0]
            for e in edges:
                paths = [routes[m] + [e] for m in self.M]
                idxs = [i for i, path in enumerate(paths) if self.is_valid(path)]
                costs = [self.is_valid_once(paths[i]) for i in idxs]
                idx = self.get_idx(convert_prob(costs), strategy='sampling', size=4)
                routes[idx] = paths[idx]
        
        return np.int32([a for tour in routes for a in tour])

    def __call__(self, variant='P', num_sample=20):
        assert self.has_instance, "please import instance first"
        actions = [self.get_once() for _ in range(num_sample)]
        tours_batch = ls(self.vars, variant=variant, actions=actions)
        actions = deserialize_tours_batch(tours_batch, self.max_seq)
        _, best = self.get_best(actions)
        return get_Ts(self.vars, actions=best[None])

class EAHCARP(BaseHCARP):
    def __init__(self, n_population=500, pathnament_size=4, mutation_rate=0.5, crossover_rate=0.9):
        super().__init__()
        self.n_population = n_population
        self.pathnament_size = pathnament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def init_population(self):
        routes = np.int32(list(range(1, self.max_seq)) + [0]*(len(self.M)-1))
        population = []
        for _ in range(self.n_population):
            routes = permutation(routes)
            while not self.is_valid(routes):
                routes = permutation(routes)
            population.append(routes)
        population = np.array(population)
        return population
    
    def _cross_over(self, p1, p2, k):
        child = p1[:k]
        child = np.concatenate((child, p2[~np.isin(p2, child)]))
        return child
    
    def cross_over(self, p1, p2, k):
        child = self._cross_over(p1, p2, k)
        while not self.is_valid(child):
            child = self._cross_over(p1, p2, k)
        return child

    def mutate(self, tours, variant):
        n = len(tours)
        tours = ls(self.vars, variant=variant, actions=tours[None])
        return deserialize_tours(tours[0], n)
    
    def get_parent(self, population, prob):
        return population[self.get_idx(prob, strategy='sampling', size=self.pathnament_size)]

    def __call__(self, n_epoch=500, variant='P', verbose=False):
        assert self.has_instance, "please import instance first"
        
        population = self.init_population()
        for epoch in range(n_epoch):
            # selecting two of the best options we have (elitism)
            obj = self.calc_obj(population)
            idxs = np.argsort(obj)
            new_population = [population[idxs[-1]], population[idxs[-2]]]
            prob = convert_prob(obj)

            for i in range(len(population) // 2 - 1):
                # CROSSOVER
                parent1 = self.get_parent(population, prob)
                parent2 = self.get_parent(population, prob)
                if random() < self.crossover_rate:
                    child1 = self.cross_over(parent1, parent2, randint(1, len(parent1)))
                    child2 = self.cross_over(parent2, parent1, randint(1, len(parent2)))

                # If crossover not happen
                else:
                    child1 = parent1
                    child2 = parent2

                # MUTATION
                if random() <self.mutation_rate:
                    child1 = self.mutate(child1, variant)
                    child2 = self.mutate(child1, variant)

                new_population.extend([child1, child2])

            population = new_population
            if verbose:
                if epoch % 10 == 0:
                    obj = self.calc_obj(population)
                    idx = np.argsort(obj)[-1]
                    print(f"epoch {epoch}:", obj[idx])


        routes = population[self.calc_obj(population).argmax()]
        return get_Ts(self.vars, actions=routes[None])
    

class ACOHCARP(BaseHCARP):
    def __init__(self, n_ant=100, alpha=1.0, beta=1.0, rho=0.5, del_tau=1.0):
        super().__init__()
        self.n_ant = n_ant
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.del_tau = del_tau

    def init_population(self):
        pheromones = np.zeros((self.nv, *(self.dms.shape)))
        ants = np.int32([np.stack([np.zeros(self.nv), 
                        permutation(range(1, self.nseq))[:self.nv]]).T
                        for _ in range(self.n_ant)])
        for m in self.M:
            ant = ants[:, m]
            np.add.at(pheromones[m], (ant[:, :-1], ant[:, 1:]), self.del_tau)
        return ants, pheromones


    def find_tour(self, ant, pheromones):
        while True:
            visited = np.unique(ant).tolist()
            _ant = ant.tolist()
            while len(visited) < len(self.s):
                for i in self.M:
                    tour = _ant[i]
                    mask = pheromones[i][tour[-1]].copy()
                    mask[visited] -= 1e8
                    next = self.get_idx(convert_prob(mask), size=1, strategy='sampling')
                    while not self.is_valid_once(tour + [next]):
                        mask[next] -= 1e8
                        next = self.get_idx(convert_prob(mask), size=1, strategy='sampling')
                    _ant[i].append(next)
                    visited.append(next)
            routes = np.int32(np.concatenate(_ant))
            if self.is_valid(routes):
                return routes[1:]

    def update_pheromones(self, best_ants, pheromones):
        best_ants = gen_tours(best_ants)
        for m in self.M:
            np.add.at(pheromones[m], (best_ants[m, :-1], best_ants[m, 1:]), 1)
        pheromones *= (1 - self.rho)
        return pheromones

    def __call__(self, n_epoch=500, variant='P', is_local_search = True, verbose=False):
        assert self.has_instance, "please import instance first"
        
        ants, pheromones = self.init_population()
        best_obj = -np.inf
        elitist_epochs = []
        for epoch in range(n_epoch):

            # Constructing tour of ants
            elitist_ants = [self.find_tour(ant, pheromones=pheromones) for ant in ants]

            # refine tours by local search
            if is_local_search:
                tours = ls(self.vars, variant=variant, actions=elitist_ants)
                elitist_ants = deserialize_tours_batch(tours, self.nseq)

            # Update pheromones 
            ant_obj, best_ants = self.get_best(elitist_ants)
            pheromones = self.update_pheromones(best_ants, pheromones)

            best_obj = max(best_obj, ant_obj)
            elitist_epochs.append(best_ants)
            if verbose:
                if epoch % 10 == 0:
                    print(f"epoch {epoch}:", best_obj)
                    
        best = self.get_best(elitist_epochs)[1][None]
        return get_Ts(self.vars, actions=best)