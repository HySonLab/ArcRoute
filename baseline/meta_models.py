import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.ops import import_instance, convert_prob
from common.nb_utils import gen_tours, deserialize_tours, deserialize_tours_batch
from common.cal_reward import get_Ts
import numpy as np
from numpy.random import random, randint, permutation
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

        self.vars = {
            'adj': self.dms,
            'service_time': self.s,
            'clss': self.clss,
            'demand': self.demands
        }


class InsertCheapestHCARP(BaseHCARP):
    
    def calc_obj(self, path):
        return -(self.s[path].sum() + self.dms[path[:-1], path[1:]].sum())

    def is_valid(self, path):
        if self.demands[path].sum() > 1:
            return False
        return True

    def __call__(self, variant='P', is_local_search=True):
        assert self.has_instance, "please import instance first"

        routes = [[0] for _ in self.M]
        for p in self.P:
            edges = np.where(self.clss==p)[0]
            for e in edges:
                paths = [routes[m] + [e] for m in self.M]
                idxs = [i for i, path in enumerate(paths) if self.is_valid(path)]
                costs = [self.calc_obj(paths[i]) for i in idxs]
                idx = np.random.choice(idxs, size=1, p=convert_prob(costs))[0]
                routes[idx] = paths[idx]
        
        routes = np.int32([a for tour in routes for a in tour])[None]

        if is_local_search:
            tours_batch = ls(self.vars, variant=variant, actions=routes)
            return get_Ts(self.vars, tours_batch=tours_batch)
        return get_Ts(self.vars, actions=routes)

class EAHCARP(BaseHCARP):
    def __init__(self, n_population=500, pathnament_size=4, mutation_rate=0.5, crossover_rate=0.9):
        super().__init__()
        self.n_population = n_population
        self.pathnament_size = pathnament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.idxs_org = np.arange(n_population)

    def is_valid(self, routes):
        if (self.demands[gen_tours(routes)].sum(-1) > 1).sum() > 0:
            return False
        return True

    def calc_obj(self, population):
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars, actions=population)
        obj = rewards @ w
        return -obj
    
    def init_population(self):
        routes = np.int32(list(range(1, len(self.s))) + [0]*(len(self.M)-1))
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
    
    def __call__(self, n_epoch=500, variant='P'):
        assert self.has_instance, "please import instance first"
        self.l_early = []
        population = self.init_population()
        for epoch in range(n_epoch):
            # selecting two of the best options we have (elitism)
            obj = self.calc_obj(population)
            idxs = np.argsort(obj)
            new_population = [population[idxs[-1]], population[idxs[-2]]]
            prob = convert_prob(obj)

            for i in range(len(population) // 2 - 1):
                # CROSSOVER
                parent1 = population[np.random.choice(self.idxs_org, size=self.pathnament_size, p=prob)[0]]
                parent2 = population[np.random.choice(self.idxs_org, size=self.pathnament_size, p=prob)[0]]
                if random() < self.crossover_rate:
                    child1 = self.cross_over(parent1, parent2, randint(1, len(parent1)))
                    child2 = self.cross_over(parent2, parent1, randint(1, len(parent1)))
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
            # if epoch % 10 == 0:
            #     print(f"epoch {epoch}:", self.l_early[-1])


        routes = population[self.calc_obj(population).argmax()]
        return get_Ts(self.vars, actions=routes[None])
    

class ACOHCARP(BaseHCARP):
    def __init__(self, n_ant=100, alpha=1.0, beta=1.0, rho=0.5, del_tau=1.0, k=4):
        super().__init__()
        self.n_ant = n_ant
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.del_tau = del_tau
        self.k = k

    def is_valid(self,route):
        if self.demands[route].sum() > 1:
            return False
        return True
    
    def calc_obj(self, population):
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars, actions=population)
        obj = rewards @ w
        return -obj

    def init_population(self):
        pheromones = np.zeros_like(self.dms) 
        heads = []
        for _ in range(self.n_ant):
            head = permutation(range(1, len(self.s)))[:self.k]
            while not self.is_valid(head):
                head = permutation(range(1, len(self.s)))[:self.k]
            heads.append(head)

        heads = np.array(heads) 
        np.add.at(pheromones, (heads[:,:-1], heads[:,1:]), self.del_tau)
        np.add.at(pheromones, (np.zeros_like(heads[:, 0]), heads[:, 0]), self.del_tau)
        return heads, pheromones
    
    def __call__(self, n_epoch=500, variant='P', is_local_search = True):
        assert self.has_instance, "please import instance first"
        
        heads, pheromones = self.init_population()
        i_max, max_obj = 0, -np.inf
        
        elitist_epochs = []
        for epoch in range(n_epoch):
            elitist_ants = []
            for head in heads:
                tour = head.tolist()
                sub = tour.copy()
                while len(tour) < len(self.s):
                    mask = pheromones[tour[-1]].copy()
                    mask[tour] -= 1e8
                    next = np.argmax(mask)
                    while not self.is_valid(sub + [next]):
                        mask[next] -= 1e8
                        next = np.argmax(mask)
                    tour.append(next)
                    sub.append(next)
                    if next == 0:
                        sub = []
                elitist_ants.append(tour)

            if is_local_search:
                n = len(elitist_ants[0])
                tours = ls(self.vars, variant=variant, actions=elitist_ants)
                elitist_ants = deserialize_tours_batch(tours, n)

            # Update elite pheromones
            obj = self.calc_obj(elitist_ants)
            i_objmax = obj.argmax()
            if obj[i_objmax] > max_obj:
                max_obj = obj[i_objmax]
                i_max = epoch
            elitist = np.int32(elitist_ants[i_objmax])
            pheromones[elitist[:-1], elitist[1:]] += self.del_tau
            pheromones *= (1 - self.rho)

            elitist_epochs.append(elitist)
            # if epoch % 10 == 0:
            #     print(f"epoch {epoch}:", max_obj)
        
        return get_Ts(self.vars, actions=elitist_epochs[i_max][None])