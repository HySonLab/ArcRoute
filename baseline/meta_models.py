import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy.random import seed, random, randint, permutation

from common.ops import import_instance, softmax
from common.nb_utils import gen_tours, gen_tours_batch, deserialize_tours, deserialize_tours_batch
from common.cal_reward import get_Ts
from common.local_search import local_search

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

    def __call__(self, variant='P', is_local_search=True):
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
        
        routes = np.int32([a for tour in routes for a in tour])

        
        vars = {
            'adj': self.dms[None],
            'service_time': self.s[None],
            'clss': self.clss[None],
            'demand': self.demands[None]
        }
        if is_local_search:
            tours = local_search(vars, variant=variant, actions=routes[None], is_train=False)
            return get_Ts(vars, tours_batch=tours)
        return get_Ts(vars, actions=routes[None])

class EAHCARP:
    def __init__(self, n_population=500, n_epoch=500, pathnament_size=4, mutation_rate=0.5, crossover_rate=0.9):
        self.has_instance = False
        self.n_population = n_population
        self.n_epoch = n_epoch
        self.pathnament_size = pathnament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.idxs_org = np.arange(n_population)
        self.variant = None

    def import_instance(self, f):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        self.dms, self.P, self.M = dms, P, M
        self.demands, self.clss, self.s, self.d = demands, clss, s, d
        self.edge_indxs = edge_indxs
        self.has_instance = True
        self.vars = {
            'adj': self.dms[None],
            'service_time': self.s[None],
            'clss': self.clss[None],
            'demand': self.demands[None]
        }

        self.vars_batch={
            'adj': [self.dms]*self.n_population,
            'service_time': [self.s]*self.n_population,
            'clss': [self.clss]*self.n_population,
            'demand': [self.demands]*self.n_population
        }

    def is_valid(self, routes):
        if (self.demands[gen_tours(routes)].sum(-1) > 1).sum() > 0:
            return False
        return True

    def convert_prob(self, arr):
        """
            arr must be negative
        """
        min_value = np.min(arr)
        shifted_arr = arr - min_value + 1e-10
        prob = shifted_arr / np.sum(shifted_arr)
        return prob

    def calc_obj(self, population):
        n = len(population)
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars_batch, tours_batch=gen_tours_batch(population))
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

    def mutate(self, tours):
        n = len(tours)
        tours = local_search(self.vars, variant=self.variant, actions=tours[None], is_train=False)
        return deserialize_tours(tours[0], n)
    
    def __call__(self, variant='P', n_el=20):
        assert self.has_instance, "please import instance first"
        self.variant = variant
        self.l_early = []
        population = self.init_population()
        for epoch in range(self.n_epoch):
            # selecting two of the best options we have (elitism)
            obj = self.calc_obj(population)
            idxs = np.argsort(obj)
            new_population = [population[idxs[-1]], population[idxs[-2]]]
            prob =self.convert_prob(obj)

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
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child1)

                new_population.extend([child1, child2])

            population = new_population
            if epoch % 10 == 0:
                self.l_early.append(self.calc_obj(population).max())
                print(f"epoch {epoch}:", self.l_early[-1])

            if np.mean(self.l_early[-n_el:]) == self.l_early[-1] and len(self.l_early) > n_el:
                break

        routes = population[self.calc_obj(population).argmax()]
        return get_Ts(self.vars, actions=routes[None])
    

class ACOHCARP:
    def __init__(self, n_ant=100, n_epoch=500, alpha=1.0, beta=1.0, rho=0.5, del_tau=1.0, k=4):
        self.has_instance = False
        self.n_ant = n_ant
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.del_tau = del_tau
        self.k = k

    def import_instance(self, f):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f)
        self.dms, self.P, self.M = dms, P, M
        self.demands, self.clss, self.s, self.d = demands, clss, s, d
        self.edge_indxs = edge_indxs
        self.has_instance = True
        self.vars = {
            'adj': self.dms[None],
            'service_time': self.s[None],
            'clss': self.clss[None],
            'demand': self.demands[None]
        }

        self.vars_batch={
            'adj': [self.dms]*self.n_ant,
            'service_time': [self.s]*self.n_ant,
            'clss': [self.clss]*self.n_ant,
            'demand': [self.demands]*self.n_ant
        }
    def is_valid(self,route):
        if self.demands[route].sum() > 1:
            return False
        return True
    
    def calc_obj(self, population):
        n = len(population)
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars_batch, tours_batch=gen_tours_batch(population))
        obj = rewards @ w
        return -obj

    def init_population(self):
        pheromones = np.zeros_like(self.dms) # (17, 17)
        heads = []
        for _ in range(self.n_ant):
            head = permutation(range(1, len(self.s)))[:self.k]
            while not self.is_valid(head):
                head = permutation(range(1, len(self.s)))[:self.k]
            heads.append(head)

        heads = np.array(heads) # (50, 4)
        np.add.at(pheromones, (heads[:,:-1], heads[:,1:]), self.del_tau)
        np.add.at(pheromones, (np.zeros_like(heads[:, 0]), heads[:, 0]), self.del_tau)
        return heads, pheromones
    
    def __call__(self, variant='P', is_local_search = True):

        heads, pheromones = self.init_population()
        i_max, max_obj = 0, -np.inf
        
        elitist_epochs = []
        for epoch in range(self.n_epoch):
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
                tours = local_search(self.vars_batch, variant=variant, actions=elitist_ants, is_train=False)
                elitist_ants = deserialize_tours_batch(tours, n)

            obj = self.calc_obj(elitist_ants)
            i_objmax = obj.argmax()
            if obj[i_objmax] > max_obj:
                max_obj = obj[i_objmax]
                i_max = i_objmax
            elitist = np.int32(elitist_ants[obj.argmax()])
            pheromones[elitist[:-1], elitist[1:]] += self.del_tau
            pheromones *= (1 - self.rho)

            elitist_epochs.append(elitist)
            if epoch % 10 == 0:
                print(f"epoch {epoch}:", max_obj)
        
        return get_Ts(self.vars, actions=elitist_epochs[i_max][None])