import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.random import random, randint, permutation

from utils.ops import import_instance, run_parallel2
from utils.ops import gen_tours, deserialize_tours, deserialize_tours_batch, convert_prob
from solvers.cal_reward import get_Ts
from utils.local_search import ls
from utils.ils_operators import perturbate, accept
from copy import deepcopy

class BaseHCARP:
    def __init__(self):
        self.has_instance = False

    def import_instance(self, f, M=None):
        dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f, M=M)
        self.dms, self.P, self.M = dms, P, M
        self.demands, self.clss, self.s, self.d = demands, clss, s, d
        self.edge_indxs = edge_indxs
        self.has_instance = True
        self.nv = len(M)
        self.nseq = len(self.s)
        self.max_len = self.nseq + (self.nv - 2)

        self.vars = {
            'adj': self.dms,
            'service_time': self.s,
            'clss': self.clss,
            'demand': self.demands,
            'nv': self.nv,
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
        # gen_tours yields variable-length routes (a ragged list), so index/sum
        # per route instead of fancy-indexing the ragged list at once.
        for tour in gen_tours(routes):
            if self.demands[tour].sum() > 1:
                return False
        return True
    
    def calc_obj(self, actions):
        w = np.array([1e3, 1e1, 1e-1])
        rewards = get_Ts(self.vars, actions=actions)
        obj = rewards @ w
        return -obj
    
    def calc_len(self, path):
        return -(self.s[path].sum() + self.dms[path[:-1], path[1:]].sum())
    
    def get_best(self, samples):
        obj = self.calc_obj(samples)
        idx = obj.argmax()
        return obj[idx], samples[idx]
    
    def get_once(self, M, clss, attemp=10):
        for _ in range(attemp):
            routes = [[0] for _ in M]
            edges = [e for p in self.P for e in permutation(np.where(clss==p)[0])]
            i = 0
            while i < len(edges):
                paths = [routes[m] + [edges[i]] for m in M]
                idxs = [i for i, path in enumerate(paths) if self.is_valid_once(path)]
                if len(idxs) == 0: 
                    break
                costs = [self.calc_len(paths[i]) for i in idxs]
                idx = self.get_idx(convert_prob(costs), strategy='sampling', size=4)
                chosen = idxs[idx]
                routes[chosen] = paths[chosen]
                i += 1
            if i >= len(edges):
                return np.int32([a for tour in routes for a in tour])[1:]
        return None

class InsertCheapestHCARP(BaseHCARP):
    def __call__(self, variant='P', num_sample=20):
        assert self.has_instance, "please import instance first"
        actions = [self.get_once(self.M, self.clss) for _ in range(num_sample)]
        tours_batch = ls(self.vars, variant=variant, actions=actions)
        actions = deserialize_tours_batch(tours_batch, self.nseq)
        _, best = self.get_best(actions)
        # print(best)
        # print(gen_tours(best))
        return get_Ts(self.vars, actions=best[None])
    
class ILSHCARP(BaseHCARP):
    """Iterated Local Search for HDCARP.

    Constructive sampling (``get_once``) seeds an initial solution that is
    refined by the inter/intra ``ls`` operators; the ILS loop then alternates a
    feasibility-preserving ``perturbate`` kick with a full local search and an
    acceptance test (``'best'`` improvement-only or ``'sa'`` simulated
    annealing). The objective (``calc_obj``, higher = better) is the negated
    lexicographically-weighted (T1, T2, T3).
    """

    def __init__(self, strength=3, accept_mode='best', T_init=0.05, T_final=0.001):
        super().__init__()
        self.strength = strength
        self.accept_mode = accept_mode
        self.T_init = T_init
        self.T_final = T_final

    def __call__(self, max_iter=200, variant='P', num_init_sample=5,
                 seed=None, verbose=False):
        assert self.has_instance, "please import instance first"
        rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

        # 1. constructive seeds -> local search -> pick the best as s0
        samples = [self.get_once(self.M, self.clss) for _ in range(num_init_sample)]
        samples = [s for s in samples if s is not None]
        assert samples, "construction failed to produce a feasible solution"
        tours_batch = ls(self.vars, variant=variant, actions=samples)
        action_batch = deserialize_tours_batch(tours_batch, self.nseq)
        cur_obj, cur_action = self.get_best(action_batch)
        cur_routes = gen_tours(cur_action)

        best_action = cur_action.copy()
        best_obj = cur_obj
        capacity = 1.0

        # 2. ILS loop: perturb -> local search -> accept
        for it in range(max_iter):
            pert = perturbate(cur_routes, self.dms, self.demands, self.clss,
                              capacity, variant, rng, strength=self.strength)
            pert_action = deserialize_tours(pert, self.max_len)
            tours = ls(self.vars, variant=variant, actions=[pert_action])
            cand_action = deserialize_tours(tours[0], self.max_len)
            cand_obj = float(self.calc_obj(cand_action[None])[0])

            if accept(cur_obj, cand_obj, it, max_iter, mode=self.accept_mode,
                      T_init=self.T_init, T_final=self.T_final):
                cur_action = cand_action
                cur_obj = cand_obj
                cur_routes = gen_tours(cand_action)

            if cand_obj > best_obj:
                best_action = cand_action.copy()
                best_obj = cand_obj

            if verbose and it % 20 == 0:
                print(f"iter {it}: cur={cur_obj:.4f} best={best_obj:.4f}")

        # 3. report the hierarchical completion times of the best solution
        self._last_best_action = best_action
        return get_Ts(self.vars, actions=best_action[None])


class EAHCARP(BaseHCARP):
    def __init__(self, n_population=50, pathnament_size=4, mutation_rate=0.5, crossover_rate=0.9):
        super().__init__()
        self.n_population = n_population
        self.pathnament_size = pathnament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def init_population(self):
        samples = [self.get_once(self.M, self.clss) for _ in range(self.n_population)]
        samples = [s for s in samples if s is not None]
        if not samples:
            raise RuntimeError("init_population: construction failed to produce any feasible individual")
        return np.array(samples)

    def _cross_over(self, p1, p2):
        child1 = p1[:np.where(p1==0)[0][0]+1]
        child2 = p2[:np.where(p2==0)[0][0]+1]
        
        clss = self.clss.copy()
        clss[child1] = 0
        clss[child2] = 0
        child = self.get_once(np.arange(len(self.M)-2), clss) if len(self.M)-2 > 0 else []
        if child is None: return None
        clss = self.clss.copy()
        clss[child] = 0
        remain = self.get_once(range(2), clss)
        if remain is None: return None
        if len(self.M)-2 > 0:
            return np.concatenate([child, np.int32([0]), remain])
        return np.int32(remain)
    
    def cross_over(self, p1, p2, attemp=10):
        for _ in range(attemp):
            child = self._cross_over(p1, p2)
            if child is None: continue
            if self.is_valid(child):
                return child
        return p1

    def mutate(self, tours, variant):
        tours = ls(self.vars, variant=variant, actions=tours[None])
        return deserialize_tours(tours[0], self.max_len)
    
    def get_parent(self, population, prob):
        return population[self.get_idx(prob, strategy='sampling', size=self.pathnament_size)]

    def operate(self, prob, population, variant):
        # CROSSOVER
        parent1 = self.get_parent(population, prob)
        parent2 = self.get_parent(population, prob)

        # default to the parents so child1/child2 are always defined
        child1, child2 = parent1, parent2
        if random() < self.crossover_rate:
            # print(parent1, parent2)
            child1 = self.cross_over(parent1, parent2)
            child2 = self.cross_over(parent2, parent1)

        # MUTATION
        if random() < self.mutation_rate:
            child1 = self.mutate(child1, variant)
            child2 = self.mutate(child2, variant)
        return child1, child2

    def __call__(self, n_epoch=50, variant='P', verbose=False):
        assert self.has_instance, "please import instance first"
        
        population = self.init_population()
        for epoch in range(n_epoch):
            # selecting two of the best options we have (elitism)
            obj = self.calc_obj(population)
            idxs = np.argsort(obj)
            new_population = [population[idxs[-1]], population[idxs[-2]]]
            prob = convert_prob(obj)
            nb = len(population) // 2 - 1
            new_population += [c for cs in run_parallel2(self.operate, [prob]*nb, population=population, variant=variant) 
                               for c in cs]
            population = new_population
            if verbose:
                if epoch % 10 == 0:
                    obj = self.calc_obj(population)
                    idx = np.argsort(obj)[-1]
                    print(f"epoch {epoch}:", obj[idx])

        routes = population[self.calc_obj(population).argmax()]
        self._last_best_action = np.asarray(routes)
        return get_Ts(self.vars, actions=routes[None])

class ACOHCARP(BaseHCARP):
    def __init__(self, n_ant=50, alpha=1.0, beta=1.0, rho=0.5, del_tau=1.0):
        super().__init__()
        self.n_ant = n_ant
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.del_tau = del_tau

    def init_population(self, variant):
        pheromones = np.zeros_like(self.dms)
        ants = []
        idx_clss = np.unique(self.clss)[1:]
        l_idxs = [np.where(self.clss == k)[0].tolist() for k in idx_clss]
        for _ in range(self.n_ant):
            tours = []
            for vs in l_idxs:
                for v in permutation(vs):
                    if len(tours) == len(self.M): break
                    tours.append([0, v])
            ants.append(tours)

        if variant == 'P':
            for i, mask in enumerate(pheromones):
                mask[self.clss < self.clss[i]] = -np.inf
        
        return ants, pheromones

    def get_next(self, t, mask, attemp=10):
        for _ in range(attemp):
            if (mask < 0).all(): return None
            next = self.get_idx(convert_prob(mask), size=1, strategy='sampling')
            if self.is_valid_once(t + [next]):
                return next
            mask[next] = -np.inf
        return None

    def _construct_route(self, a, masks, variant='P'):
        visited = list(np.unique(a))
        visited_set = set(int(x) for x in visited)
        while len(visited) < self.nseq:
            # F5: enforce a GLOBAL priority frontier in P-variant. The per-arc
            # pheromone mask only forbids a route from dropping to a lower class;
            # it does NOT stop routes from racing ahead to higher classes while
            # high-priority arcs are still unserved, which deadlocks construction
            # (every route ends on a class above the stranded arcs and can't go
            # back). Mirror `get_once`: only the lowest still-unserved class is
            # eligible, so classes are exhausted in priority order.
            frontier = None
            if variant == 'P':
                frontier = min(self.clss[i] for i in range(self.nseq)
                               if i not in visited_set)
            all_none = True
            for i in permutation(range(len(a))):
                mask = masks[a[i][-1]].copy()
                if frontier is not None:
                    mask[self.clss > frontier] = -np.inf
                next = self.get_next(a[i], mask)
                if next is None:
                    continue
                a[i].append(next)
                visited.append(next)
                visited_set.add(int(next))
                masks[:, next] = -np.inf
                all_none = False
            if all_none: return None

        routes = np.int32(np.concatenate(a))
        if self.is_valid(routes):
            return routes[1:]
        return None

    def construct_route(self, ant, pheromones, variant='P', attemp=10):
        pheromones = pheromones.copy()
        pheromones[:, np.unique(ant)] = -np.inf

        for _ in range(attemp):
            route = self._construct_route(deepcopy(ant), pheromones.copy(), variant=variant)
            # print(route)
            if route is not None:
                return route
        return None

    def update_pheromones(self, best_ants, pheromones):
        best_ants = gen_tours(best_ants)
        for tour in best_ants:
            np.add.at(pheromones, (tour[:-1], tour[1:]), 1)
        pheromones *= (1 - self.rho)
        return pheromones

    def __call__(self, n_epoch=500, variant='P', is_local_search = True, verbose=False):
        assert self.has_instance, "please import instance first"
        
        ants, pheromones = self.init_population(variant)
        best_obj = -np.inf
        elitist_epochs = []
        for epoch in range(n_epoch):

            # Constructing tour of ants
            elitist_ants = run_parallel2(self.construct_route, ants, pheromones=pheromones, variant=variant)
            elitist_ants = [ant for ant in elitist_ants if ant is not None]

            # F3: every ant failed to construct a feasible solution this epoch ->
            # nothing to score / deposit. Skip rather than crash get_best([]).
            if not elitist_ants:
                continue

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

        # F3: guard the all-epochs-empty case before scoring the elite archive.
        assert elitist_epochs, "ACO failed to construct any feasible solution"
        # elitist_epochs is a list of (possibly ragged) 1-D flat actions; get_best
        # -> calc_obj -> get_Ts iterates element-wise, so ragged lengths are fine.
        _, best_action = self.get_best(elitist_epochs)
        # F1: expose the winning arc order for downstream logging (cf. ILS/EA).
        self._last_best_action = np.asarray(best_action)
        return get_Ts(self.vars, actions=best_action[None])