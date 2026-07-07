import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.random import random, permutation

from utils.ops import run_parallel2
from utils.ops import gen_tours, deserialize_tours, convert_prob
from utils.ops import centered_lex_rank
from solvers.evaluator import evaluate_batch as get_Ts
from utils.local_search import ls
from utils.ils_operators import perturbate, accept
from copy import deepcopy

class BaseHCARP:
    def __init__(self):
        self.has_instance = False
        # Active problem variant ('P' precedence / 'U' upgrading). A sane
        # default; every algorithm's __call__ overwrites it from its `variant`
        # parameter so calc_obj/get_Ts score solutions with the right decoder.
        self.variant = 'P'

    def _load_data(self, dms, P, M, demands, clss, s, d, edge_indxs):
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

    def import_instance(self, f, M=None):
        from utils.ops import iter_instances
        self._load_data(*next(iter_instances(f, M=M)))
    
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
    
    # Lexicographic dominance base for calc_obj's Horner encoding
    # obj = sum_p T_p * LEX_C**(P-1-p). Strict lex order (T1 > T2 > T3) is
    # preserved as long as every T_p stays well below LEX_C: benchmark logs
    # (logs/bm_*.out) show max T3 ~ 68, so LEX_C = 1e4 gives a ~150x safety
    # margin — a T1 improvement of d dominates the lower levels whenever
    # d > T_max/LEX_C ~ 0.007, far below any meaningful T1 difference, while
    # keeping the total (~68e8 for P=3) small enough that float64 still
    # resolves T3 differences down to ~1e-6. The old weights [1e3,1e1,1e-1]
    # (100x per level) let a 0.01 T1 loss be outvoted by T2/T3 gains.
    LEX_C = 1e4

    def calc_obj(self, actions):
        rewards = get_Ts(self.vars, actions=actions, variant=self.variant)  # (N, P), LOWER is better
        P = rewards.shape[1]
        w = self.LEX_C ** np.arange(P - 1, -1, -1, dtype=np.float64)
        obj = rewards @ w
        return -obj                                     # higher = better
    
    def calc_len(self, path):
        return -(self.s[path].sum() + self.dms[path[:-1], path[1:]].sum())
    
    def get_best(self, samples):
        obj = self.calc_obj(samples)
        idx = obj.argmax()
        return obj[idx], samples[idx]
    
    def ls_lex_guarded(self, actions, max_len=None):
        """Lex-guarded local search (cf. solvers/rl.py's hybrid funnel).

        ``ls`` minimises total deadheading cost, NOT the lexicographic
        (T1, T2, T3) objective. Under variant 'P' its class-restricted moves
        keep routes class-sorted so the hierarchy mostly survives, but under
        'U' the unrestricted any-class neighbourhood happily drags class-1
        arcs deep into routes — trading T1 away for deadhead (measured on
        osm_test_small: T1 +40..80% right after the first ``ls`` pass).

        Guard: run ``ls`` and keep each sample's result only if ``calc_obj``
        (lex-weighted, higher = better) does not get worse than the pre-LS
        input. Under 'U' a class-restricted 'P' ``ls`` arm is tried as well —
        every P-move is U-feasible, so variant U retains at least P's
        local-search power even when the unrestricted arm is rejected.

        Parameters
        ----------
        actions : list of flat 0-separated arc-order arrays (no ``None``s)
        max_len : re-serialisation width for post-LS routes
                  (default ``self.max_len``; warm starts may need more)

        Returns
        -------
        (actions_list, obj_array) — per-sample lex-best over all arms; the
        LS arms are preferred on exact ties so deadhead gains are kept.
        """
        if max_len is None:
            max_len = self.max_len
        actions = [np.asarray(a, dtype=np.int32) for a in actions]
        variants = ('U', 'P') if self.variant == 'U' else (self.variant,)
        arms = []
        for var in variants:
            tours = ls(self.vars, variant=var, actions=actions)
            arms.append([deserialize_tours(t, max_len) for t in tours])
        arms.append(actions)  # pre-LS fallback, last so LS arms win ties
        objs = self._calc_obj_arms(arms)                       # (n_arms, N)
        best = objs.argmax(axis=0)  # first max -> LS arm preferred on ties
        idx = np.arange(len(actions))
        return [arms[k][i] for i, k in enumerate(best)], objs[best, idx]

    def _calc_obj_arms(self, arms):
        """``calc_obj`` for several arms of the same sample batch, scoring each
        distinct action once per sample.

        Exact: the evaluator's decode depends only on an action's nonzero arc
        order (every path in ``SolutionEvaluator`` starts from ``a[a != 0]``;
        0-separators and padding are ignored), so two actions with the same
        nonzero order provably receive the same T — e.g. when the U and P
        local-search arms converge to the same tours, or an arm returns its
        input unchanged. Values are identical to scoring every arm in full.
        """
        n = len(arms[0])
        objs = np.empty((len(arms), n), dtype=np.float64)
        memo = [{} for _ in range(n)]
        for k, arm in enumerate(arms):
            pending, pending_actions = [], []
            for i, act in enumerate(arm):
                a = np.asarray(act)
                key = a[a != 0].astype(np.int64, copy=False).tobytes()
                if key in memo[i]:
                    objs[k, i] = memo[i][key]
                else:
                    pending.append((i, key))
                    pending_actions.append(act)
            if pending_actions:
                vals = self.calc_obj(pending_actions)
                for (i, key), val in zip(pending, vals):
                    objs[k, i] = val
                    memo[i][key] = val
        return objs

    def get_once(self, M, clss, attemp=10):
        # Bin-packing lower bound: if the eligible arcs' total demand exceeds
        # the fleet's total capacity (1.0 per vehicle, demands are normalised
        # by C), no assignment can serve them all — every attempt below would
        # deterministically fail after burning ~attemp full construction
        # loops. This dominates EAHCARP.cross_over on tight instances (load
        # factor ~0.92): parents' first routes overlap, so the leftover arcs
        # routinely exceed the len(M)-2 = 1 vehicle _cross_over gives them —
        # measured 96% of get_once calls / 88% of EA wall time on
        # osm_test_small before this check. Skipping is outcome-identical
        # (the doomed calls returned None anyway); it only shortens the
        # np.random stream those futile loops consumed, so re-seeded
        # trajectories differ draw-by-draw but the process distribution is
        # unchanged. The 1e-6 slack keeps borderline-feasible sums (float32
        # summation noise) on the "try it" side.
        eligible = np.isin(clss, self.P)
        if self.demands[eligible].sum() > len(M) + 1e-6:
            return None
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
        self.variant = variant
        assert self.has_instance, "please import instance first"
        actions = [self.get_once(self.M, self.clss) for _ in range(num_sample)]
        actions = [a for a in actions if a is not None]
        assert actions, "construction failed to produce a feasible solution"
        # Lex-guarded local search (see BaseHCARP.ls_lex_guarded): the raw
        # deadhead-only ``ls`` must never lex-worsen (T1,T2,T3) — under
        # variant U it otherwise trades T1 away for deadhead.
        acts, objs = self.ls_lex_guarded(actions)
        best = acts[int(objs.argmax())]
        return get_Ts(self.vars, actions=best[None], variant=self.variant)
    
class ILSHCARP(BaseHCARP):
    """Iterated Local Search for HDCARP.

    Constructive sampling (``get_once``) seeds an initial solution that is
    refined by the inter/intra ``ls`` operators; the ILS loop then alternates a
    feasibility-preserving ``perturbate`` kick with a full local search and an
    acceptance test (``'best'`` improvement-only or ``'sa'`` simulated
    annealing). The objective (``calc_obj``, higher = better) is the negated
    lexicographically-weighted (T1, T2, T3).

    Every ``ls`` application goes through ``ls_lex_guarded``: the deadhead-
    minimising local search is kept only when it does not lex-worsen the
    solution (under variant 'U' it otherwise trades T1 for deadhead), with a
    class-restricted P arm as an extra U-feasible neighbourhood.
    """

    def __init__(self, strength=3, accept_mode='best', T_init=0.05, T_final=0.001):
        super().__init__()
        self.strength = strength
        self.accept_mode = accept_mode
        self.T_init = T_init
        self.T_final = T_final

    def __call__(self, max_iter=200, variant='P', num_init_sample=5,
                 seed=None, verbose=False):
        self.variant = variant
        assert self.has_instance, "please import instance first"
        rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

        # 1. constructive seeds -> lex-guarded local search -> best as s0.
        # The guard matters: unguarded, the deadhead-only ``ls`` poisons
        # the pool before get_best ever scores the raw constructions
        # (under U it wrecks T1), and the lex-strict acceptance below can
        # then only ever choose among already-degraded candidates.
        samples = [self.get_once(self.M, self.clss) for _ in range(num_init_sample)]
        samples = [s for s in samples if s is not None]
        assert samples, "construction failed to produce a feasible solution"
        acts, objs = self.ls_lex_guarded(samples)
        k = int(objs.argmax())
        cur_obj, cur_action = float(objs[k]), acts[k]
        cur_routes = gen_tours(cur_action)

        best_action = cur_action.copy()
        best_obj = cur_obj
        capacity = 1.0
        # Defensive: never re-serialise narrower than the current action (the
        # constructive path always yields exactly self.max_len, so this is a
        # no-op there).
        max_len = max(self.max_len, len(cur_action))

        # 2. ILS loop: perturb -> lex-guarded local search -> accept
        for it in range(max_iter):
            pert = perturbate(cur_routes, self.dms, self.demands, self.clss,
                              capacity, variant, rng, strength=self.strength)
            pert_action = deserialize_tours(pert, max_len)
            cands, cobjs = self.ls_lex_guarded([pert_action], max_len=max_len)
            cand_action = cands[0]
            cand_obj = float(cobjs[0])

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
        return get_Ts(self.vars, actions=best_action[None], variant=self.variant)


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
        # Lex-guarded local search (see BaseHCARP.ls_lex_guarded): the raw
        # deadhead-only ``ls`` must never lex-worsen (T1,T2,T3) — under
        # variant U it otherwise trades T1 away for deadhead. The guard reads
        # self.variant, which __call__ sets to the same ``variant`` threaded
        # here through operate(); the parameter is kept for call-site
        # compatibility and asserted consistent.
        assert variant == self.variant, (
            f"mutate called with variant={variant!r} but self.variant="
            f"{self.variant!r}; ls_lex_guarded scores with self.variant")
        acts, _ = self.ls_lex_guarded([tours])
        return acts[0]
    
    def get_parent(self, population, prob):
        return population[self.get_idx(prob, strategy='sampling', size=self.pathnament_size)]

    def make_pair(self, prob, population):
        """Selection + crossover for one parent pair; mutation is deferred.

        Returns ``(child1, child2, do_mutate)``. The mutation gate is drawn
        HERE (same np.random position as the historical inline ``operate``)
        but the lex-guarded local search itself consumes no randomness, so
        the caller may apply it later — batched across the whole generation —
        with a byte-identical result (see __call__).
        """
        # CROSSOVER
        parent1 = self.get_parent(population, prob)
        parent2 = self.get_parent(population, prob)

        # default to the parents so child1/child2 are always defined
        child1, child2 = parent1, parent2
        if random() < self.crossover_rate:
            child1 = self.cross_over(parent1, parent2)
            child2 = self.cross_over(parent2, parent1)

        return child1, child2, random() < self.mutation_rate

    def operate(self, prob, population, variant):
        child1, child2, do_mutate = self.make_pair(prob, population)
        # MUTATION
        if do_mutate:
            child1 = self.mutate(child1, variant)
            child2 = self.mutate(child2, variant)
        return child1, child2

    def __call__(self, n_epoch=50, variant='P', verbose=False):
        self.variant = variant
        assert self.has_instance, "please import instance first"

        population = self.init_population()
        for epoch in range(n_epoch):
            # selecting two of the best options we have (elitism)
            obj = self.calc_obj(population)
            idxs = np.argsort(obj)
            new_population = [population[idxs[-1]], population[idxs[-2]]]
            # Roulette-wheel selection: softmax over the exact-lex calc_obj would
            # collapse to near-one-hot (LEX_C-scale dynamic range), killing the
            # fitness-proportional diversity EA relies on. Rank instead: the
            # centered hierarchical midrank is bounded in [-1, +1] regardless of
            # the population's absolute T spread. gamma=0.5 matches GRPO's
            # adv_gamma default — same multi-priority-combination philosophy.
            T_population = get_Ts(self.vars, actions=population, variant=self.variant)
            prob = convert_prob(centered_lex_rank(T_population, gamma=0.5))
            nb = len(population) // 2 - 1
            # Build the whole generation first (selection + crossover), then
            # lex-guard all children flagged for mutation in ONE batched
            # ls_lex_guarded call instead of ~nb per-child calls. This is
            # byte-identical to the historical per-pair operate() loop:
            # make_pair draws the exact same np.random sequence (mutation
            # never consumed randomness — ls and calc_obj are deterministic),
            # children don't feed back into this epoch's selection, and
            # ls_lex_guarded/_calc_obj_arms treat every sample independently.
            # Batching only amortises the per-call Python/evaluator overhead.
            pairs = run_parallel2(self.make_pair, [prob]*nb, population=population)
            children, mut_idx = [], []
            for child1, child2, do_mutate in pairs:
                if do_mutate:
                    mut_idx.extend((len(children), len(children) + 1))
                children.extend((child1, child2))
            if mut_idx:
                acts, _ = self.ls_lex_guarded([children[k] for k in mut_idx])
                for k, act in zip(mut_idx, acts):
                    children[k] = act
            new_population += children
            population = new_population
            if verbose:
                if epoch % 10 == 0:
                    obj = self.calc_obj(population)
                    idx = np.argsort(obj)[-1]
                    print(f"epoch {epoch}:", obj[idx])

        routes = population[self.calc_obj(population).argmax()]
        self._last_best_action = np.asarray(routes)
        return get_Ts(self.vars, actions=routes[None], variant=self.variant)

class ACOHCARP(BaseHCARP):
    # Small positive floor added to the pheromone trail before raising it to
    # `alpha`. Pheromones start at exact zero (see `init_population`), so
    # without this floor `tau**alpha == 0` for every candidate on the very
    # first construction pass, which would silently fall back to visibility-
    # only (or, before this fix existed, to a uniform draw) regardless of how
    # `alpha`/`beta` are tuned.
    TAU0 = 1.0
    # Deadheading-cost floor so eta = 1/adj stays finite for zero-distance
    # arc pairs (colocated arcs / immediate depot hops).
    ETA_EPS = 1e-6

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
        """Sample the next arc via the standard ACO transition rule

            p_ij ∝ tau_ij**alpha * eta_ij**beta

        ``mask`` is the pheromone row `tau` for the ant's current position
        (``-inf`` entries are structurally forbidden -- class-mask/priority-
        frontier/visited exclusions already baked in by the caller) and
        ``eta = 1 / adj`` is the inverse-deadheading-cost heuristic, read
        from the same distance matrix (``self.dms``) `ls`/the evaluator use
        elsewhere -- no new distance representation. Previously this
        softmaxed the raw pheromone row alone (`alpha`/`beta` were dead
        fields): with ~100+ candidate arcs at large.npz scale and pheromone
        counts confined to a narrow range by evaporation, that is close to a
        uniform random draw and never sharpens (see the flat epoch-best
        traces in the investigation this fixes). Mixing in `eta` gives
        construction an actual cost signal from epoch 0.
        """
        from_arc = int(t[-1])
        eta = 1.0 / (np.asarray(self.dms[from_arc], dtype=np.float64) + self.ETA_EPS)
        for _ in range(attemp):
            allowed = mask >= 0
            if not allowed.any():
                return None
            # Dummy positive fill (log=0) at forbidden slots -- discarded by
            # the np.where below, just avoids a log(-inf)/log(negative)
            # RuntimeWarning on values we never use.
            tau = np.where(allowed, mask + self.TAU0, 1.0)
            logits = np.where(allowed,
                               self.alpha * np.log(tau) + self.beta * np.log(eta),
                               -np.inf)
            next = self.get_idx(convert_prob(logits), size=1, strategy='sampling')
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

    def _search(self, n_epoch, variant, is_local_search, verbose):
        """Core ant-colony loop for one variant; returns the winning flat action.

        Factored out of ``__call__`` so the variant-U P-anchor arm (see
        ``__call__``) can run a genuine ACO-P search -- construction,
        pheromone feedback and elite selection all scored consistently as
        'P' via ``self.variant`` -- without duplicating this loop or
        entangling it with the outer variant. Callers must set
        ``self.variant`` to ``variant`` before calling (``ls_lex_guarded``/
        ``get_best`` read it).
        """
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

            # refine tours by lex-guarded local search (see
            # BaseHCARP.ls_lex_guarded): the raw deadhead-only ``ls`` must
            # never lex-worsen (T1,T2,T3) — under variant U it otherwise
            # trades T1 away for deadhead. Reuse the guard's objectives for
            # the elite pick instead of re-scoring via get_best.
            if is_local_search:
                elitist_ants, objs = self.ls_lex_guarded(elitist_ants)
                k = int(objs.argmax())
                ant_obj, best_ants = objs[k], elitist_ants[k]
            else:
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
        return np.asarray(best_action)

    def __call__(self, n_epoch=500, variant='P', is_local_search = True, verbose=False):
        self.variant = variant
        assert self.has_instance, "please import instance first"

        best_action = self._search(n_epoch, variant, is_local_search, verbose)
        best_T = get_Ts(self.vars, actions=best_action[None], variant=self.variant)

        if variant == 'U':
            # P-anchor arm (cf. BaseHCARP.ls_lex_guarded's LS arms and
            # RLHybHCARP's greedy-anchor arm in solvers/rl.py): variant U's
            # construction drops both the pheromone class-mask and the
            # priority frontier that P's construction relies on (see
            # `init_population`/`_construct_route`), so at scale it is much
            # closer to an unguided random-permutation search and can end up
            # lex-WORSE than a plain ACO-P run on the same instance (measured
            # on data/test_bm/large.npz[71] pre-fix: ACO-U 22.16/50.56/86.51
            # vs ACO-P 17.11/44.33/69.46 -- and ACO-P's own winning action
            # rescored under the U evaluator reproduces its own score
            # exactly). Run the real ACO-P search as a second arm -- same
            # construction/pheromone/elitism loop, `self.variant` flipped to
            # 'P' for the duration so it is a faithful standalone ACO-P run,
            # not a hybrid scored under U -- then rescore its winning action
            # under the U evaluator. U's decode is provably >= as good as P's
            # decode of the same action (`SolutionEvaluator.__call__`'s U
            # branch returns the lex-better of its own segment+LPT decode and
            # `_build_P`), so this arm can never push the final answer below
            # what a standalone ACO-P call would report. Take the lex-best of
            # the two arms -- the same take-the-argmax-over-arms pattern as
            # `ls_lex_guarded`/`upgrade_to_u`.
            self.variant = 'P'
            p_action = self._search(n_epoch, 'P', is_local_search, verbose)
            self.variant = 'U'
            p_as_u_T = get_Ts(self.vars, actions=p_action[None], variant='U')
            if tuple(p_as_u_T[0].tolist()) < tuple(best_T[0].tolist()):
                best_action, best_T = p_action, p_as_u_T

        # F1: expose the winning arc order for downstream logging (cf. ILS/EA).
        self._last_best_action = best_action
        return best_T