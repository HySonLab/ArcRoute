import os
import sys
import unittest
from collections import Counter
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.ops import gen_tours
from solvers.meta import ACOHCARP


def make_symmetric_adj(n, seed=0):
    """Random symmetric arc-to-arc matrix with zero diagonal (n includes depot)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float64)


def arc_multiset(action):
    """Counter of all non-depot arcs in a flat action (or list of routes)."""
    arr = np.asarray(action).ravel()
    return Counter(int(a) for a in arr if int(a) != 0)


def make_instance(nv=2, seed=5, **aco_kwargs):
    """Build an ACOHCARP with attributes set directly (no .npz needed).

    6 arcs + depot; classes 1,1,2,2,3,3; equal demands (0.3 each).
    """
    n = 7  # 6 arcs + depot
    adj = make_symmetric_adj(n, seed=seed)
    clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    demands = np.full(n, 0.3, dtype=np.float64)
    demands[0] = 0.0
    s = np.zeros(n, dtype=np.float64)   # service times
    d = np.zeros(n, dtype=np.float64)   # traversal

    al = ACOHCARP(**aco_kwargs)
    al.dms = adj
    al.P = [1, 2, 3]
    al.M = list(range(nv))
    al.demands = demands.astype(np.float32)
    al.clss = clss.astype(np.int32)
    al.s = s
    al.d = d
    al.nv = nv
    al.nseq = n
    al.max_len = al.nseq + (al.nv - 2)
    al.vars = {
        'adj': al.dms,
        'service_time': al.s,
        'clss': al.clss,
        'demand': al.demands,
        'nv': al.nv,
    }
    al.has_instance = True
    return al


# the full arc set every individual must cover (indices 1..6, once each)
EXPECTED_ARCS = Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})


# --------------------------------------------------------------------------- #
# TestInitPopulation
# --------------------------------------------------------------------------- #
class TestInitPopulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.al = make_instance(nv=2, seed=5, n_ant=15)
        self.ants, self.pheromones = self.al.init_population(variant='P')

    def test_init_population_returns_ants_and_pheromones(self):
        out = self.al.init_population(variant='P')
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_init_population_ant_count(self):
        self.assertEqual(len(self.ants), self.al.n_ant)

    def test_init_population_pheromone_shape(self):
        # pheromones is zeros_like(dms) -> same shape as adjacency (nseq, nseq)
        self.assertEqual(self.pheromones.shape, self.al.dms.shape)
        self.assertEqual(self.pheromones.shape, (self.al.nseq, self.al.nseq))

    def test_init_population_variant_P_pheromone_mask(self):
        # For row i (arc class c_i > 0), columns j with clss[j] < c_i are -inf:
        # in P-variant you may not transition to a strictly higher-priority arc.
        _, ph = self.al.init_population(variant='P')
        clss = self.al.clss
        for i in range(self.al.nseq):
            for j in range(self.al.nseq):
                if clss[j] < clss[i]:
                    self.assertTrue(np.isneginf(ph[i, j]),
                                    msg=f"ph[{i},{j}] should be -inf "
                                        f"(clss {clss[j]} < {clss[i]})")
        # at least one such masked entry exists for this instance
        self.assertTrue(np.isneginf(ph).any())

    def test_init_population_variant_U_no_mask(self):
        _, ph = self.al.init_population(variant='U')
        self.assertFalse(np.isneginf(ph).any())
        self.assertTrue(np.all(ph == 0.0))

    def test_init_population_ants_start_at_depot(self):
        # each ant is a list of partial tours, each starting with depot 0
        for ant in self.ants:
            self.assertLessEqual(len(ant), len(self.al.M))
            for tour in ant:
                self.assertEqual(tour[0], 0)


# --------------------------------------------------------------------------- #
# TestGetNext
# --------------------------------------------------------------------------- #
class TestGetNext(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.al = make_instance(nv=2, seed=5, n_ant=10)

    def test_get_next_returns_valid_arc(self):
        mask = np.zeros(self.al.nseq, dtype=np.float64)
        # block depot column so we get a real arc
        mask[0] = -np.inf
        nxt = self.al.get_next([0], mask.copy())
        self.assertIsNotNone(nxt)
        self.assertTrue(self.al.is_valid_once([0, nxt]))

    def test_get_next_returns_none_when_all_infeasible(self):
        mask = np.full(self.al.nseq, -np.inf, dtype=np.float64)
        self.assertIsNone(self.al.get_next([0], mask))

    def test_get_next_respects_capacity(self):
        # partial path already at capacity: demands 0.3*3 = 0.9; adding any
        # 0.3-demand arc -> 1.2 > 1.0 -> get_next must give up (None).
        mask = np.zeros(self.al.nseq, dtype=np.float64)
        mask[0] = -np.inf
        path = [0, 1, 2, 3]  # load 0.9
        self.assertFalse(self.al.is_valid_once(path + [4]))
        nxt = self.al.get_next(path, mask.copy())
        self.assertIsNone(nxt)

    def test_get_next_excludes_depot_when_masked(self):
        # construct_route masks column 0, so depot is never returned as next.
        mask = np.zeros(self.al.nseq, dtype=np.float64)
        mask[0] = -np.inf
        for _ in range(30):
            nxt = self.al.get_next([0], mask.copy())
            self.assertNotEqual(nxt, 0)


# --------------------------------------------------------------------------- #
# TestConstructRoute
# --------------------------------------------------------------------------- #
class TestConstructRoute(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)
        self.al = make_instance(nv=2, seed=5, n_ant=10)
        self.ants, self.pheromones = self.al.init_population(variant='P')

    def _first_valid_route(self):
        for ant in self.ants:
            r = self.al.construct_route(ant, self.pheromones, attemp=20)
            if r is not None:
                return r
        return None

    def test_construct_route_conserves_arcs(self):
        route = self._first_valid_route()
        self.assertIsNotNone(route)
        self.assertEqual(arc_multiset(route), EXPECTED_ARCS)

    def test_construct_route_feasible(self):
        route = self._first_valid_route()
        self.assertIsNotNone(route)
        self.assertTrue(self.al.is_valid(route))
        for tour in gen_tours(route):
            self.assertLessEqual(self.al.demands[tour].sum(), 1.0 + 1e-9)

    def test_construct_route_returns_none_when_infeasible(self):
        # if every _construct_route attempt fails, construct_route returns None
        with mock.patch.object(self.al, '_construct_route', return_value=None):
            out = self.al.construct_route(self.ants[0], self.pheromones, attemp=5)
        self.assertIsNone(out)

    def test_construct_route_is_flat_action(self):
        route = self._first_valid_route()
        self.assertIsNotNone(route)
        self.assertEqual(np.asarray(route).ndim, 1)


# --------------------------------------------------------------------------- #
# TestUpdatePheromones
# --------------------------------------------------------------------------- #
class TestUpdatePheromones(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.al = make_instance(nv=2, seed=5, n_ant=10, rho=0.5)

    def test_update_pheromones_shape_preserved(self):
        ph = np.zeros((self.al.nseq, self.al.nseq), dtype=np.float64)
        best = np.array([1, 2, 3, 0, 4, 5, 6], dtype=np.int32)
        out = self.al.update_pheromones(best, ph)
        self.assertEqual(out.shape, ph.shape)

    def test_update_pheromones_evaporation(self):
        # entries that already carried pheromone but are NOT on the deposited
        # tour are scaled by (1 - rho).
        ph = np.zeros((self.al.nseq, self.al.nseq), dtype=np.float64)
        ph[5, 6] = 10.0  # pre-existing, will not receive a deposit below
        best = np.array([1, 2, 3], dtype=np.int32)  # tour [0,1,2,3,0]
        out = self.al.update_pheromones(best, ph)
        self.assertAlmostEqual(out[5, 6], 10.0 * (1 - self.al.rho))

    def test_update_pheromones_deposit(self):
        ph = np.zeros((self.al.nseq, self.al.nseq), dtype=np.float64)
        best = np.array([1, 2, 3], dtype=np.int32)  # gen_tours -> [0,1,2,3,0]
        out = self.al.update_pheromones(best, ph)
        # edges (0,1),(1,2),(2,3),(3,0) each got +1 then *(1-rho)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            self.assertAlmostEqual(out[a, b], 1.0 * (1 - self.al.rho))


# --------------------------------------------------------------------------- #
# TestACOEndToEnd
# --------------------------------------------------------------------------- #
class TestACOEndToEnd(unittest.TestCase):
    def test_aco_result_feasible(self):
        np.random.seed(7)
        al = make_instance(nv=2, seed=5, n_ant=15)
        T = al(n_epoch=8, variant='P')
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T > 0), msg=f"expected all T_k > 0, got {T}")

    def test_aco_last_best_action_set(self):
        np.random.seed(8)
        al = make_instance(nv=2, seed=5, n_ant=15)
        al(n_epoch=8, variant='P')
        self.assertTrue(hasattr(al, '_last_best_action'))
        self.assertIsNotNone(al._last_best_action)
        self.assertEqual(arc_multiset(al._last_best_action), EXPECTED_ARCS)

    def test_aco_with_local_search(self):
        np.random.seed(9)
        al = make_instance(nv=2, seed=5, n_ant=15)
        T = al(n_epoch=5, variant='P', is_local_search=True)
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T > 0))

    def test_aco_without_local_search(self):
        np.random.seed(10)
        al = make_instance(nv=2, seed=5, n_ant=15)
        T = al(n_epoch=5, variant='P', is_local_search=False)
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T > 0))

    def test_aco_empty_epochs_guarded(self):
        # if construction always fails, every epoch is skipped; the final
        # assert must fire instead of a confusing matmul error.
        np.random.seed(11)
        al = make_instance(nv=2, seed=5, n_ant=10)
        al.construct_route = lambda ant, pheromones, variant='P', attemp=10: None
        with self.assertRaises(AssertionError):
            al(n_epoch=3, variant='P')


if __name__ == '__main__':
    unittest.main()
