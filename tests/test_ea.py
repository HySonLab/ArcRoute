import os
import sys
import unittest
from collections import Counter
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.ops import gen_tours
from solvers.meta import EAHCARP


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


def make_instance(nv=3, seed=5, **ea_kwargs):
    """Build an EAHCARP with attributes set directly (no .npz needed).

    6 arcs + depot; classes 1,1,2,2,3,3; equal demands (0.3 each).
    """
    n = 7  # 6 arcs + depot
    adj = make_symmetric_adj(n, seed=seed)
    clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    demands = np.full(n, 0.3, dtype=np.float64)
    demands[0] = 0.0
    s = np.zeros(n, dtype=np.float64)   # service times
    d = np.zeros(n, dtype=np.float64)   # traversal

    al = EAHCARP(**ea_kwargs)
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
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()

    def test_init_population_size(self):
        self.assertGreater(len(self.pop), 0)
        self.assertLessEqual(len(self.pop), self.al.n_population)

    def test_init_population_no_none(self):
        for ind in self.pop:
            self.assertIsNotNone(ind)

    def test_init_population_all_arcs_present(self):
        for ind in self.pop:
            self.assertEqual(arc_multiset(ind), EXPECTED_ARCS)

    def test_init_population_feasible(self):
        for ind in self.pop:
            self.assertTrue(self.al.is_valid(ind))


# --------------------------------------------------------------------------- #
# TestIsValid (BaseHCARP)
# --------------------------------------------------------------------------- #
class TestIsValid(unittest.TestCase):
    def setUp(self):
        self.al = make_instance(nv=3, seed=5)

    def test_is_valid_feasible(self):
        # two routes, each load 3 * 0.3 = 0.9 <= 1.0
        action = np.array([1, 2, 3, 0, 4, 5, 6], dtype=np.int32)
        self.assertTrue(self.al.is_valid(action))

    def test_is_valid_overcapacity(self):
        # first route has 4 arcs => load 1.2 > 1.0
        action = np.array([1, 2, 3, 4, 0, 5, 6], dtype=np.int32)
        self.assertFalse(self.al.is_valid(action))

    def test_is_valid_ragged_routes(self):
        # routes of different lengths must not crash (regression for the
        # ragged fancy-index bug). 3 / 2 / 1 arcs, all feasible.
        action = np.array([1, 2, 3, 0, 4, 5, 0, 6], dtype=np.int32)
        result = self.al.is_valid(action)
        self.assertIsInstance(result, bool)
        self.assertTrue(result)


# --------------------------------------------------------------------------- #
# TestCrossOver
# --------------------------------------------------------------------------- #
class TestCrossOver(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.p1, self.p2 = self.pop[0], self.pop[1]

    def test_crossover_conserves_arcs(self):
        for _ in range(10):
            child = self.al.cross_over(self.p1, self.p2)
            self.assertEqual(arc_multiset(child), EXPECTED_ARCS)

    def test_crossover_feasible(self):
        for _ in range(10):
            child = self.al.cross_over(self.p1, self.p2)
            self.assertTrue(self.al.is_valid(child))

    def test_crossover_fallback_to_p1(self):
        with mock.patch.object(self.al, '_cross_over', return_value=None):
            child = self.al.cross_over(self.p1, self.p2, attemp=5)
        self.assertTrue(np.array_equal(child, self.p1))

    def test_crossover_nv2_edge_case(self):
        np.random.seed(2)
        al = make_instance(nv=2, seed=5, n_population=20)
        pop = al.init_population()
        child = al.cross_over(pop[0], pop[1])
        self.assertEqual(arc_multiset(child), EXPECTED_ARCS)
        self.assertTrue(al.is_valid(child))


# --------------------------------------------------------------------------- #
# TestMutate
# --------------------------------------------------------------------------- #
class TestMutate(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.ind = self.al.init_population()[0]

    def test_mutate_conserves_arcs(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertEqual(arc_multiset(out), EXPECTED_ARCS)

    def test_mutate_feasible(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertTrue(self.al.is_valid(out))

    def test_mutate_returns_correct_length(self):
        out = self.al.mutate(self.ind, variant='P')
        self.assertEqual(len(out), self.al.max_len)


# --------------------------------------------------------------------------- #
# TestOperate
# --------------------------------------------------------------------------- #
class TestOperate(unittest.TestCase):
    def setUp(self):
        np.random.seed(4)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.prob = np.full(len(self.pop), 1.0 / len(self.pop))

    def test_operate_returns_two_children(self):
        out = self.al.operate(self.prob, self.pop, variant='P')
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_operate_children_conserve_arcs(self):
        for _ in range(10):
            c1, c2 = self.al.operate(self.prob, self.pop, variant='P')
            self.assertEqual(arc_multiset(c1), EXPECTED_ARCS)
            self.assertEqual(arc_multiset(c2), EXPECTED_ARCS)

    def test_operate_always_defined(self):
        # Force every combination of the crossover / mutation branches and make
        # sure both children are always defined (no UnboundLocalError).
        for r in (0.0, 1.0):
            with mock.patch('solvers.meta.random', return_value=r):
                c1, c2 = self.al.operate(self.prob, self.pop, variant='P')
                self.assertIsNotNone(c1)
                self.assertIsNotNone(c2)


# --------------------------------------------------------------------------- #
# TestGetParent
# --------------------------------------------------------------------------- #
class TestGetParent(unittest.TestCase):
    def setUp(self):
        np.random.seed(5)
        self.al = make_instance(nv=3, seed=5, n_population=20)
        self.pop = self.al.init_population()
        self.prob = np.full(len(self.pop), 1.0 / len(self.pop))

    def _matching_indices(self, parent):
        return [i for i, row in enumerate(self.pop)
                if np.array_equal(np.asarray(row), np.asarray(parent))]

    def test_get_parent_returns_valid_index(self):
        parent = self.al.get_parent(self.pop, self.prob)
        self.assertTrue(len(self._matching_indices(parent)) > 0,
                        msg="returned parent is not a member of the population")

    def test_get_parent_distribution(self):
        seen = set()
        for _ in range(1000):
            parent = self.al.get_parent(self.pop, self.prob)
            seen.update(self._matching_indices(parent))
        self.assertEqual(seen, set(range(len(self.pop))))


# --------------------------------------------------------------------------- #
# TestEAEndToEnd
# --------------------------------------------------------------------------- #
class TestEAEndToEnd(unittest.TestCase):
    def test_ea_improves_over_init(self):
        np.random.seed(7)
        al = make_instance(nv=3, seed=5, n_population=20)

        pop = al.init_population()
        rand_ind = pop[np.random.randint(len(pop))]
        base_obj = float(al.calc_obj(rand_ind[None])[0])

        T = al(n_epoch=10, variant='P')
        self.assertEqual(T.shape, (1, 3))
        w = np.array([1e3, 1e1, 1e-1])
        ea_obj = -float(T[0] @ w)

        self.assertGreaterEqual(ea_obj, base_obj - 1e-6)

    def test_ea_result_feasible(self):
        np.random.seed(8)
        al = make_instance(nv=3, seed=5, n_population=20)
        T = al(n_epoch=10, variant='P')
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T > 0),
                        msg=f"expected all T_k > 0, got {T}")

    def test_ea_last_best_action_set(self):
        np.random.seed(9)
        al = make_instance(nv=3, seed=5, n_population=20)
        al(n_epoch=10, variant='P')
        self.assertTrue(hasattr(al, '_last_best_action'))
        self.assertIsNotNone(al._last_best_action)
        self.assertEqual(arc_multiset(al._last_best_action), EXPECTED_ARCS)


if __name__ == '__main__':
    unittest.main()
