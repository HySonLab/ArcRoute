import os
import sys
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.ils_operators import perturbate, accept, routes_to_action
from utils.nb_utils import gen_tours
from solvers.meta import BaseHCARP, ILSHCARP


def make_symmetric_adj(n, seed=0):
    """Random symmetric arc-to-arc matrix with zero diagonal (n includes depot)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float64)


def arc_multiset(routes):
    """Counter of all non-depot arcs across a route list."""
    return Counter(int(a) for r in routes for a in np.asarray(r) if int(a) != 0)


# --------------------------------------------------------------------------- #
# instance factories
# --------------------------------------------------------------------------- #
def make_vars(n=7, seed=5):
    """6 arcs + depot; classes 1,1,2,2,3,3; equal demands."""
    adj = make_symmetric_adj(n, seed=seed)
    clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)[:n]
    demands = np.full(n, 0.3, dtype=np.float64)
    demands[0] = 0.0
    return adj, clss, demands


def make_instance(nv=3, seed=5):
    """Build a BaseHCARP-derived solver with attributes set directly (no .npz)."""
    n = 7  # 6 arcs + depot
    adj, clss, demands = make_vars(n=n, seed=seed)
    s = np.zeros(n, dtype=np.float64)         # service times
    d = np.zeros(n, dtype=np.float64)         # traversal
    al = ILSHCARP(strength=2)
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


# --------------------------------------------------------------------------- #
# 1a. perturbate
# --------------------------------------------------------------------------- #
class TestPerturbate(unittest.TestCase):
    def setUp(self):
        self.adj, self.clss, self.demands = make_vars()
        self.capacity = 1.0
        # two routes, each load 0.9 <= 1.0
        self.routes = [
            np.array([0, 1, 3, 5, 0], dtype=np.int32),
            np.array([0, 2, 4, 6, 0], dtype=np.int32),
        ]

    def test_perturbate_conserves_arcs(self):
        rng = np.random.default_rng(0)
        before = arc_multiset(self.routes)
        for _ in range(20):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            self.assertEqual(arc_multiset(out), before)

    def test_perturbate_capacity_feasible(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            for r in out:
                self.assertLessEqual(self.demands[np.asarray(r)].sum(),
                                     self.capacity + 1e-9)

    def test_perturbate_priority_order_P(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'P', rng, strength=3)
            for r in out:
                interior = [int(a) for a in np.asarray(r)[1:-1]]
                cls_seq = [int(self.clss[a]) for a in interior]
                self.assertEqual(cls_seq, sorted(cls_seq),
                                 msg=f"route {interior} not class-sorted")

    def test_perturbate_U_no_sort(self):
        rng = np.random.default_rng(3)
        before = arc_multiset(self.routes)
        for _ in range(50):
            out = perturbate(self.routes, self.adj, self.demands, self.clss,
                             self.capacity, 'U', rng, strength=3)
            self.assertEqual(arc_multiset(out), before)
            for r in out:
                self.assertLessEqual(self.demands[np.asarray(r)].sum(),
                                     self.capacity + 1e-9)

    def test_perturbate_does_not_mutate_input(self):
        rng = np.random.default_rng(4)
        snapshot = [r.copy() for r in self.routes]
        perturbate(self.routes, self.adj, self.demands, self.clss,
                   self.capacity, 'P', rng, strength=3)
        for orig, snap in zip(self.routes, snapshot):
            self.assertTrue(np.array_equal(orig, snap))

    def test_perturbate_strength_zero_returns_copy(self):
        rng = np.random.default_rng(5)
        out = perturbate(self.routes, self.adj, self.demands, self.clss,
                         self.capacity, 'P', rng, strength=0)
        self.assertEqual(arc_multiset(out), arc_multiset(self.routes))
        for r, o in zip(self.routes, out):
            self.assertTrue(np.array_equal(np.asarray(r), np.asarray(o)))

    def test_perturbate_single_route(self):
        rng = np.random.default_rng(6)
        routes = [np.array([0, 1, 2, 3, 0], dtype=np.int32)]
        before = arc_multiset(routes)
        for _ in range(20):
            out = perturbate(routes, self.adj, self.demands, self.clss,
                             self.capacity, 'U', rng, strength=2)
            # arcs conserved and still in the single route
            self.assertEqual(arc_multiset(out), before)
            self.assertEqual(len(out), 1)


# --------------------------------------------------------------------------- #
# 1b. accept
# --------------------------------------------------------------------------- #
class TestAccept(unittest.TestCase):
    def test_accept_best_mode(self):
        self.assertTrue(accept(0.0, 1.0, 0, 100, mode='best'))   # better
        self.assertTrue(accept(1.0, 1.0, 0, 100, mode='best'))   # equal
        self.assertFalse(accept(1.0, 0.0, 0, 100, mode='best'))  # worse

    def test_accept_sa_always_accepts_better(self):
        for it in range(0, 100, 10):
            self.assertTrue(accept(0.0, 5.0, it, 100, mode='sa'))
        # equal also accepted
        self.assertTrue(accept(2.0, 2.0, 50, 100, mode='sa'))

    def test_accept_sa_sometimes_rejects_worse(self):
        np.random.seed(0)
        n_accept = sum(
            accept(0.0, -10.0, 100, 100, mode='sa',
                   T_init=1e-3, T_final=1e-6)
            for _ in range(100)
        )
        # T -> T_final = 1e-6, delta = -10 => prob ~ 0 => essentially never accept
        self.assertLessEqual(n_accept, 1)


# --------------------------------------------------------------------------- #
# 1c. routes_to_action
# --------------------------------------------------------------------------- #
class TestRoutesToAction(unittest.TestCase):
    def test_roundtrip(self):
        routes = [
            np.array([0, 1, 2, 0], dtype=np.int32),
            np.array([0, 3, 4, 0], dtype=np.int32),
        ]
        action = routes_to_action(routes, nseq=10)
        self.assertTrue(np.array_equal(gen_tours(action), routes)
                        or arc_multiset(gen_tours(action)) == arc_multiset(routes))
        # explicit separator structure: [1,2,0,3,4, ...padding]
        self.assertEqual(action[0], 1)
        self.assertEqual(action[2], 0)
        self.assertEqual(action[3], 3)


# --------------------------------------------------------------------------- #
# Step 2. ILSHCARP end-to-end
# --------------------------------------------------------------------------- #
class TestILS(unittest.TestCase):
    def test_ils_improves_over_construction(self):
        al = make_instance(nv=3, seed=5)

        # greedy construction (no LS) baseline objective
        construct = al.get_once(al.M, al.clss)
        self.assertIsNotNone(construct)
        base_obj = float(al.calc_obj(construct[None])[0])

        T = al(max_iter=50, variant='P', num_init_sample=5, seed=123)
        # get_Ts returns shape (1, P)
        self.assertEqual(T.shape, (1, 3))
        w = np.array([1e3, 1e1, 1e-1])
        ils_obj = -float(T[0] @ w)

        # ILS result must be at least as good as raw construction
        self.assertGreaterEqual(ils_obj, base_obj - 1e-6)

    def test_ils_sa_mode_runs(self):
        al = make_instance(nv=3, seed=7)
        al.accept_mode = 'sa'
        T = al(max_iter=30, variant='U', num_init_sample=3, seed=7)
        self.assertEqual(T.shape, (1, 3))
        self.assertTrue(np.all(T >= 0))


if __name__ == '__main__':
    unittest.main()
