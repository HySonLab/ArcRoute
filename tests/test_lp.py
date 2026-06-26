import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from solvers.lp import build_graph, solve_milp_p, solve_milp_u


# --------------------------------------------------------------------------- #
# Small hand-built instances (no .npz needed)
# --------------------------------------------------------------------------- #
def tiny_instance(P=2, M=2, C=10.0):
    """Directed cycle 0->1->2->3->0, every arc required.

    classes: (0,1),(2,3) -> class 1 ; (1,2),(3,0) -> class 2.
    demand 1 each, service 2, traversal 1 (service = 2*traversal, Ha 2024).
    Strongly connected by the cycle alone => deadheading is always feasible.
    """
    req = np.array([
        [0, 1, 1, 1, 2, 1],
        [2, 3, 1, 1, 2, 1],
        [1, 2, 1, 2, 2, 1],
        [3, 0, 1, 2, 2, 1],
    ], dtype=float)
    nonreq = np.zeros((0, 6), dtype=float)
    return {'req': req, 'nonreq': nonreq, 'P': P, 'M': M, 'C': C}


REAL = os.path.join(os.path.dirname(__file__), '..',
                    'data', 'ood', 'osm_cityB', '40', '38_14_859.npz')


# --------------------------------------------------------------------------- #
# TestBuildGraph
# --------------------------------------------------------------------------- #
class TestBuildGraph(unittest.TestCase):
    def setUp(self):
        self.G = build_graph(tiny_instance())

    def test_dummy_node_not_in_V(self):
        self.assertNotIn(self.G['dummy'], self.G['nodes'])
        self.assertEqual(self.G['dummy'], max(self.G['nodes']) + 1)

    def test_Af_At_zero_cost(self):
        for a in self.G['A_f'] + self.G['A_t']:
            self.assertEqual(self.G['cost'][a], 0.0)

    def test_Af_At_cover_Vt(self):
        target = self.G['V_t']  # already includes depot
        self.assertEqual({v for (_, v) in self.G['A_f']}, target)
        self.assertEqual({v for (v, _) in self.G['A_t']}, target)

    def test_A_prime_contains_A(self):
        self.assertTrue(set(self.G['A']).issubset(set(self.G['A_prime'])))
        # A' = A U A_f U A_t
        self.assertEqual(set(self.G['A_prime']),
                         set(self.G['A']) | set(self.G['A_f']) | set(self.G['A_t']))


# --------------------------------------------------------------------------- #
# helpers for solution inspection
# --------------------------------------------------------------------------- #
def served_by_p(model, info):
    """MILP-P: dict required arc -> list of vehicles servicing it."""
    G, x, M = info['G'], info['x'], info['M']
    out = {}
    for a in G['req_arcs']:
        out[a] = [m for m in M if model.getVal(x[(m, a)]) > 0.5]
    return out


def served_by_u(model, info):
    """MILP-U: dict required arc -> list of (vehicle, level) servicing it."""
    G, x, M, levels = info['G'], info['x'], info['M'], info['levels']
    out = {}
    for a in G['req_arcs']:
        out[a] = [(m, h) for m in M for h in levels
                  if model.getVal(x[(m, a, h)]) > 0.5]
    return out


# --------------------------------------------------------------------------- #
# TestMILPP_SmallInstance
# --------------------------------------------------------------------------- #
class TestMILPP_SmallInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.model, cls.info = solve_milp_p(
            tiny_instance(), time_limit=60, threads=4, return_model=True)

    def test_milp_p_solves_to_optimality(self):
        self.assertEqual(self.model.getStatus(), 'optimal')
        self.assertIsNotNone(self.T)
        self.assertTrue(np.all(self.T > 0))

    def test_milp_p_each_arc_served_once(self):
        served = served_by_p(self.model, self.info)
        for a, vs in served.items():
            self.assertEqual(len(vs), 1, msg=f'arc {a} served by {vs}')

    def test_milp_p_capacity_respected(self):
        G, x, M = self.info['G'], self.info['x'], self.info['M']
        for m in M:
            load = sum(G['q'][a] * self.model.getVal(x[(m, a)])
                       for a in G['req_arcs'])
            self.assertLessEqual(load, G['Q'] + 1e-6)

    def test_milp_p_priority_respected(self):
        # feasible solution; precedence variant: completion times non-decreasing
        self.assertTrue(np.all(self.T > 0))
        self.assertLessEqual(self.T[0], self.T[1] + 1e-6)

    def test_milp_p_returns_numpy_array(self):
        self.assertIsInstance(self.T, np.ndarray)
        self.assertEqual(self.T.shape, (2,))


# --------------------------------------------------------------------------- #
# TestMILPU_SmallInstance
# --------------------------------------------------------------------------- #
class TestMILPU_SmallInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Tu, cls.modelu, cls.infou = solve_milp_u(
            tiny_instance(), time_limit=60, threads=4, return_model=True)
        cls.Tp = solve_milp_p(tiny_instance(), time_limit=60, threads=4)

    def test_milp_u_solves_to_optimality(self):
        self.assertEqual(self.modelu.getStatus(), 'optimal')
        self.assertIsNotNone(self.Tu)
        self.assertTrue(np.all(self.Tu > 0))

    def test_milp_u_each_arc_served_once(self):
        served = served_by_u(self.modelu, self.infou)
        for a, mh in served.items():
            self.assertEqual(len(mh), 1, msg=f'arc {a} served by {mh}')

    def test_milp_u_capacity_respected(self):
        G, x, M, levels = (self.infou['G'], self.infou['x'],
                           self.infou['M'], self.infou['levels'])
        for m in M:
            load = sum(G['q'][a] * self.modelu.getVal(x[(m, a, h)])
                       for a in G['req_arcs'] for h in levels)
            self.assertLessEqual(load, G['Q'] + 1e-6)

    def test_milp_u_leq_milp_p(self):
        # weighted lexicographic objective: U has >= the feasible set of P
        w = np.array([10.0 ** (3 * (len(self.Tu) - k))
                      for k in range(1, len(self.Tu) + 1)])
        self.assertLessEqual(float(self.Tu @ w), float(self.Tp @ w) + 1e-6)


# --------------------------------------------------------------------------- #
# TestMILPP_RealInstance
# --------------------------------------------------------------------------- #
@unittest.skipUnless(os.path.exists(REAL), 'real instance not available')
class TestMILPP_RealInstance(unittest.TestCase):
    def test_milp_p_real_instance(self):
        T = solve_milp_p(REAL, time_limit=60, threads=8)
        self.assertIsNotNone(T)
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(T > 0))
        # precedence: completion times non-decreasing across classes
        self.assertLessEqual(T[0], T[1] + 1e-6)
        self.assertLessEqual(T[1], T[2] + 1e-6)


if __name__ == '__main__':
    unittest.main()
