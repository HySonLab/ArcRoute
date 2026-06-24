import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.local_search import (
    calc_swap_delta_intra,
    calc_swap_delta_inter,
    best_swap_intra,
    intra_route_opt,
    best_swap_inter,
    inter_route_opt,
    get_subtour_p,
    get_subtour_u,
    ls,
    lsRL,
)
from utils.nb_utils import gen_tours, deserialize_tours


def make_symmetric_adj(n, seed=0):
    """Random symmetric arc-to-arc matrix with zero diagonal (n includes depot)."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float64)


def route_cost(route, adj):
    route = np.asarray(route)
    return float(adj[route[:-1], route[1:]].sum())


class TestIntra(unittest.TestCase):
    def test_intra_swap_cost_decreases(self):
        # Geometric arcs on a line: positions 1..5 at coords 1,5,2,4,3.
        # adj = |coord_i - coord_j|. depot (0) at coord 0.
        coords = np.array([0, 1, 5, 2, 4, 3], dtype=np.float64)
        n = len(coords)
        adj = np.abs(coords[:, None] - coords[None, :])
        # A deliberately scrambled route; optimal order is sorted -> 1,2,3,4,5
        route = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int32)  # arcs in coord-order 1,5,2,4,3
        before = route_cost(route, adj)
        clss = np.ones(n, dtype=np.int64)
        clss[0] = 0
        delta, i, j = best_swap_intra(route, adj, clss, variant='U')
        self.assertLess(delta, 0.0)
        opt = intra_route_opt(route, adj, clss, variant='U')
        after = route_cost(opt, adj)
        self.assertLess(after, before - 1e-9)

    def test_intra_swap_arcs_unchanged(self):
        adj = make_symmetric_adj(8, seed=1)
        route = np.array([0, 3, 5, 1, 7, 2, 0], dtype=np.int32)
        clss = np.ones(8, dtype=np.int64)
        clss[0] = 0
        opt = intra_route_opt(route, adj, clss, variant='U')
        self.assertEqual(set(route.tolist()), set(opt.tolist()))
        self.assertEqual(len(route), len(opt))
        # endpoints remain depot
        self.assertEqual(opt[0], 0)
        self.assertEqual(opt[-1], 0)

    def test_intra_P_class_constraint(self):
        adj = make_symmetric_adj(7, seed=2)
        # arcs 1,2 class 1 ; arcs 3,4 class 2 ; arcs 5,6 class 3
        clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        route = np.array([0, 1, 3, 5, 2, 4, 6, 0], dtype=np.int32)
        # record class sequence; after P-optimisation, the multiset of classes
        # at each position must be unchanged (only same-class swaps allowed).
        before_classes = clss[route].copy()
        opt_route = route.copy()
        classes = [c for c in np.unique(clss) if c != 0]
        for p in classes:
            opt_route = intra_route_opt(opt_route, adj, clss, variant='P', p=p)
        after_classes = clss[opt_route]
        # same-class swaps keep the class at each position identical
        self.assertTrue(np.array_equal(before_classes, after_classes))
        self.assertEqual(set(route.tolist()), set(opt_route.tolist()))

    def test_intra_U_allows_cross_class(self):
        # build a case where the only improving swap is across classes
        coords = np.array([0, 1, 10, 2], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.array([0, 1, 2, 1], dtype=np.int64)  # arc1=cl1, arc2=cl2, arc3=cl1
        route = np.array([0, 2, 1, 3, 0], dtype=np.int32)  # coords 10,1,2 -> bad
        # P with per-class restriction: arc2(cl2) is alone in its class, arc1&arc3
        # (cl1) swap doesn't help here -> no cross-class move possible.
        dP, iP, jP = best_swap_intra(route, adj, clss, variant='P')
        # U: can move arc2 (the far one) -> improving cross-class swap exists
        dU, iU, jU = best_swap_intra(route, adj, clss, variant='U')
        self.assertLess(dU, 0.0)
        # the improving U move involves the class-2 arc (position of arc 2)
        pos_of_arc2 = int(np.where(route == 2)[0][0])
        self.assertIn(pos_of_arc2, (iU, jU))
        # and that move is cross-class
        self.assertNotEqual(clss[route[iU]], clss[route[jU]])


class TestInter(unittest.TestCase):
    def test_inter_swap_cost_decreases(self):
        coords = np.array([0, 1, 2, 8, 9], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.ones(5, dtype=np.int64)
        clss[0] = 0
        demands = np.array([0, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
        # bad assignment: route1 has near+far, route2 has near+far -> crossing
        r1 = np.array([0, 1, 8, 0], dtype=np.int32)  # arcs 1(coord1), 3(coord8)
        # remap: use arc indices: arc1=coord1, arc2=coord2, arc3=coord8, arc4=coord9
        r1 = np.array([0, 1, 3, 0], dtype=np.int32)
        r2 = np.array([0, 4, 2, 0], dtype=np.int32)
        before = route_cost(r1, adj) + route_cost(r2, adj)
        opt = inter_route_opt([r1, r2], adj, demands, clss, capacity=1.0, variant='U')
        after = sum(route_cost(r, adj) for r in opt)
        self.assertLess(after, before - 1e-9)

    def test_inter_capacity_not_violated(self):
        adj = make_symmetric_adj(6, seed=3)
        clss = np.ones(6, dtype=np.int64)
        clss[0] = 0
        # demands chosen so each route is near full; any swap that raises a
        # route's load above 1.0 must be rejected.
        demands = np.array([0, 0.6, 0.3, 0.5, 0.4, 0.2], dtype=np.float64)
        r1 = np.array([0, 1, 2, 0], dtype=np.int32)  # load 0.9
        r2 = np.array([0, 3, 4, 5, 0], dtype=np.int32)  # load 1.1 -> already over? no: 0.5+0.4+0.2=1.1
        # adjust so both start feasible
        demands = np.array([0, 0.6, 0.3, 0.5, 0.3, 0.1], dtype=np.float64)
        # r1 load 0.9, r2 load 0.9
        opt = inter_route_opt([r1.copy(), r2.copy()], adj, demands, clss,
                              capacity=1.0, variant='U')
        for r in opt:
            self.assertLessEqual(demands[r].sum(), 1.0 + 1e-9)

    def test_inter_arcs_conserved(self):
        adj = make_symmetric_adj(8, seed=4)
        clss = np.ones(8, dtype=np.int64)
        clss[0] = 0
        demands = np.full(8, 0.1, dtype=np.float64)
        demands[0] = 0
        r1 = np.array([0, 1, 2, 3, 0], dtype=np.int32)
        r2 = np.array([0, 4, 5, 6, 7, 0], dtype=np.int32)
        before = sorted([a for r in (r1, r2) for a in r if a != 0])
        opt = inter_route_opt([r1, r2], adj, demands, clss, capacity=1.0, variant='U')
        after = sorted([int(a) for r in opt for a in r if a != 0])
        self.assertEqual(before, after)


class TestEndToEnd(unittest.TestCase):
    def _make_instance(self, seed=5):
        # 6 arcs + depot
        n = 7
        adj = make_symmetric_adj(n, seed=seed)
        clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        demands = np.array([0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)
        vars = {'adj': adj, 'clss': clss, 'demand': demands,
                'service_time': np.zeros(n, dtype=np.float64)}
        return vars

    def _total_cost(self, routes, adj):
        return sum(route_cost(r, adj) for r in routes)

    def test_ls_end_to_end(self):
        vars = self._make_instance()
        adj = vars['adj']
        demands = vars['demand']
        # two routes, each <= 1.0 load: [1,3,5 | 2,4,6]
        action = np.array([1, 3, 5, 0, 2, 4, 6], dtype=np.int32)
        before_routes = gen_tours(action)
        before_cost = self._total_cost(before_routes, adj)

        out = ls(vars, variant='P', actions=[action])
        self.assertEqual(len(out), 1)
        routes = out[0]

        # validity: capacity
        for r in routes:
            self.assertLessEqual(demands[r].sum(), 1.0 + 1e-9)
        # all arcs present exactly once
        before_arcs = sorted([a for r in before_routes for a in r if a != 0])
        after_arcs = sorted([int(a) for r in routes for a in r if a != 0])
        self.assertEqual(before_arcs, after_arcs)
        # cost non-increased
        after_cost = self._total_cost(routes, adj)
        self.assertLessEqual(after_cost, before_cost + 1e-9)

    def test_ls_handles_none(self):
        vars = self._make_instance()
        out = ls(vars, variant='P', actions=[None])
        self.assertEqual(out, [[]])

    def test_lsRL_runs_and_conserves(self):
        vars = self._make_instance()
        td = {
            'adj': vars['adj'],
            'service_times': vars['service_time'],
            'clss': vars['clss'],
        }
        # tours as a padded (M, max_len) array
        tours = np.array([
            [0, 1, 2, 3, 0, 0],
            [0, 4, 5, 6, 0, 0],
        ], dtype=np.int32)
        before = sorted([int(a) for row in tours for a in row if a != 0])
        out = lsRL(td, tours)
        self.assertEqual(out.shape, tours.shape)
        after = sorted([int(a) for row in out for a in row if a != 0])
        self.assertEqual(before, after)


class TestNbUtils(unittest.TestCase):
    def test_gen_tours_roundtrip(self):
        action = np.array([1, 2, 0, 3, 4, 0, 5], dtype=np.int32)
        routes = gen_tours(action)
        # deserialize back with the original flat length
        flat = deserialize_tours(routes, len(action))
        self.assertTrue(np.array_equal(flat, action))

    def test_nb_utils_gen_tours(self):
        action = np.array([1, 2, 0, 3, 4, 0, 5], dtype=np.int32)
        routes = gen_tours(action)
        self.assertEqual(len(routes), 3)
        self.assertTrue(np.array_equal(routes[0], np.array([0, 1, 2, 0])))
        self.assertTrue(np.array_equal(routes[1], np.array([0, 3, 4, 0])))
        self.assertTrue(np.array_equal(routes[2], np.array([0, 5, 0])))

    def test_gen_tours_skips_empty_segments(self):
        action = np.array([1, 0, 0, 2], dtype=np.int32)
        routes = gen_tours(action)
        self.assertEqual(len(routes), 2)


class TestDeltaConsistency(unittest.TestCase):
    """O(1) deltas must equal the brute-force recomputed cost change."""

    def test_intra_delta_matches_bruteforce(self):
        adj = make_symmetric_adj(9, seed=7)
        route = np.array([0, 1, 4, 2, 7, 3, 0], dtype=np.int32)
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route) - 1):
                base = route_cost(route, adj)
                cand = route.copy()
                cand[i], cand[j] = cand[j], cand[i]
                brute = route_cost(cand, adj) - base
                fast = calc_swap_delta_intra(route, i, j, adj)
                self.assertAlmostEqual(brute, fast, places=9,
                                       msg=f"i={i} j={j}")

    def test_inter_delta_matches_bruteforce(self):
        adj = make_symmetric_adj(9, seed=8)
        r1 = np.array([0, 1, 2, 3, 0], dtype=np.int32)
        r2 = np.array([0, 4, 5, 6, 0], dtype=np.int32)
        for i in range(1, len(r1) - 1):
            for j in range(1, len(r2) - 1):
                base = route_cost(r1, adj) + route_cost(r2, adj)
                c1, c2 = r1.copy(), r2.copy()
                c1[i], c2[j] = c2[j], c1[i]
                brute = (route_cost(c1, adj) + route_cost(c2, adj)) - base
                fast = calc_swap_delta_inter(r1, i, r2, j, adj)
                self.assertAlmostEqual(brute, fast, places=9,
                                       msg=f"i={i} j={j}")


if __name__ == '__main__':
    unittest.main()
