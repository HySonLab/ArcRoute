import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.local_search import (
    calc_swap_delta_intra,
    calc_swap_delta_inter,
    calc_relocate_delta_intra,
    calc_relocate_delta_inter,
    apply_relocate_intra,
    best_swap_intra,
    best_relocate_intra,
    intra_route_opt,
    best_relocate_inter,
    inter_route_opt,
    ls,
)
from utils.ops import gen_tours, deserialize_tours


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


class TestRelocateDeltaConsistency(unittest.TestCase):
    """Or-opt-1 O(1) deltas must equal the brute-force recomputed cost change,
    for every (i, j) pair including the adjacent/no-op edge cases."""

    def test_intra_relocate_delta_matches_bruteforce(self):
        rng = np.random.default_rng(12)
        adj_asym = rng.random((9, 9))
        np.fill_diagonal(adj_asym, 0.0)
        for A in (make_symmetric_adj(9, seed=11), adj_asym):
            route = np.array([0, 1, 4, 2, 7, 3, 0], dtype=np.int32)
            base = route_cost(route, A)
            for i in range(1, len(route) - 1):
                for j in range(0, len(route) - 1):
                    cand = apply_relocate_intra(route, i, j)
                    brute = route_cost(cand, A) - base
                    fast = calc_relocate_delta_intra(route, i, j, A)
                    self.assertAlmostEqual(brute, fast, places=9,
                                           msg=f"i={i} j={j}")

    def test_inter_relocate_delta_matches_bruteforce(self):
        adj = make_symmetric_adj(9, seed=13)
        r1 = np.array([0, 1, 2, 3, 0], dtype=np.int32)
        r2 = np.array([0, 4, 5, 6, 0], dtype=np.int32)
        base = route_cost(r1, adj) + route_cost(r2, adj)
        for i in range(1, len(r1) - 1):
            for j in range(0, len(r2) - 1):
                c1 = np.delete(r1, i)
                c2 = np.insert(r2, j + 1, r1[i])
                brute = (route_cost(c1, adj) + route_cost(c2, adj)) - base
                fast = calc_relocate_delta_inter(r1, i, r2, j, adj)
                self.assertAlmostEqual(brute, fast, places=9,
                                       msg=f"i={i} j={j}")


class TestRelocateIntra(unittest.TestCase):
    def test_relocate_improves_and_delta_is_exact(self):
        # coords of arcs 1..5 are 1,5,2,4,3; identity route is scrambled in
        # space; relocating arc 2 (coord 5) after arc 4 (coord 4) sorts it.
        coords = np.array([0, 1, 5, 2, 4, 3], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.ones(len(coords), dtype=np.int64)
        clss[0] = 0
        route = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int32)
        before = route_cost(route, adj)
        delta, i, j = best_relocate_intra(route, adj, clss, variant='U')
        self.assertLess(delta, 0.0)
        after = route_cost(apply_relocate_intra(route, i, j), adj)
        self.assertAlmostEqual(after - before, delta, places=9)

    def test_relocate_P_singleton_classes_blocked_U_allowed(self):
        # every arc its own class -> P has no legal relocate at all, while U
        # still finds an improving (necessarily cross-class) one.
        coords = np.array([0, 1, 10, 2], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.array([0, 1, 2, 3], dtype=np.int64)
        route = np.array([0, 2, 1, 3, 0], dtype=np.int32)
        dP, iP, jP = best_relocate_intra(route, adj, clss, variant='P')
        self.assertEqual((dP, iP, jP), (0.0, -1, -1))
        dU, iU, jU = best_relocate_intra(route, adj, clss, variant='U')
        self.assertLess(dU, 0.0)
        # and the move really is cross-class relative to its new predecessor
        self.assertNotEqual(clss[route[iU]], clss[route[jU]])

    def test_relocate_P_preserves_class_sequence(self):
        # interleaved classes: any P relocate/swap combination applied by
        # intra_route_opt must keep the class sequence identical.
        adj = make_symmetric_adj(7, seed=14)
        clss = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        route = np.array([0, 1, 3, 5, 2, 4, 6, 0], dtype=np.int32)
        before_classes = clss[route].copy()
        opt = route.copy()
        for p in (1, 2, 3):
            opt = intra_route_opt(opt, adj, clss, variant='P', p=p)
        self.assertTrue(np.array_equal(before_classes, clss[opt]))
        self.assertEqual(set(route.tolist()), set(opt.tolist()))

    def test_relocate_P_reorders_within_class_block(self):
        # class-sorted route; the class-1 block [3,1,2] (coords) untangles via
        # relocate while staying inside its block.
        coords = np.array([0, 3, 1, 2, 30, 31], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
        route = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int32)
        before_classes = clss[route].copy()
        before = route_cost(route, adj)
        opt = route.copy()
        for p in (1, 2):
            opt = intra_route_opt(opt, adj, clss, variant='P', p=p)
        self.assertTrue(np.array_equal(before_classes, clss[opt]))
        self.assertLessEqual(route_cost(opt, adj), before + 1e-9)

    def test_intra_route_opt_converges_with_both_moves(self):
        adj = make_symmetric_adj(8, seed=15)
        clss = np.ones(8, dtype=np.int64)
        clss[0] = 0
        route = np.array([0, 3, 5, 1, 7, 2, 0], dtype=np.int32)
        opt = intra_route_opt(route, adj, clss, variant='U')
        # arcs conserved, endpoints depot, no further improving move of either kind
        self.assertEqual(set(route.tolist()), set(opt.tolist()))
        self.assertEqual(opt[0], 0)
        self.assertEqual(opt[-1], 0)
        self.assertEqual(best_swap_intra(opt, adj, clss, variant='U')[1], -1)
        self.assertEqual(best_relocate_intra(opt, adj, clss, variant='U')[1], -1)


class TestRelocateInter(unittest.TestCase):
    def test_relocate_moves_far_arc_to_other_route(self):
        # arc 2 (coord 9) sits in r1 next to coord 1; r2 serves coord 8.
        # moving arc 2 into r2 after arc 3 is a big win.
        coords = np.array([0, 1, 9, 8], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.ones(4, dtype=np.int64)
        clss[0] = 0
        demands = np.array([0, 0.3, 0.3, 0.3], dtype=np.float64)
        r1 = np.array([0, 1, 2, 0], dtype=np.int32)
        r2 = np.array([0, 3, 0], dtype=np.int32)
        before = route_cost(r1, adj) + route_cost(r2, adj)
        d, i, j = best_relocate_inter(r1, r2, adj, demands, 1.0, clss,
                                      variant='U')
        self.assertLess(d, 0.0)
        c1 = np.delete(r1, i)
        c2 = np.insert(r2, j + 1, r1[i])
        after = route_cost(c1, adj) + route_cost(c2, adj)
        self.assertAlmostEqual(after - before, d, places=9)

    def test_relocate_inter_respects_capacity(self):
        adj = make_symmetric_adj(6, seed=16)
        clss = np.ones(6, dtype=np.int64)
        clss[0] = 0
        # receiving route load 0.9; every arc demands 0.3 -> nothing may move in.
        demands = np.array([0, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)
        r1 = np.array([0, 1, 2, 0], dtype=np.int32)
        r2 = np.array([0, 3, 4, 5, 0], dtype=np.int32)
        out = best_relocate_inter(r1, r2, adj, demands, 1.0, clss, variant='U')
        self.assertEqual(out, (0.0, -1, -1))

    def test_relocate_inter_P_needs_same_class_anchor(self):
        # r2 contains no class-2 arc -> a class-2 arc cannot move in under P,
        # but can under U.
        coords = np.array([0, 1, 9, 8], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.array([0, 1, 2, 1], dtype=np.int64)
        demands = np.array([0, 0.3, 0.3, 0.3], dtype=np.float64)
        r1 = np.array([0, 1, 2, 0], dtype=np.int32)
        r2 = np.array([0, 3, 0], dtype=np.int32)
        outP = best_relocate_inter(r1, r2, adj, demands, 1.0, clss, variant='P')
        self.assertEqual(outP, (0.0, -1, -1))
        dU, iU, _ = best_relocate_inter(r1, r2, adj, demands, 1.0, clss,
                                        variant='U')
        self.assertLess(dU, 0.0)
        self.assertEqual(int(r1[iU]), 2)

    def test_inter_route_opt_with_relocate_conserves_and_improves(self):
        coords = np.array([0, 1, 9, 8, 2], dtype=np.float64)
        adj = np.abs(coords[:, None] - coords[None, :])
        clss = np.ones(5, dtype=np.int64)
        clss[0] = 0
        demands = np.full(5, 0.2, dtype=np.float64)
        demands[0] = 0.0
        r1 = np.array([0, 1, 2, 0], dtype=np.int32)   # coords 1, 9
        r2 = np.array([0, 3, 4, 0], dtype=np.int32)   # coords 8, 2
        before_arcs = sorted(a for r in (r1, r2) for a in r.tolist() if a != 0)
        before = route_cost(r1, adj) + route_cost(r2, adj)
        out = inter_route_opt([r1, r2], adj, demands, clss,
                              capacity=1.0, variant='U')
        after_arcs = sorted(int(a) for r in out for a in r if a != 0)
        self.assertEqual(before_arcs, after_arcs)
        for r in out:
            self.assertLessEqual(demands[np.asarray(r)].sum(), 1.0 + 1e-9)
            self.assertEqual(int(r[0]), 0)
            self.assertEqual(int(r[-1]), 0)
        self.assertLess(sum(route_cost(r, adj) for r in out), before - 1e-9)


if __name__ == '__main__':
    unittest.main()
