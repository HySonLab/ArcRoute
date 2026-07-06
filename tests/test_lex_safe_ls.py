"""Tests for the move-level lex-safe U local search
(`utils.local_search.u_route_opt_lex_safe` + `_u_improving_moves`) and its
wiring into `RLHybHCARP._upgrade_winner_to_u`.

Background (measured on data/test_bm/small.npz, production RL-P config): the
deadhead-greedy U kernels run to convergence always bundle at least one
T1-worsening move, so a whole-pass lex guard rejected the result 100/100
times. The lex-safe search moves the acceptance check to the individual move:
a candidate is applied only if the full-solution (T1, T2, T3) does not
lex-worsen. These tests pin down:

  1. TestAdversarialT1Safety — a hand-built case where the BEST deadhead move
     relocates the class-1 arc later (T1-worsening): the plain greedy U
     kernel takes it (T1 regresses), the lex-safe search rejects it while
     still applying the smaller T1-safe deadhead improvement.
  2. TestCandidateDeltas     — `_u_improving_moves` deltas match brute-force
     route-cost recomputation for every candidate, and capacity-infeasible
     inter moves are never proposed.
  3. TestLexSafeProperties   — random instances: final tuple lex<= initial
     (so T1 never regresses), per-move monotonicity via max_iter prefixes,
     arc conservation, per-trip capacity feasibility, sentinels intact.
  4. TestUpgradeWiring       — `_upgrade_winner_to_u` (fake ckpt) returns a
     candidate whose tuple is lex<= the winner's own U-scored tuple, i.e.
     the pass can no longer hand the guard a T1-regressed bundle.

Run: uv run python -m unittest tests.test_lex_safe_ls -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.local_search import (
    _u_improving_moves,
    apply_relocate_intra,
    intra_route_opt,
    u_route_opt_lex_safe,
)
from solvers.evaluator import SolutionEvaluator


# --------------------------------------------------------------------- helpers
def make_td(adj, svc, clss):
    return {
        "adj": np.asarray(adj, dtype=np.float64),
        "service_times": np.asarray(svc, dtype=np.float64),
        "clss": np.asarray(clss, dtype=np.int64),
    }


def make_veh_score(td):
    ev = SolutionEvaluator(variant='U')
    ev._cache_td(td)
    return ev, (lambda veh_trips: ev._completion_times([veh_trips], td))


def score_routes(ev, td, routes, owner, M):
    """Full-solution (T1,T2,T3) of a sentinel-route list via the evaluator."""
    vehicles = [[] for _ in range(M)]
    for m, r in zip(owner, routes):
        interior = np.asarray(r)[1:-1]
        if len(interior):
            vehicles[m].append(interior)
    return ev._completion_times(vehicles, td)


def total_deadhead(routes, adj):
    return float(sum(adj[np.asarray(r)[:-1], np.asarray(r)[1:]].sum()
                     for r in routes))


def apply_candidate(routes, cand):
    """Apply one `_u_improving_moves` candidate to copies of `routes`."""
    delta, kind, ta, tb, i, j = cand
    routes = [np.asarray(r).copy() for r in routes]
    if kind == 'iswap':
        routes[ta][i], routes[ta][j] = routes[ta][j], routes[ta][i]
    elif kind == 'irel':
        routes[ta] = apply_relocate_intra(routes[ta], i, j)
    elif kind == 'xswap':
        routes[ta][i], routes[tb][j] = routes[tb][j], routes[ta][i]
    else:  # xrel
        x = routes[ta][i]
        routes[tb] = np.insert(routes[tb], j + 1, x)
        routes[ta] = np.delete(routes[ta], i)
    return routes


# ----------------------------------------------------------------------- 1
class TestAdversarialT1Safety(unittest.TestCase):
    """Hand-built single route [0, 1, 2, 3, 4, 5, 0], clss [_, 1, 2, 3, 3, 3]:

    - arc 5's tail edges are very expensive (adj[4,5] = adj[5,0] = 4) while
      opening with it is cheap (adj[0,5] = adj[5,1] = 1), so the single BEST
      deadhead move (delta -6) relocates arc 5 to the FRONT -- delaying the
      class-1 arc behind it and worsening T1: lex-UNSAFE.
    - swapping arcs 3 and 4 (both class 3, after the class-1 arc) is a
      smaller (delta -4) but T1/T2-neutral, T3-improving move: lex-SAFE.
    """

    def setUp(self):
        n = 6
        adj = np.full((n, n), 2.0)
        np.fill_diagonal(adj, 0.0)
        adj[0, 1] = 1.0
        # expensive tail around arc 5, cheap opening with it -> the best
        # deadhead move is pulling 5 to the front (T1-unsafe, delta -6)
        adj[4, 5] = 4.0
        adj[5, 0] = 4.0
        adj[4, 0] = 1.0
        adj[0, 5] = 1.0
        adj[5, 1] = 1.0
        # safe improvement: 2->4->3 beats 2->3->4 (delta -4)
        adj[2, 4] = 1.0
        adj[4, 3] = 1.0
        self.adj = adj
        self.svc = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.clss = np.array([0, 1, 2, 3, 3, 3])
        self.demands = np.array([0.0, 0.15, 0.15, 0.15, 0.15, 0.15])
        self.route = np.array([0, 1, 2, 3, 4, 5, 0])
        self.td = make_td(adj, self.svc, self.clss)

    def test_best_deadhead_move_is_the_unsafe_one(self):
        """Sanity of the adversarial construction: the top-ranked candidate
        pulls arc 5 in front of the class-1 arc and would worsen T1."""
        cands = _u_improving_moves([self.route], self.adj, self.demands,
                                   1.0, [self.demands[self.route].sum()])
        self.assertGreater(len(cands), 1)
        delta, kind, ta, tb, i, j = cands[0]
        self.assertEqual((kind, ta, i, j), ('irel', 0, 5, 0))  # 5 -> front
        self.assertAlmostEqual(delta, -6.0, places=9)
        ev, _ = make_veh_score(self.td)
        T0 = score_routes(ev, self.td, [self.route], [0], 1)
        T_bad = score_routes(ev, self.td, apply_candidate([self.route], cands[0]),
                             [0], 1)
        self.assertGreater(T_bad[0], T0[0])              # T1 regresses

    def test_greedy_kernel_takes_it_lex_safe_rejects_it(self):
        ev, veh_score = make_veh_score(self.td)
        T0 = score_routes(ev, self.td, [self.route], [0], 1)

        # plain deadhead-greedy U kernel: takes the unsafe move, T1 regresses
        greedy = intra_route_opt(self.route, self.adj, self.clss, variant='U')
        T_greedy = score_routes(ev, self.td, [greedy], [0], 1)
        self.assertGreater(T_greedy[0], T0[0])

        # lex-safe search: rejects it, applies the safe 3<->4 improvement
        out = u_route_opt_lex_safe([self.route], [0], 1, self.adj,
                                   self.demands, veh_score)
        T_safe = score_routes(ev, self.td, out, [0], 1)
        self.assertEqual(T_safe[0], T0[0])               # T1 untouched
        self.assertLess(tuple(T_safe.tolist()), tuple(T0.tolist()))
        self.assertEqual(int(out[0][1]), 1)              # class-1 arc stays first
        # the safe move really was taken: deadhead strictly decreased
        self.assertLess(total_deadhead(out, self.adj),
                        total_deadhead([self.route], self.adj) - 1e-9)


# ----------------------------------------------------------------------- 2
class TestCandidateDeltas(unittest.TestCase):
    def _rand_routes(self, seed):
        rng = np.random.default_rng(seed)
        n_arcs = 12
        adj = rng.uniform(0.1, 2.0, (n_arcs + 1, n_arcs + 1))
        np.fill_diagonal(adj, 0.0)
        adj[0, 0] = 0.0
        demands = np.concatenate([[0.0], rng.uniform(0.05, 0.3, n_arcs)])
        arcs = rng.permutation(np.arange(1, n_arcs + 1))
        cuts = sorted(rng.choice(np.arange(2, n_arcs - 1), 2, replace=False))
        parts = np.split(arcs, cuts)
        routes = [np.concatenate([[0], p, [0]]) for p in parts]
        return adj, demands, routes

    def test_deltas_match_bruteforce(self):
        for seed in range(5):
            adj, demands, routes = self._rand_routes(seed)
            loads = [demands[r].sum() for r in routes]
            cands = _u_improving_moves(routes, adj, demands, 1.0, loads)
            self.assertGreater(len(cands), 0)
            # sorted ascending
            deltas = [c[0] for c in cands]
            self.assertEqual(deltas, sorted(deltas))
            for cand in cands:
                new_routes = apply_candidate(routes, cand)
                brute = total_deadhead(new_routes, adj) - total_deadhead(routes, adj)
                self.assertAlmostEqual(brute, cand[0], places=9,
                                       msg=f"seed={seed} cand={cand}")
                self.assertLess(cand[0], 0.0)

    def test_capacity_infeasible_moves_not_proposed(self):
        adj, demands, routes = self._rand_routes(0)
        # tight capacity: each route is exactly at its own load -> any inter
        # move that increases either route's load must be filtered out.
        loads = [demands[r].sum() for r in routes]
        cap = max(loads)
        cands = _u_improving_moves(routes, adj, demands, cap, loads)
        for cand in cands:
            new_routes = apply_candidate(routes, cand)
            for r in new_routes:
                self.assertLessEqual(demands[np.asarray(r)].sum(), cap + 1e-9)


# ----------------------------------------------------------------------- 3
class TestLexSafeProperties(unittest.TestCase):
    def _rand_instance(self, seed):
        rng = np.random.default_rng(seed)
        n_arcs = 14
        M = 2
        adj = rng.uniform(0.1, 2.0, (n_arcs + 1, n_arcs + 1))
        np.fill_diagonal(adj, 0.0)
        svc = np.concatenate([[0.0], rng.uniform(0.05, 0.3, n_arcs)])
        clss = np.concatenate([[0], rng.integers(1, 4, n_arcs)])
        demands = np.concatenate([[0.0], rng.uniform(0.05, 0.35, n_arcs)])
        arcs = rng.permutation(np.arange(1, n_arcs + 1))
        # 4 trips, owners [0, 0, 1, 1]
        parts = np.array_split(arcs, 4)
        routes = [np.concatenate([[0], p, [0]]) for p in parts]
        owner = [0, 0, 1, 1]
        # capacity: make every trip feasible but not slack-free
        cap = max(demands[r].sum() for r in routes) * 1.15
        td = make_td(adj, svc, clss)
        return adj, demands, routes, owner, M, cap, td

    def test_lex_never_worsens_and_feasible(self):
        for seed in range(8):
            adj, demands, routes, owner, M, cap, td = self._rand_instance(seed)
            ev, veh_score = make_veh_score(td)
            T0 = score_routes(ev, td, routes, owner, M)
            out = u_route_opt_lex_safe(routes, owner, M, adj, demands,
                                       veh_score, capacity=cap)
            T1_ = score_routes(ev, td, out, owner, M)
            # lex non-worsening => T1 in particular never regresses
            self.assertLessEqual(tuple(T1_.tolist()), tuple(T0.tolist()),
                                 msg=f"seed={seed}")
            self.assertLessEqual(T1_[0], T0[0] + 0.0)
            # arcs conserved (single service preserved)
            before = sorted(int(a) for r in routes for a in np.asarray(r)[1:-1])
            after = sorted(int(a) for r in out for a in np.asarray(r)[1:-1])
            self.assertEqual(before, after)
            # per-trip capacity + sentinels
            for r in out:
                r = np.asarray(r)
                self.assertEqual(r[0], 0)
                self.assertEqual(r[-1], 0)
                self.assertTrue((r[1:-1] != 0).all())
                self.assertLessEqual(demands[r].sum(), cap + 1e-9)

    def test_per_move_monotonicity_via_max_iter_prefixes(self):
        """max_iter=k replays the same deterministic accepted-move sequence
        truncated at k, so the tuple after each k must be lex non-increasing:
        the acceptance criterion holds at EVERY move, not just the bundle."""
        for seed in range(4):
            adj, demands, routes, owner, M, cap, td = self._rand_instance(seed)
            ev, veh_score = make_veh_score(td)
            prev = tuple(score_routes(ev, td, routes, owner, M).tolist())
            for k in range(1, 9):
                out = u_route_opt_lex_safe(routes, owner, M, adj, demands,
                                           veh_score, capacity=cap, max_iter=k)
                cur = tuple(score_routes(ev, td, out, owner, M).tolist())
                self.assertLessEqual(cur, prev,
                                     msg=f"seed={seed} k={k}")
                prev = cur

    def test_deadhead_strictly_decreases_when_it_moves(self):
        moved_any = False
        for seed in range(8):
            adj, demands, routes, owner, M, cap, td = self._rand_instance(seed)
            _, veh_score = make_veh_score(td)
            out = u_route_opt_lex_safe(routes, owner, M, adj, demands,
                                       veh_score, capacity=cap)
            changed = len(out) != len(routes) or any(
                not np.array_equal(a, b) for a, b in zip(routes, out))
            if changed:
                moved_any = True
                self.assertLess(total_deadhead(out, adj),
                                total_deadhead(routes, adj) - 1e-12)
        self.assertTrue(moved_any, "no random instance produced any safe move")

    def test_degenerate_inputs(self):
        adj = np.zeros((2, 2))
        demands = np.zeros(2)
        _, veh_score = make_veh_score(make_td(adj, np.zeros(2), np.zeros(2, int)))
        # empty route list
        self.assertEqual(u_route_opt_lex_safe([], [], 1, adj, demands,
                                              veh_score), [])
        # single empty trip [0, 0] and single-arc trip: nothing movable
        out = u_route_opt_lex_safe([np.array([0, 0]), np.array([0, 1, 0])],
                                   [0, 0], 1, adj, demands, veh_score)
        self.assertTrue(np.array_equal(out[0], [0, 0]))
        self.assertTrue(np.array_equal(out[1], [0, 1, 0]))


# ----------------------------------------------------------------------- 4
class TestUpgradeWiring(unittest.TestCase):
    """End-to-end through `RLHybHCARP._upgrade_winner_to_u` with the fake
    checkpoint: the upgrade candidate must be lex<= the winner's own U-scored
    tuple (per-move safety implies the whole pass can only help or tie), so
    the caller's guard can never see a T1-regressed candidate."""

    def test_upgrade_candidate_lex_safe_vs_winner(self):
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from test_rl import _make_hyb
        al = _make_hyb(shortlist=4, eps=0.01, upgrade_to_u=True)

        torch.manual_seed(123)
        out_on = al(num_sample=16)[0]
        info = al.last_upgrade_info
        self.assertIsNotNone(info)
        # per-move safety => post is lex<= pre, and T1 never regresses
        self.assertLessEqual(tuple(info["post_T"].tolist()),
                             tuple(info["pre_T"].tolist()))
        self.assertLessEqual(info["post_T"][0], info["pre_T"][0])
        self.assertTrue(np.array_equal(out_on, info["post_T"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
