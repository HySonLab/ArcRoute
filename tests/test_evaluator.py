"""SolutionEvaluator variant-U decode path (_segment + _assign).

Until the evaluate_batch variant fix, the metaheuristics never exercised this
path (everything was silently scored through the P route-builder), so it had
zero direct tests. Covers:
  - hand-computed T values for both variants on the same instance/action,
  - U decode on a class-interleaved action: capacity-feasible, fleet-sized,
    every arc served exactly once, finite T,
  - regression: evaluate_batch(variant=...) actually reaches SolutionEvaluator,
  - regression: capacity reloads never spawn phantom vehicles (#trips > M),
  - U <= P sanity: U's scheduling freedom never worsens the best achievable
    objective (brute force over all arc orders of a tiny instance).

Run: uv run python -m unittest tests.test_evaluator -v
"""
import os
import sys
import unittest
from itertools import permutations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from solvers.evaluator import SolutionEvaluator, evaluate_batch


# --------------------------------------------------------------------------- #
# hand-built instances (no .npz, no torch needed)
# --------------------------------------------------------------------------- #
def tiny_td():
    """Depot + 2 arcs; adj = 1 off-diagonal; service = [0, 2, 3].

    arc 1 -> class 2, arc 2 -> class 1; demands 0.3 each (one trip for M=1).
    For action [1, 2] with M=1 every completion time is hand-computable:

      U keeps the given (interleaved) order [1, 2]:
        path 0->1->2->0 ; t = [0, 2+1, 3+1, 0+1] ; cum = [0, 3, 7, 8]
        arc1 (class 2) done @3, arc2 (class 1) done @7  =>  T = (7, 3, 0)
      P re-sorts by class to [2, 1]:
        path 0->2->1->0 ; t = [0, 3+1, 2+1, 0+1] ; cum = [0, 4, 7, 8]
        arc2 (class 1) done @4, arc1 (class 2) done @7  =>  T = (4, 7, 0)
    """
    n = 3
    adj = np.ones((n, n), dtype=np.float64) - np.eye(n)
    return {
        "adj": adj,
        "service_times": np.array([0.0, 2.0, 3.0]),
        "clss": np.array([0, 2, 1], dtype=np.int64),
        "demand": np.array([0.0, 0.3, 0.3]),
    }


def interleaved_td(n_arcs=6, seed=21):
    """Depot + n_arcs across 3 classes with a random (asymmetric) adj."""
    n = n_arcs + 1
    rng = np.random.default_rng(seed)
    adj = rng.random((n, n))
    np.fill_diagonal(adj, 0.0)
    svc = np.concatenate([[0.0], rng.random(n_arcs) + 0.5])
    clss = np.array([0] + [1 + (i % 3) for i in range(n_arcs)], dtype=np.int64)
    demand = np.concatenate([[0.0], np.full(n_arcs, 0.3)])
    return {"adj": adj, "service_times": svc, "clss": clss, "demand": demand}


# --------------------------------------------------------------------------- #
# hand-computed correctness of the U decode
# --------------------------------------------------------------------------- #
class TestUDecodeHandComputed(unittest.TestCase):
    def test_u_segment_arm_exact_T(self):
        """The order-preserving segment arm keeps the interleaved order
        [1, 2] and hand-computes to (7, 3, 0)."""
        td = tiny_td()
        ev = SolutionEvaluator(variant="U")
        ev._cache_td(td)
        vehicles = ev._decode_U_segment([1, 2], td, M=1)
        self.assertEqual(len(vehicles), 1)
        self.assertEqual([int(a) for a in vehicles[0][0]], [1, 2])
        np.testing.assert_allclose(ev._completion_times(vehicles, td),
                                   [7.0, 3.0, 0.0])

    def test_u_exact_T_takes_lex_better_decode(self):
        """The full U decode returns the lex-better of the segment arm
        (7, 3, 0) and the P-feasible class-sorted arm (4, 7, 0) — here the
        class-sorted schedule wins on T1. (U's feasible set is a superset of
        P's, so U must never score an action worse than the P decoder.)"""
        td = tiny_td()
        vehicles, T = SolutionEvaluator(variant="U")([1, 2], td, M=1)
        self.assertEqual(len(vehicles), 1)
        np.testing.assert_allclose(T, [4.0, 7.0, 0.0])

    def test_p_exact_T(self):
        td = tiny_td()
        vehicles, T = SolutionEvaluator(variant="P")([1, 2], td, M=1)
        self.assertEqual(len(vehicles), 1)
        np.testing.assert_allclose(T, [4.0, 7.0, 0.0])


class TestUDecodeInterleaved(unittest.TestCase):
    def test_interleaved_action_feasible_and_finite(self):
        td = interleaved_td()
        # cross-class interleaving, with 0-separators like metaheuristic actions
        action = np.array([1, 4, 2, 0, 5, 3, 6], dtype=np.int64)
        M = 2
        vehicles, T = SolutionEvaluator(variant="U")(action, td, M=M)
        # fleet size respected
        self.assertEqual(len(vehicles), M)
        # every trip capacity-feasible
        for veh in vehicles:
            for trip in veh:
                self.assertLessEqual(td["demand"][np.asarray(trip)].sum(),
                                     1.0 + 1e-9)
        # every arc served exactly once
        served = sorted(int(a) for veh in vehicles for trip in veh for a in trip)
        self.assertEqual(served, [1, 2, 3, 4, 5, 6])
        # T finite, non-negative, one entry per class
        self.assertEqual(T.shape, (3,))
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertTrue(np.all(T >= 0))

    def test_over_capacity_single_vehicle_is_infeasible(self):
        """Reload-bug fix (was test_no_phantom_vehicles_when_trips_exceed_M):
        total demand 2.4 with ONE vehicle needs >1 trip. The old code served it
        by chaining reload trips onto one vehicle's clock (finite T assuming an
        illegal multi-trip). The fixed evaluator must score it INFEASIBLE
        (+inf) — a single vehicle cannot serve more than its capacity Q."""
        n = 5
        adj = np.ones((n, n)) - np.eye(n)
        td = {
            "adj": adj,
            "service_times": np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
            "clss": np.array([0, 1, 1, 1, 1], dtype=np.int64),
            "demand": np.array([0.0, 0.6, 0.6, 0.6, 0.6]),  # total 2.4 > Q
        }
        _, T_seq = SolutionEvaluator(variant="U")([1, 2, 3, 4], td, M=1)
        self.assertFalse(np.all(np.isfinite(T_seq)),
                         "single vehicle serving >Q should be infeasible (reload)")


# --------------------------------------------------------------------------- #
# regression for the Item-1 bug: variant kwarg must reach SolutionEvaluator
# --------------------------------------------------------------------------- #
class TestEvaluateBatchVariant(unittest.TestCase):
    def _vars(self):
        td = tiny_td()
        return {
            "adj": td["adj"],
            "service_time": td["service_times"],
            "clss": td["clss"],
            "demand": td["demand"],
            "nv": 1,
        }

    def test_variant_reaches_evaluator(self):
        """Regression: before the fix evaluate_batch silently defaulted to
        'P' for both variants, so U and P scored identically everywhere.
        Uses an instance where the two variants MUST produce different
        T-vectors. With the reload fix, the class-1 arc (demand 0.6) plus the
        two class-2 arcs (0.3 each) sum to 1.2 > Q on the single unpartitioned
        route [1,2,3], so the P decode (which honours the action's own —
        here absent — vehicle boundaries: one route) is INFEASIBLE, while the
        U segment arm splits it into two feasible single-trip vehicles."""
        td = split_win_td()
        vars = {
            "adj": td["adj"],
            "service_time": td["service_times"],
            "clss": td["clss"],
            "demand": td["demand"],
            "nv": 2,
        }
        actions = np.array([[1, 2, 3]], dtype=np.int64)
        TU = evaluate_batch(vars, actions, variant="U")
        TP = evaluate_batch(vars, actions, variant="P")
        np.testing.assert_allclose(TU, [[2.0, 4.0, 0.0]])
        self.assertFalse(np.all(np.isfinite(TP)))          # single route > Q
        self.assertFalse(np.allclose(TU, TP))

    def test_default_is_P(self):
        vars = self._vars()
        actions = np.array([[1, 2]], dtype=np.int64)
        np.testing.assert_allclose(evaluate_batch(vars, actions),
                                   evaluate_batch(vars, actions, variant="P"))


# --------------------------------------------------------------------------- #
# U dual decode: lex-better of segment arm and P-feasible class-sorted arm
# (regression for the ILS-U benchmark T1 blow-up: the segment+LPT decode
# alone could score an action WORSE than the P decoder — e.g. stacking all
# class-1 arcs on one vehicle where _build_P spreads them — violating the
# "U never worse than P" property the paper relies on)
# --------------------------------------------------------------------------- #
def split_win_td():
    """Instance where the segment decode strictly lex-beats the P decode.

    arc 1: class 1, demand 0.6; arcs 2, 3: class 2, demand 0.3, adjacent to
    each other but far from arc 1. With M=2 and action [1, 2, 3]:

      segment: segs [1] | [2, 3]  ->  veh0=[1], veh1=[2, 3]
        arc1 @ 1+1=2 ; arc2 @ 1+1=2 ; arc3 @ 2+1+1=4  =>  T = (2, 4, 0)
      _build_P: class-2 chunks [2] -> veh1, [3] -> veh0  ->  veh0=[1, 3]
        arc3 @ 2 + d(1,3)=9 + 1 = 12                    =>  T = (2, 12, 0)
    """
    n = 4
    adj = np.full((n, n), 5.0)
    np.fill_diagonal(adj, 0.0)
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    adj[2, 3] = 1.0
    adj[1, 3] = 9.0
    return {
        "adj": adj,
        "service_times": np.array([0.0, 1.0, 1.0, 1.0]),
        "clss": np.array([0, 1, 2, 2], dtype=np.int64),
        "demand": np.array([0.0, 0.6, 0.3, 0.3]),
    }


class TestUDualDecode(unittest.TestCase):
    def test_segment_arm_wins_when_it_is_lex_better(self):
        td = split_win_td()
        vehicles, T = SolutionEvaluator(variant="U")([1, 2, 3], td, M=2)
        np.testing.assert_allclose(T, [2.0, 4.0, 0.0])
        # the winning schedule is the segment one (order-preserving split)
        served = sorted(int(a) for veh in vehicles for t in veh for a in t)
        self.assertEqual(served, [1, 2, 3])

    def test_u_pointwise_never_worse_than_p(self):
        """For EVERY action on random instances, the U decode's T-vector is
        lex <= the P decode's (the P-feasible arm guarantees it)."""
        rng = np.random.default_rng(7)
        for trial in range(20):
            n_arcs = int(rng.integers(3, 7))
            n = n_arcs + 1
            adj = rng.random((n, n)) * 3.0
            np.fill_diagonal(adj, 0.0)
            td = {
                "adj": adj,
                "service_times": np.concatenate([[0.0],
                                                 rng.random(n_arcs) + 0.5]),
                "clss": np.concatenate(
                    [[0], rng.integers(1, 4, n_arcs)]).astype(np.int64),
                "demand": np.concatenate([[0.0],
                                          rng.uniform(0.1, 0.5, n_arcs)]),
            }
            M = int(rng.integers(1, 4))
            action = rng.permutation(np.arange(1, n)).tolist()
            TU = SolutionEvaluator(variant="U")(action, td, M=M)[1]
            TP = SolutionEvaluator(variant="P")(action, td, M=M)[1]
            self.assertLessEqual(tuple(np.round(TU, 9)),
                                 tuple(np.round(TP, 9)),
                                 msg=f"trial {trial}: U {TU} lex-worse than P {TP}")


# --------------------------------------------------------------------------- #
# U <= P sanity (mirrors tests/test_lp.py::test_milp_u_leq_milp_p)
# --------------------------------------------------------------------------- #
class TestULeqP(unittest.TestCase):
    def test_best_u_never_worse_than_best_p(self):
        """Brute force every arc order of a 3-arc instance (classes NOT sorted
        in most of them), decode each appropriately per variant, and compare
        the best achievable values: U's extra scheduling freedom must never
        make the objective worse."""
        rng = np.random.default_rng(33)
        n = 4
        adj = rng.random((n, n)) * 2.0
        np.fill_diagonal(adj, 0.0)
        td = {
            "adj": adj,
            "service_times": np.concatenate([[0.0], rng.random(3) + 0.5]),
            "clss": np.array([0, 2, 1, 2], dtype=np.int64),
            "demand": np.array([0.0, 0.2, 0.2, 0.2]),   # single trip
        }
        evU = SolutionEvaluator(variant="U")
        evP = SolutionEvaluator(variant="P")
        TU = np.array([evU(list(a), td, M=1)[1] for a in permutations([1, 2, 3])])
        TP = np.array([evP(list(a), td, M=1)[1] for a in permutations([1, 2, 3])])

        # primary objective
        self.assertLessEqual(TU[:, 0].min(), TP[:, 0].min() + 1e-9)
        # weighted-lexicographic objective (same spirit as the MILP test)
        w = np.array([1e6, 1e3, 1.0])
        self.assertLessEqual((TU @ w).min(), (TP @ w).min() + 1e-6)


# --------------------------------------------------------------------------- #
# reload-bug fix: whole-route single budget, no reload trips
# --------------------------------------------------------------------------- #
class TestReloadScoring(unittest.TestCase):
    def _td(self):
        # depot + 2 class-1 arcs, demand 0.6 each. adj=1 off-diagonal.
        n = 3
        adj = np.ones((n, n)) - np.eye(n)
        return {
            "adj": adj,
            "service_times": np.array([0.0, 1.0, 1.0]),
            "clss": np.array([0, 1, 1], dtype=np.int64),
            "demand": np.array([0.0, 0.6, 0.6]),
        }

    def test_old_build_P_reload_chunk_now_infeasible(self):
        """0.6 + 0.6 = 1.2 > Q on ONE vehicle. The OLD _build_P fed the route
        through _split_by_capacity, which inserted a depot reload and returned
        a finite (illegally multi-trip) T. The fix scores it +inf."""
        td = self._td()
        _, T = SolutionEvaluator(variant="P")([1, 2], td, M=1)   # no separators
        self.assertFalse(np.all(np.isfinite(T)))

    def test_partition_respected_maps_trips_to_distinct_vehicles(self):
        """Same two arcs, but the action's own 0-separator splits them across
        two vehicles: feasible, single-trip each, scored as PARALLEL routes
        (each arc completes at 0->arc = svc+adj = 2, not chained)."""
        td = self._td()
        vehicles, T = SolutionEvaluator(variant="P")([1, 0, 2], td, M=2)
        # exactly one trip per used vehicle, both arcs served once
        n_trips = sum(len(v) for v in vehicles)
        self.assertEqual(n_trips, 2)
        served = sorted(int(a) for v in vehicles for t in v for a in t)
        self.assertEqual(served, [1, 2])
        # parallel single trips: T1 = max over the two vehicles' class-1 arc
        # completion = 0->arc->(depot) : svc(1)+adj(0,arc)=1+1 = 2 (NOT 4, which
        # a reload/chained decode would give).
        np.testing.assert_allclose(T, [2.0, 0.0, 0.0])

    def test_more_trips_than_vehicles_infeasible(self):
        """A 2-vehicle instance whose action needs 3 single-trip routes
        (three 0-separated segments) can't be served by 2 vehicles -> +inf."""
        n = 4
        adj = np.ones((n, n)) - np.eye(n)
        td = {
            "adj": adj,
            "service_times": np.array([0.0, 1.0, 1.0, 1.0]),
            "clss": np.array([0, 1, 1, 1], dtype=np.int64),
            "demand": np.array([0.0, 0.9, 0.9, 0.9]),  # each needs its own trip
        }
        _, T = SolutionEvaluator(variant="P")([1, 0, 2, 0, 3], td, M=2)
        self.assertFalse(np.all(np.isfinite(T)))


if __name__ == "__main__":
    unittest.main()
