"""Regression tripwires for the reload-bug fix (cumulative capacity).

Root cause (see docs/root-cause-allow.md): the OLD CARPEnv reset a vehicle's
``used_capacity`` to 0 on any depot visit, letting one vehicle serve more than
its capacity Q via mid-route "reloads" — a multi-trip capability the HDCARP
problem definition, the paper's MILP, and every classical baseline forbid.

These tests pin the FIXED semantics and FAIL by construction against the old
env:
  * a single vehicle can never serve total demand > Q (it must dead-end);
  * per vehicle, cumulative served demand never exceeds Q at any trajectory
    step (the mask enforces it);
  * a dead-ended rollout is penalised proportionally to its unserved demand.

Run: uv run python -m unittest tests.test_reload_fix -v
"""
import os
import sys
import unittest

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from env.env import CARPEnv


def _make_td(demands, clss, svc, adj, M):
    demands = np.asarray(demands, np.float64)
    return TensorDict({
        "demands": torch.tensor(demands[None], dtype=torch.float32),
        "clss": torch.tensor(np.asarray(clss)[None], dtype=torch.int64),
        "service_times": torch.tensor(np.asarray(svc, np.float64)[None], dtype=torch.float32),
        "traversal_times": torch.tensor(np.asarray(svc, np.float64)[None], dtype=torch.float32),
        "adj": torch.tensor(np.asarray(adj, np.float64)[None], dtype=torch.float32),
        "num_vehicle": torch.tensor([[int(M)]], dtype=torch.int64),
    }, batch_size=[1])


def _drive(env, td, rng, max_steps=5000):
    """Random feasible-action rollout (no policy). Returns
    (steps, td, max_capacity_seen, looped)."""
    max_cap = 0.0
    for t in range(max_steps):
        if bool(td["done"].all()):
            return t, td, max_cap, False
        feas = torch.nonzero(td["action_mask"][0]).reshape(-1).tolist()
        assert feas, f"step {t}: mask is all-infeasible (never allowed)"
        a = int(rng.choice(feas))
        td.set("action", torch.tensor([a], dtype=torch.int64))
        td = env.step(td)
        max_cap = max(max_cap, float(td["used_capacity"].max()))
    return max_steps, td, max_cap, True


class TestReloadForbidden(unittest.TestCase):
    def test_single_vehicle_over_capacity_deadends(self):
        """Total demand 1.5Q with ONE vehicle. The OLD env served all 3 arcs by
        reloading; the FIXED env must strand >=1 arc (dead-end)."""
        adj = np.ones((4, 4)) - np.eye(4)
        td = _make_td([0, .5, .5, .5], [0, 1, 1, 1], [0, 1, 1, 1], adj, M=1)
        env = CARPEnv(num_vehicle=1, variant='P', reward_mode='vector')
        td = env.reset(td)
        steps, td, max_cap, looped = _drive(env, td, np.random.default_rng(0))
        served = int((td["visited"][0, 0, 1:] != 0).sum())
        self.assertFalse(looped, "rollout did not terminate")
        self.assertLess(served, 3, "single vehicle served >Q worth of arcs (reload!)")
        self.assertLessEqual(max_cap, 1.0 + 1e-9, "cumulative capacity exceeded Q")

    def test_two_vehicles_serve_same_demand(self):
        """The same 1.5Q of demand IS servable with a second single-trip vehicle."""
        adj = np.ones((4, 4)) - np.eye(4)
        td = _make_td([0, .5, .5, .5], [0, 1, 1, 1], [0, 1, 1, 1], adj, M=2)
        env = CARPEnv(num_vehicle=2, variant='P', reward_mode='vector')
        td = env.reset(td)
        steps, td, max_cap, looped = _drive(env, td, np.random.default_rng(1))
        served = int((td["visited"][0, 0, 1:] != 0).sum())
        self.assertFalse(looped)
        self.assertEqual(served, 3)
        self.assertLessEqual(max_cap, 1.0 + 1e-9)

    def test_deadend_penalty_proportional_to_unserved_demand(self):
        """Penalty tracks unserved demand (not flat): more stranded demand ->
        strictly larger reward-vector penalty."""
        adj = np.ones((5, 5)) - np.eye(5)
        env = CARPEnv(num_vehicle=1, variant='P', reward_mode='vector')
        # strand ~0.6 (one arc)
        td1 = env.reset(_make_td([0, .6, .6, .0001, .0001], [0, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 1], adj, M=1))
        # strand ~1.2 (two arcs)
        td2 = env.reset(_make_td([0, .5, .6, .6, .6], [0, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 1], adj, M=1))
        _, td1, _, _ = _drive(env, td1, np.random.default_rng(2))
        _, td2, _, _ = _drive(env, td2, np.random.default_rng(3))
        r1 = float(env.get_reward(td1).max())
        r2 = float(env.get_reward(td2).max())
        self.assertGreater(r1, 100.0)
        self.assertGreater(r2, r1, "penalty is not proportional to unserved demand")


class TestCapacityInvariant(unittest.TestCase):
    def test_random_rollouts_respect_whole_route_budget(self):
        """Across many random feasible instances/variants: every rollout
        terminates and no vehicle's cumulative served demand ever exceeds Q."""
        rng = np.random.default_rng(7)
        for _ in range(60):
            n = int(rng.integers(3, 8)); n1 = n + 1
            adj = rng.random((n1, n1)); np.fill_diagonal(adj, 0.0)
            dem = np.concatenate([[0.0], rng.uniform(0.1, 0.5, n)])
            cl = np.concatenate([[0], rng.integers(1, 4, n)]).astype(np.int64)
            sv = np.concatenate([[0.0], rng.random(n) + 0.5])
            M = int(rng.integers(1, 4))
            variant = str(rng.choice(['P', 'U']))
            env = CARPEnv(num_vehicle=M, variant=variant, reward_mode='vector')
            td = env.reset(_make_td(dem, cl, sv, adj, M))
            steps, td, max_cap, looped = _drive(env, td, rng)
            self.assertFalse(looped, "rollout failed to terminate")
            self.assertLessEqual(max_cap, 1.0 + 1e-9,
                                 "a vehicle exceeded Q at some trajectory step")


class TestOracleWholeRouteBudget(unittest.TestCase):
    """The class-1 opening-prefix oracle must reject a partition that fits on
    its own but leaves no room for the higher-class arcs sharing the same
    single-trip budget (reload-bug fix, whole-route budget)."""

    def _run(self, class2_demand):
        from solvers.oracle import t1_heuristic_argmin
        n = 3
        adj = np.ones((n, n)) - np.eye(n)
        svc = np.array([0.0, 1.0, 1.0])
        clss = np.array([0, 1, 2], dtype=np.int64)
        demands = np.array([0.0, 0.9, class2_demand])   # class-1 = 0.9 (fits)
        return t1_heuristic_argmin(adj, svc, clss, demands, M=1, cap=1.0)

    def test_class1_fits_but_full_route_would_not_is_infeasible(self):
        # class-1 (0.9) fits alone, but a 0.5 class-2 arc can't share the 0.1
        # residual -> whole route 1.4 > Q -> oracle must report infeasible.
        res = self._run(0.5)
        self.assertFalse(np.isfinite(res["T1"]))
        self.assertIsNone(res["parts"])

    def test_class1_prefix_with_room_stays_feasible(self):
        # a 0.05 class-2 arc fits the 0.1 residual -> completable -> feasible.
        res = self._run(0.05)
        self.assertTrue(np.isfinite(res["T1"]))
        self.assertIsNotNone(res["parts"])


if __name__ == "__main__":
    unittest.main()
