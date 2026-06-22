"""Phase 1 gate tests for common.scheduler.Scheduler (M-agnostic policy + Φ)
and the calc_reward -> Scheduler rewire, incl. an integration rollout smoke.

Run: uv run python -m unittest tests.test_scheduler -v
"""
import unittest

import numpy as np
import torch

from common.scheduler import Scheduler


# ----------------------------------------------------------------- fixtures
def toy_instance(n=12, demand_each=0.24):
    """Single-instance td: depot 0 + n small required arcs (Σdemand≈2.88 -> k_min=3).
    Small arcs so the capacity-balanced re-split lands ~K segments smoothly."""
    clss = torch.tensor([0] + [1, 2, 3] * (n // 3), dtype=torch.int64)
    service_times = torch.tensor([0.0] + [1.0] * n)
    idx = torch.arange(n + 1, dtype=torch.float64)
    adj = (idx[:, None] - idx[None, :]).abs() * 0.3            # (n+1, n+1)
    demand = torch.tensor([0.0] + [demand_each] * n)
    return {
        "clss": clss,
        "service_times": service_times,
        "adj": adj,
        "demand": demand,
        "vehicle_capacity": torch.tensor([1.0]),
    }


# arc order 1..n, with depot 0s interleaved (must be IGNORED by the Scheduler).
def order_action(n=12):
    seq = []
    for i in range(1, n + 1):
        seq.append(i)
        if i % 3 == 0:
            seq.append(0)
    return np.array(seq, dtype=np.int64)


def parallel_ref(segments, td, pos_val=(1, 2, 3)):
    """Independent reference: classic single-trip parallel makespan per class
    (each segment its own vehicle, starting at t=0)."""
    per = {p: [0.0] for p in pos_val}
    for seg in segments:
        path = np.concatenate(([0], np.asarray(seg, dtype=np.int64), [0]))
        service = td["service_times"][path].to(torch.float64)
        travel = td["adj"][path[:-1], path[1:]].to(torch.float64)
        t = service.clone()
        t[1:] = t[1:] + travel
        cum = torch.cumsum(t, dim=0)
        for i, arc in enumerate(seg.tolist()):
            c = int(td["clss"][arc])
            if c in per:
                per[c].append(float(cum[i + 1]))
    return np.array([max(per[p]) for p in pos_val], dtype=np.float64)


def n_trips(vehicles):
    return sum(len(v) for v in vehicles)


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.td = toy_instance()
        self.action = order_action()

    # U-path accounting reduces to the classic parallel formula (1 trip / vehicle).
    def test_accounting_matches_parallel_reference(self):
        sched = Scheduler(variant="U")
        for M in (3, 6, 12):                       # M >= k_min -> all parallel
            segs = sched._segment(self.action, self.td, M)
            _, T = sched(self.action, self.td, M=M)
            np.testing.assert_allclose(T, parallel_ref(segs, self.td), atol=1e-9)

    # U re-split ignores the policy's 0s and respects the hard capacity cap.
    def test_resplit_feasible_and_ignores_depot(self):
        sched = Scheduler(variant="U")
        segs = sched._segment(self.action, self.td, M=5)
        dem = self.td["demand"]
        for seg in segs:
            self.assertLessEqual(float(dem[seg].sum()), 1.0 + 1e-9)
        # all 12 arcs present exactly once (0s dropped)
        self.assertEqual(sorted(int(a) for s in segs for a in s), list(range(1, 13)))

    # ⭐ M matters: more vehicles -> more (smaller) segments -> lower T_1.
    def test_M_matters_more_vehicles_lower_makespan(self):
        sched = Scheduler()
        v3, T3 = sched(self.action, self.td, M=3)
        v6, T6 = sched(self.action, self.td, M=6)
        self.assertGreater(n_trips(v6), n_trips(v3))      # uses more vehicles
        self.assertGreaterEqual(T3[0] + 1e-9, T6[0])      # makespan не increases

    # ⭐ M=2 < k_min: total/M > cap -> each vehicle needs reloads (multi-trip).
    def test_m2_multitrip(self):
        sched = Scheduler()                                # default P
        vehicles, T = sched(self.action, self.td, M=2)
        self.assertEqual(len(vehicles), 2)
        self.assertTrue(any(len(v) >= 2 for v in vehicles))  # some vehicle reloads
        self.assertTrue(np.all(np.isfinite(T)))
        _, Tm3 = sched(self.action, self.td, M=3)
        self.assertGreaterEqual(T[0] + 1e-9, Tm3[0])      # fewer vehicles >= more

    def test_monotonic_in_M(self):
        sched = Scheduler()
        ts = [sched(self.action, self.td, M=m)[1][0] for m in (1, 2, 3, 6)]
        for a, b in zip(ts, ts[1:]):
            self.assertGreaterEqual(a + 1e-9, b)

    def test_deterministic(self):
        sched = Scheduler()
        a = sched(self.action, self.td, M=2)[1]
        b = sched(self.action, self.td, M=2)[1]
        np.testing.assert_array_equal(a, b)

    def test_variants_run(self):
        for variant in ("P", "U"):
            _, T = Scheduler(variant=variant)(self.action, self.td, M=2)
            self.assertEqual(T.shape, (3,))
            self.assertTrue(np.all(np.isfinite(T)))

    # ⭐ P routes respect precedence: each route is non-decreasing in class.
    def test_P_routes_precedence(self):
        clss = self.td["clss"].numpy()
        for M in (2, 3, 5, 6):
            veh, _ = Scheduler(variant="P")(self.action, self.td, M=M)
            for v in veh:
                for trip in v:
                    self.assertTrue(bool(np.all(np.diff(clss[trip]) >= 0)),
                                    f"non-monotone route {clss[trip].tolist()} M={M}")

    # bug repro: an order that within depot-segments is 1,2,3 but concatenates to
    # ...3 | 1,2... must NOT yield a non-monotone P route.
    def test_P_bug_repro(self):
        td = toy_instance(n=6, demand_each=0.4)
        td["clss"] = torch.tensor([0, 1, 2, 3, 1, 2, 3], dtype=torch.int64)
        action = np.array([1, 2, 3, 0, 4, 5, 6])          # 1,2,3 | 1,2,3
        clss = td["clss"].numpy()
        veh, _ = Scheduler(variant="P")(action, td, M=2)
        for v in veh:
            for trip in v:
                self.assertTrue(bool(np.all(np.diff(clss[trip]) >= 0)))

    # ⭐ P spreads class-1 across the vehicles (=> low T_1) and T_1 drops with M.
    def test_P_class1_spread_and_T1_decreases(self):
        clss = self.td["clss"].numpy()
        n_c1 = int((clss == 1).sum())
        t1_prev = None
        for M in (2, 3, 6):
            veh, T = Scheduler(variant="P")(self.action, self.td, M=M)
            c1_vehicles = sum(1 for v in veh if any(1 in clss[t] for t in v))
            self.assertGreaterEqual(c1_vehicles, min(M, n_c1) - 1)   # ~ spread to M
            if t1_prev is not None:
                self.assertLessEqual(T[0], t1_prev + 1e-9)          # T_1 non-increasing
            t1_prev = T[0]

    # P and U now genuinely DIFFER (P pays for precedence); both stay feasible.
    def test_P_and_U_differ(self):
        tp = Scheduler(variant="P")(self.action, self.td, M=3)[1]
        tu = Scheduler(variant="U")(self.action, self.td, M=3)[1]
        self.assertFalse(np.allclose(tp, tu))

    # [assign] priority-first within a vehicle: lower class runs earlier.
    def test_order_trips_priority_first(self):
        sched = Scheduler()
        c3 = np.array([1])      # arc 1 is class 2 in the toy; build explicit td below
        td = toy_instance()
        td["clss"] = torch.tensor([0, 3, 3, 1, 1] + [2] * 8, dtype=torch.int64)
        ordered = sched._order_trips([np.array([1]), np.array([3])], td)  # class3, class1
        self.assertEqual(int(td["clss"][ordered[0][0]]), 1)   # class-1 trip first

    # multi-trip (M=1): the class-1 trip is LAST in the order but, ordered
    # priority-first on the vehicle, it runs first => T_1 < T_3.
    def test_multitrip_priority_reduces_T1(self):
        td = toy_instance()
        # arcs 1..8 = class 3 (first trips), arcs 9..12 = class 1 (last trip)
        td["clss"] = torch.tensor([0] + [3] * 8 + [1] * 4, dtype=torch.int64)
        action = np.arange(1, 13)
        _, T = Scheduler()(action, td, M=1)            # all 3 trips on one vehicle
        self.assertLess(T[0], T[2])                    # class 1 (late) finishes first

    def test_empty_class_is_zero(self):
        td = toy_instance()
        td["clss"] = torch.tensor([0] + [1] * 12, dtype=torch.int64)  # only class 1
        _, T = Scheduler()(self.action, td, M=3)
        self.assertAlmostEqual(T[1], 0.0)
        self.assertAlmostEqual(T[2], 0.0)

    def test_reads_M_from_td(self):
        td = toy_instance()
        td["num_vehicle"] = torch.tensor([2])
        _, T = Scheduler()(self.action, td)
        self.assertTrue(np.all(np.isfinite(T)))

    # calc_reward now delegates to the Scheduler (reads td['num_vehicle']).
    def test_calc_reward_delegates(self):
        from common.cal_reward import calc_reward
        td = toy_instance()
        td["num_vehicle"] = torch.tensor([3])
        rs = calc_reward(self.action, td, return_numpy=True)
        _, T = Scheduler()(self.action, td, M=3)
        np.testing.assert_allclose(rs.astype(np.float64), T, atol=1e-9)


class TestRolloutSmoke(unittest.TestCase):
    """⭐ env.reset -> policy(M-agnostic) -> Scheduler reward -> backward,
    across M and variants. reward finite, gradients finite."""

    def _run(self, M, variant):
        from env.env import CARPEnv
        from env.generator import generate_dataset
        from policy.policy import AttentionModelPolicy

        torch.manual_seed(0)
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=M, variant=variant)
        # get_reward's run_parallel uses num_epochs=10 -> batch must be >= 10.
        batch = generate_dataset(12, 15, 15, M, num_workers=0)
        td = env.reset(batch)
        self.assertIn("num_vehicle", td.keys())
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
        out = policy(td.clone(), env, phase="train")
        reward = out["reward"]
        self.assertTrue(torch.isfinite(reward).all())
        loss = -out["log_likelihood"].sum()
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_rollout_smoke_grid(self):
        for M in (2, 3, 7):
            for variant in ("P", "U"):
                with self.subTest(M=M, variant=variant):
                    self._run(M, variant)

    def test_reward_equals_weighted_objective(self):
        """RL reward = -(T . obj_weights) (hierarchical), not just -T_1, so the
        policy optimises the same lexicographic objective as the baselines and
        T_2/T_3 get a (tie-breaking) signal."""
        from env.env import CARPEnv
        from env.generator import generate_dataset
        from policy.policy import AttentionModelPolicy

        torch.manual_seed(0)
        w = [1.0, 1e-2, 1e-4]                                # default (scaled) weights
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P", obj_weights=w)
        td = env.reset(generate_dataset(12, 15, 15, 3, num_workers=0))
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
        out = policy(td.clone(), env, decode_type="greedy")
        T = np.asarray(env.get_objective(td, out["actions"]))           # (B,3)
        expected = -(T * np.array(w)).sum(-1)
        got = out["reward"].detach().cpu().numpy().reshape(-1)
        np.testing.assert_allclose(got, expected, rtol=1e-4)

    def test_rollout_smoke_mixed_fleet(self):
        """Phase 3: a single batch with MIXED M per instance flows through
        rollout -> per-instance Scheduler reward -> backward (M is scalar, so no
        torch.cat breakage; each instance is rewarded under its own M)."""
        from env.env import CARPEnv
        from env.generator import generate_dataset
        from policy.policy import AttentionModelPolicy

        torch.manual_seed(0)
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=[2, 3, 7], variant="P")
        batch = generate_dataset(12, 15, 15, [2, 3, 7], num_workers=0)
        self.assertGreaterEqual(len(set(batch["num_vehicle"].tolist())), 2)
        td = env.reset(batch)
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
        out = policy(td.clone(), env, phase="train")
        self.assertTrue(torch.isfinite(out["reward"]).all())
        (-out["log_likelihood"].sum()).backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
