"""Scheduler Φ — maps a policy's arc ORDER to the routes of M vehicles and the
hierarchical completion times (T_1, ..., T_p).

Design (dynamic_plan Phase 1):
  - The policy is M-AGNOSTIC: it only produces an ordered action sequence `α`
    (arc indices, with 0 = depot). M does NOT enter the encoder/decoder/mask.
  - This Scheduler is where M lives. Q (vehicle capacity) is FIXED = Σq/3 + 0.5
    (Ha 2024), so the per-trip capacity constraint is already satisfied by the
    rollout mask; trips here are taken as capacity-feasible.
  - Multi-trip: trips are assigned to M vehicles; a vehicle with ≥2 trips runs
    them back-to-back (returns to depot to reload). When #trips ≤ M this reduces
    to the classic single-trip parallel model and reproduces `calc_reward`.

Because Q is fixed (not M-dependent), one policy rollout serves ANY M — only this
Scheduler is re-run per M ("train once, M is an eval-time parameter").

Core pieces: `_segment` re-partitions the order into max(M, k_min) capacity-
feasible segments (so M genuinely controls the split); `_assign` maps segments to
M vehicles (LPT multi-trip when k_min > M); `_completion_times` does the
hierarchical T_k accounting (sequential offsets for multi-trip).

TODOs (enhancements):
  - [hierarchy] HDCARP-U: compute T_k via hierarchy-levels (Ha 2024 §problem)
    instead of raw per-class max (currently identical for P and U).
  - [assign] Replace greedy LPT with hierarchical-aware / optimal assignment
    (order trips within a vehicle so high-priority arcs finish first).
"""

import numpy as np
import torch


class Scheduler:
    def __init__(self, variant: str = "P", pos_val=(1, 2, 3)):
        # variant: 'P' (precedence) or 'U' (upgrading). Currently only affects the
        # (TODO) hierarchy-level computation; T_k formula below is shared.
        self.variant = variant
        self.pos_val = list(pos_val)

    # ------------------------------------------------------------------ public
    def __call__(self, action, td, M=None):
        """Return (vehicles, T_vec).

        action : 1D sequence of arc indices (0 = depot). np.ndarray or torch.
        td     : single-instance mapping with 'service_times', 'adj', 'clss'
                 (1D/2D tensors, depot at index 0). Optional 'num_vehicle'.
        M      : number of vehicles; if None, read from td['num_vehicle'].
        vehicles : list (len M) of lists-of-trips (each trip = 1D arc-index array).
        T_vec  : np.ndarray (len p) of hierarchical completion times.
        """
        if M is None and "num_vehicle" in td:
            M = int(td["num_vehicle"])
        # M may still be None -> parallel: one vehicle per capacity-min segment.

        trips = self._segment(action, td, M)
        n_veh = len(trips) if M is None else int(M)
        vehicles = self._assign(trips, td, n_veh)
        T_vec = self._completion_times(vehicles, td)
        return vehicles, T_vec

    def reward(self, action, td, M=None):
        """Scalar RL reward = -T_1 (maximise => minimise top-priority makespan).
        (Phase 1: `calc_reward` will delegate here.)"""
        _, T = self.__call__(action, td, M)
        return -float(T[0])

    # ------------------------------------------------------------- internals
    def _capacity(self, td):
        """Per-route capacity in the normalized space (demands are divided by Q in
        the data, so capacity = vehicle_capacity = 1.0)."""
        if "vehicle_capacity" in td:
            return float(np.asarray(td["vehicle_capacity"]).reshape(-1)[0])
        return 1.0

    def _segment(self, action, td, M):
        """Re-partition the policy's ORDER into `K = max(M, k_min)` capacity-
        feasible contiguous segments (k_min = ⌈Σdemand/cap⌉ = minimum #trips).

        The policy's own depot returns (0s) are IGNORED — the policy is M-agnostic
        and only provides the arc ORDER; the Scheduler owns the partition so that
        the SAME rollout serves any M (more vehicles → finer split → lower makespan).

        Greedy with a balance target `total/K` and a hard cap, so each segment's
        demand ≤ cap and the count lands in [k_min, K].
        """
        a = np.asarray(action).astype(np.int64).ravel()
        order = a[a != 0]                                  # pure arc order
        if len(order) == 0:
            return []

        demand = np.asarray(td["demand"]).astype(np.float64).reshape(-1)
        cap = self._capacity(td)
        loads = demand[order]
        total = float(loads.sum())
        k_min = max(1, int(np.ceil(total / cap - 1e-9)))
        target = k_min if M is None else max(int(M), k_min)
        K = min(target, len(order))                        # can't exceed #arcs
        soft = total / K                                   # ≤ cap since K ≥ k_min

        segs, cur, load, closed = [], [], 0.0, 0
        for arc, d in zip(order.tolist(), loads.tolist()):
            over_cap = load + d > cap + 1e-9
            balanced = load >= soft - 1e-9 and closed < K - 1
            if cur and (over_cap or balanced):
                segs.append(np.array(cur, dtype=np.int64))
                cur, load, closed = [], 0.0, closed + 1
            cur.append(arc)
            load += d
        if cur:
            segs.append(np.array(cur, dtype=np.int64))
        return segs

    def _assign(self, trips, td, M):
        """Assign trips to M vehicles. ≤M trips → one per vehicle (single-trip,
        parallel). >M trips → greedy LPT (longest trip to least-loaded vehicle),
        producing multi-trip schedules."""
        vehicles = [[] for _ in range(M)]
        if len(trips) == 0:
            return vehicles
        if len(trips) <= M:
            for i, t in enumerate(trips):
                vehicles[i].append(t)
            return vehicles

        # multi-trip: LPT on trip duration
        durations = [float(self._trip_profile(t, td)[2]) for t in trips]
        order = np.argsort(durations)[::-1]  # longest first
        loads = np.zeros(M)
        for idx in order:
            v = int(np.argmin(loads))
            vehicles[v].append(trips[idx])
            loads[v] += durations[idx]
        # TODO[assign]: within each vehicle, order trips so high-priority arcs
        # finish early (hierarchical objective). Skeleton keeps assignment order.
        return vehicles

    def _trip_profile(self, trip, td):
        """For one trip (1D arc indices, no depot) return:
        (arc_class, arc_completion, duration), mirroring `calc_reward`'s math.
          path = [depot, *trip, depot]; cost per step = service + incoming travel.
        """
        path = np.concatenate(([0], np.asarray(trip, dtype=np.int64), [0]))
        service = td["service_times"][path].to(torch.float64)
        travel = td["adj"][path[:-1], path[1:]].to(torch.float64)
        t = service.clone()
        t[1:] = t[1:] + travel
        cum = torch.cumsum(t, dim=0)
        n = len(trip)
        arc_completion = cum[1 : 1 + n]                  # completion per arc
        arc_class = td["clss"][np.asarray(trip, dtype=np.int64)]
        duration = cum[-1]                               # back at depot
        return arc_class, arc_completion, duration

    def _completion_times(self, vehicles, td):
        """Hierarchical completion times T_k = max absolute completion over arcs
        of class k. A trip's arcs start at the vehicle's running offset (sum of
        durations of that vehicle's earlier trips) → multi-trip accounting.

        Single-trip (offset 0 everywhere) reproduces `calc_reward` exactly.
        """
        per_class = {p: [0.0] for p in self.pos_val}     # empty class -> T_k = 0
        for veh_trips in vehicles:
            offset = 0.0
            for trip in veh_trips:
                arc_class, arc_comp, duration = self._trip_profile(trip, td)
                for c, ac in zip(arc_class.tolist(), arc_comp.tolist()):
                    if c in per_class:
                        per_class[c].append(offset + ac)
                offset += float(duration)
        return np.array([max(per_class[p]) for p in self.pos_val], dtype=np.float64)
