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

The two variants produce DIFFERENT routes (though the T_k *formula* is shared per
Ha 2024 = max completion over class-k arcs):
- 'P' (precedence): each route must be non-decreasing in class. The env mask makes
  the order globally class-sorted; `_build_P` splits EACH class across the M
  vehicles (chunks rotated for load balance) so class-1 is spread (low T_1), then
  inserts capacity reloads -> routes are precedence-feasible by construction and
  T_1 drops as M grows.
- 'U' (upgrading): no precedence -> `_segment` re-partitions into max(M, k_min)
  capacity-feasible segments and `_assign` maps them to M vehicles (LPT multi-trip
  when k_min > M); `_order_trips` puts higher-priority trips first.
`_completion_times` (sequential offsets for multi-trip) is shared by both.
=> P generally costs more than U; they are NOT interchangeable.

TODO[assign]: U trip->vehicle assignment is greedy LPT; P chunking is a balanced
heuristic — neither is provably optimal.
"""

import numpy as np
import torch


class Scheduler:
    def __init__(self, variant: str = "P", pos_val=(1, 2, 3)):
        # variant: 'P' (precedence) or 'U' (upgrading). Informational only — T_k is
        # variant-independent (max over class-k arcs); the P/U difference is in the
        # rollout mask, not here. Kept for API clarity.
        self.variant = variant
        self.pos_val = list(pos_val)
        # numpy caches populated by __call__ (or lazily by _cache_td on first use).
        self._svc_np = self._adj_np = self._cls_np = None

    def _cache_td(self, td):
        """Cache hot-path tensors as numpy arrays. Called once per __call__;
        also triggered lazily when internal methods are called directly."""
        self._adj_np = self._np(td["adj"]).astype(np.float64)
        self._svc_np = self._np(td["service_times"]).astype(np.float64)
        self._cls_np = self._np(td["clss"]).reshape(-1).astype(np.int64)

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

        # Convert hot-path tensors to numpy ONCE per call. _trip_profile is
        # invoked O(M*trips) times per instance; torch per-op overhead (~µs/op)
        # on tiny arrays dominates over compute at that scale. Pure numpy avoids
        # the kernel-launch cost entirely.
        self._cache_td(td)

        if self.variant == "P":
            # HDCARP-P: the env mask makes the order globally class-sorted; split
            # EACH class across the M vehicles (spread) and order each route class 1
            # -> p (precedence). Capacity reloads split a route into trips.
            vehicles = self._build_P(action, td, M)
        else:
            # HDCARP-U: no precedence -> capacity re-split + LPT multi-trip.
            trips = self._segment(action, td, M)
            n_veh = len(trips) if M is None else int(M)
            n_veh = max(n_veh, 1) if trips else 0
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
            return float(self._np(td["vehicle_capacity"]).reshape(-1)[0])
        return 1.0

    @staticmethod
    def _np(x):
        """Detach-safe conversion to a numpy array (demand/capacity never carry
        grad, but be defensive)."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

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

        demand = self._np(td["demand"]).astype(np.float64).reshape(-1)
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

    # --- HDCARP-P: per-class spread across M vehicles -------------------------
    def _build_P(self, action, td, M):
        """Build M vehicle schedules for variant P. The env mask makes `action`
        globally class-sorted, so each class is a coherent contiguous block. Split
        EACH class's block into M demand-balanced contiguous chunks; vehicle m's
        route = chunk_1m · chunk_2m · … · chunk_pm (non-decreasing class). Capacity
        reloads then split each route into trips. => every class is spread across
        all M vehicles (low T_1), and every route respects precedence."""
        a = np.asarray(action).astype(np.int64).ravel()
        order = a[a != 0]
        M = max(int(M), 1)
        vehicles = [[] for _ in range(M)]
        if len(order) == 0:
            return vehicles
        clss = self._cls_np
        demand = self._np(td["demand"]).astype(np.float64).reshape(-1)
        cap = self._capacity(td)

        routes = [[] for _ in range(M)]                     # per-vehicle arc lists
        ocls = clss[order]
        for ki, k in enumerate(np.unique(ocls)):            # classes ascending
            block = order[ocls == k]                        # coherent class-k tour
            # Rotate which vehicle gets the heavier chunk per class so no single
            # vehicle is heavy in every class (balances loads -> avoids needless
            # capacity reloads). Order within a route stays class-ascending.
            for i, chunk in enumerate(self._balanced_chunks(block, demand, M)):
                routes[(i + ki) % M].extend(chunk.tolist())
        for m in range(M):
            vehicles[m] = self._split_by_capacity(np.array(routes[m], dtype=np.int64),
                                                  demand, cap)
        return vehicles

    @staticmethod
    def _balanced_chunks(block, demand, n):
        """Split a 1D arc array into `n` demand-balanced CONTIGUOUS chunks by
        cumulative-demand quantile (arc -> chunk ⌊cum_before / (total/n)⌋). Spreads
        evenly; some chunks may be empty if len(block) < n."""
        chunks = [[] for _ in range(n)]
        if len(block) == 0:
            return [np.array([], dtype=np.int64) for _ in range(n)]
        d = demand[block]
        target = float(d.sum()) / n
        cum = 0.0
        for arc, dd in zip(block.tolist(), d.tolist()):
            c = min(n - 1, int(cum / target)) if target > 1e-12 else 0
            chunks[c].append(arc)
            cum += dd
        return [np.array(c, dtype=np.int64) for c in chunks]

    @staticmethod
    def _split_by_capacity(route, demand, cap):
        """Split a class-ordered route into capacity-feasible trips (insert a depot
        reload whenever the next arc would exceed cap). Each trip stays a contiguous
        slice -> still non-decreasing in class."""
        trips, cur, load = [], [], 0.0
        for arc in route.tolist():
            d = float(demand[arc])
            if cur and load + d > cap + 1e-9:
                trips.append(np.array(cur, dtype=np.int64)); cur, load = [], 0.0
            cur.append(arc); load += d
        if cur:
            trips.append(np.array(cur, dtype=np.int64))
        return trips

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
        return vehicles

    def _order_trips(self, veh_trips, td):
        """Order a vehicle's trips so higher-priority (lower class) arcs finish
        first -> lower T_1 then T_2 (lexicographic objective). No-op for ≤1 trip.
        Key: (min class present, duration) ascending."""
        if len(veh_trips) <= 1:
            return veh_trips
        if self._cls_np is None:
            self._cache_td(td)
        clss = self._cls_np

        def key(trip):
            cs = clss[np.asarray(trip, dtype=np.int64)]
            return (int(cs.min()), float(self._trip_profile(trip, td)[2]))

        return sorted(veh_trips, key=key)

    def _trip_profile(self, trip, td):
        """For one trip (1D arc indices, no depot) return:
        (arc_class, arc_completion, duration), mirroring `calc_reward`'s math.
          path = [depot, *trip, depot]; cost per step = service + incoming travel.

        Uses self._svc_np / _adj_np / _cls_np cached by __call__ to avoid
        repeated torch tensor indexing (kernel-launch overhead >> compute on
        tiny arrays of 10-30 arcs).
        """
        if self._svc_np is None:
            self._cache_td(td)
        trip = np.asarray(trip, dtype=np.int64)
        path = np.empty(len(trip) + 2, dtype=np.int64)
        path[0] = 0
        path[1:-1] = trip
        path[-1] = 0
        t = self._svc_np[path].copy()
        t[1:] += self._adj_np[path[:-1], path[1:]]
        cum = np.cumsum(t)
        n = len(trip)
        return self._cls_np[trip], cum[1:1 + n], cum[-1]

    def _completion_times(self, vehicles, td):
        """Hierarchical completion times T_k = max absolute completion over arcs
        of class k. A trip's arcs start at the vehicle's running offset (sum of
        durations of that vehicle's earlier trips) → multi-trip accounting.

        Single-trip (offset 0 everywhere) reproduces `calc_reward` exactly.
        """
        per_class = {p: [0.0] for p in self.pos_val}     # empty class -> T_k = 0
        for veh_trips in vehicles:
            offset = 0.0
            for trip in self._order_trips(veh_trips, td):   # priority-first

                arc_class, arc_comp, duration = self._trip_profile(trip, td)
                for c, ac in zip(arc_class.tolist(), arc_comp.tolist()):
                    if c in per_class:
                        per_class[c].append(offset + ac)
                offset += float(duration)
        return np.array([max(per_class[p]) for p in self.pos_val], dtype=np.float64)
