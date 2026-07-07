"""HDCARP solution evaluator: action sequence → per-vehicle routes → T_k.

Two input modes:
  - Arc-order (metaheuristics): action = arc indices with 0-separators.
    SolutionEvaluator partitions them into M capacity-feasible trips.
  - Joint M×N (rl.py inference): action = flat m*(N+1)+j indices.
    Routes are extracted directly via `routes_from_joint_actions`.

Variants:
  - 'P' (precedence): `_build_P` spreads each class across M vehicles then
    inserts capacity reloads. Routes are globally class-sorted.
  - 'U' (upgrading): lex-better of the order-preserving segment decode
    (`_segment` + greedy LPT `_assign`) and the `_build_P` schedule — U's
    feasible set is a superset of P's, so the U decoder must never score an
    action worse than the P decoder.

T_k = max absolute completion time over class-k arcs (both variants).
"""

import numpy as np
import torch


class SolutionEvaluator:
    def __init__(self, variant: str = "P", pos_val=(1, 2, 3)):
        self.variant = variant
        self.pos_val = list(pos_val)
        self._svc_np = self._adj_np = self._cls_np = None

    def _cache_td(self, td):
        # copy=False: skip the (potentially large) matrix copies when the
        # dtype already matches -- the evaluator only ever reads these.
        self._adj_np = self._np(td["adj"]).astype(np.float64, copy=False)
        self._svc_np = self._np(td["service_times"]).astype(np.float64, copy=False)
        self._cls_np = self._np(td["clss"]).reshape(-1).astype(np.int64, copy=False)

    # ------------------------------------------------------------------ public
    def routes_from_joint_actions(self, actions, M, N1):
        """Decode flat Joint M×N action sequence → per-vehicle trips.

        actions : 1D array, action = vehicle_m * N1 + arc_j (j==0 → depot).
        Returns : list (len M) of lists-of-trips (each trip = 1D arc-index array).
        """
        M = max(int(M), 1)
        vehicles = [[] for _ in range(M)]
        cur_trip = [[] for _ in range(M)]
        for a in np.asarray(actions).ravel().tolist():
            m = int(a) // N1
            j = int(a) % N1
            if m >= M:
                continue
            if j == 0:
                if cur_trip[m]:
                    vehicles[m].append(np.array(cur_trip[m], dtype=np.int64))
                    cur_trip[m] = []
            else:
                cur_trip[m].append(j)
        for m in range(M):
            if cur_trip[m]:
                vehicles[m].append(np.array(cur_trip[m], dtype=np.int64))
        return vehicles

    def __call__(self, action, td, M=None, joint=False):
        """Evaluate one instance. Return (vehicles, T_vec).

        action  : 1D arc-order sequence or Joint M×N flat index sequence.
        td      : single-instance dict with 'service_times', 'adj', 'clss'.
        M       : vehicle count; reads td['num_vehicle'] if None.
        joint   : force Joint M×N decode (else auto-detected).
        """
        if M is None and "num_vehicle" in td:
            M = int(td["num_vehicle"])
        self._cache_td(td)

        a = np.asarray(action).ravel()
        N1 = self._svc_np.shape[0]
        is_joint = joint or (a.size > 0 and int(a.max()) >= N1)
        if is_joint:
            vehicles = self.routes_from_joint_actions(a, M, N1)
            return vehicles, self._completion_times(vehicles, td)

        if self.variant == "P":
            vehicles = self._build_P(action, td, M)
            return vehicles, self._completion_times(vehicles, td)

        # Variant U. U's scheduling freedom is a strict superset of P's:
        # every class-sorted P schedule is U-feasible, so a correct U decoder
        # must never score an action WORSE than the P decoder does. The
        # segment+LPT decode alone violated that (its greedy contiguous
        # partition + duration-LPT assignment isn't T1-aware, e.g. it can
        # stack all class-1 arcs onto one vehicle where _build_P spreads
        # them) — decode both ways and return the lex-better schedule.
        vehicles_u = self._decode_U_segment(action, td, M)
        T_u = self._completion_times(vehicles_u, td)
        vehicles_p = self._build_P(action, td, self._fleet_size(action, td, M))
        T_p = self._completion_times(vehicles_p, td)
        if tuple(T_p.tolist()) < tuple(T_u.tolist()):
            return vehicles_p, T_p
        return vehicles_u, T_u

    # ------------------------------------------------------------- internals
    def _capacity(self, td):
        if "vehicle_capacity" in td:
            return float(self._np(td["vehicle_capacity"]).reshape(-1)[0])
        return 1.0

    @staticmethod
    def _np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _fleet_size(self, action, td, M):
        """Effective vehicle count: M when known, else #capacity segments."""
        if M is not None:
            return max(int(M), 1)
        return max(len(self._segment(action, td, M)), 1)

    def _decode_U_segment(self, action, td, M):
        """Order-preserving U decode: contiguous capacity segments + LPT.

        Fleet size is a hard constraint: exactly M vehicles when M is known.
        (A previous revision used max(M, len(trips)), which gave every extra
        capacity-reload trip its own phantom vehicle and made _assign's LPT
        branch dead code — U objectives came out unrealistically low whenever
        #trips > M.)
        """
        trips = self._segment(action, td, M)
        n_veh = max(int(M), 1) if M is not None else max(len(trips), 1)
        return self._assign(trips, td, n_veh)

    def _segment(self, action, td, M):
        """Partition arc-order into K = max(M, k_min) capacity-feasible segments."""
        a = np.asarray(action).astype(np.int64).ravel()
        order = a[a != 0]
        if len(order) == 0:
            return []
        demand = self._np(td["demand"]).astype(np.float64).reshape(-1)
        cap = self._capacity(td)
        loads = demand[order]
        total = float(loads.sum())
        k_min = max(1, int(np.ceil(total / cap - 1e-9)))
        target = k_min if M is None else max(int(M), k_min)
        K = min(target, len(order))
        soft = total / K

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

    def _build_P(self, action, td, M):
        """Build M vehicle routes for variant P (global precedence).

        Spreads each class-k block across M vehicles in demand-balanced chunks,
        then splits by capacity. All routes are non-decreasing in class.
        """
        a = np.asarray(action).astype(np.int64).ravel()
        order = a[a != 0]
        M = max(int(M), 1)
        vehicles = [[] for _ in range(M)]
        if len(order) == 0:
            return vehicles
        clss = self._cls_np
        demand = self._np(td["demand"]).astype(np.float64).reshape(-1)
        cap = self._capacity(td)

        routes = [[] for _ in range(M)]
        ocls = clss[order]
        for ki, k in enumerate(np.unique(ocls)):
            block = order[ocls == k]
            for i, chunk in enumerate(self._balanced_chunks(block, demand, M)):
                routes[(i + ki) % M].extend(chunk.tolist())
        for m in range(M):
            vehicles[m] = self._split_by_capacity(
                np.array(routes[m], dtype=np.int64), demand, cap)
        return vehicles

    @staticmethod
    def _balanced_chunks(block, demand, n):
        """Split block into n demand-balanced contiguous chunks."""
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
        """Split route into capacity-feasible trips."""
        trips, cur, load = [], [], 0.0
        for arc in route.tolist():
            d = float(demand[arc])
            if cur and load + d > cap + 1e-9:
                trips.append(np.array(cur, dtype=np.int64))
                cur, load = [], 0.0
            cur.append(arc)
            load += d
        if cur:
            trips.append(np.array(cur, dtype=np.int64))
        return trips

    def _assign(self, trips, td, M):
        """Assign trips to M vehicles. Greedy LPT when #trips > M."""
        vehicles = [[] for _ in range(M)]
        if not trips:
            return vehicles
        if len(trips) <= M:
            for i, t in enumerate(trips):
                vehicles[i].append(t)
            return vehicles
        durations = [float(self._trip_profile(t, td)[2]) for t in trips]
        order = np.argsort(durations)[::-1]
        loads = np.zeros(M)
        for idx in order:
            v = int(np.argmin(loads))
            vehicles[v].append(trips[idx])
            loads[v] += durations[idx]
        return vehicles

    def _trip_profile(self, trip, td):
        """Return (arc_class, arc_completion, duration) for one trip."""
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
        """T_k = max absolute completion time over class-k arcs across all vehicles."""
        per_class = {p: [0.0] for p in self.pos_val}
        for veh_trips in vehicles:
            # Profile each trip once, then order a vehicle's trips lower class
            # first, then by duration (sorting the cached profiles avoids
            # re-profiling every trip inside the sort key).
            profiles = [self._trip_profile(t, td) for t in veh_trips]
            if len(profiles) > 1:
                profiles.sort(key=lambda pr: (int(pr[0].min()), float(pr[2])))
            offset = 0.0
            for arc_class, arc_comp, duration in profiles:
                for c, ac in zip(arc_class.tolist(), arc_comp.tolist()):
                    if c in per_class:
                        per_class[c].append(offset + ac)
                offset += float(duration)
        return np.array([max(per_class[p]) for p in self.pos_val], dtype=np.float64)


def evaluate_batch(vars, actions, pos_val=(1, 2, 3), variant='P'):
    """Compute T_k for a batch of arc-order action sequences (used by metaheuristics).

    vars    : dict with 'adj', 'service_time', 'clss', 'demand', optionally 'nv'.
    actions : 2D array (batch, n) — arc-index sequences with 0-separators.
    variant : 'P' (precedence route-builder) or 'U' (segment+assign decode).
    Returns : np.ndarray shape (batch, len(pos_val)).
    """
    td = {
        "adj":           vars["adj"],
        "service_times": vars["service_time"],
        "clss":          vars["clss"],
        "demand":        vars["demand"],
    }
    M = vars.get("nv", None)
    ev = SolutionEvaluator(variant=variant, pos_val=tuple(pos_val))
    return np.array([ev(np.asarray(a), td, M=M)[1] for a in actions],
                    dtype=np.float64)
