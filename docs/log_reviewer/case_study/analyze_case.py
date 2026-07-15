"""Reconstruct the EVALUATED vehicle partition for every case-study solution
and verify it reproduces the .sol-reported (T1,T2,T3).

Why reconstruction is needed: the metaheuristics' .sol 'route' lines are
gen_tours(solver._last_best_action) -- the raw 0-separated arc ORDER. Their
reported T comes from SolutionEvaluator._build_P, which strips the separators
and rebuilds the M-vehicle partition itself (demand-balanced class chunks).
HARMONY's .sol lines, by contrast, are already the evaluated vehicles.

Outputs (per tier): verified vehicle partitions -> case/<alg>_<tier>_vehicles.json
plus per-vehicle class-1 completion times and class composition for the prose.
"""
import sys, json
sys.path.insert(0, '/home/quangnm/projects/ArcRoute/src')
import numpy as np
from utils.ops import import_instance
from solvers.evaluator import SolutionEvaluator

CASE = "/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm/case"


def parse_sol(path):
    Ts, lines = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("T1:"):
                p = line.replace("T1:", "").replace("T2:", "").replace("T3:", "").split()
                Ts = tuple(float(x) for x in p)
            elif line.startswith("route"):
                lines.append([int(a) for a in line.split(":")[1].split()])
    return Ts, lines


def per_vehicle_stats(vehicles, ev, td, clss):
    """Mimic _completion_times per vehicle: (class-1 completion, [c1,c2,c3] counts)."""
    out = []
    for veh_trips in vehicles:
        profiles = [ev._trip_profile(t, td) for t in veh_trips]
        if len(profiles) > 1:
            profiles.sort(key=lambda pr: (int(pr[0].min()), float(pr[2])))
        offset, c1_done = 0.0, 0.0
        counts = [0, 0, 0]
        for arc_class, arc_comp, duration in profiles:
            for c, ac in zip(arc_class.tolist(), arc_comp.tolist()):
                counts[int(c) - 1] += 1
                if c == 1:
                    c1_done = max(c1_done, offset + ac)
            offset += float(duration)
        out.append((c1_done, counts, float(offset)))
    return out


for tier in ["small", "large"]:
    dms, P, M, demands, clss, s, d, edge_indxs = import_instance(f"{CASE}/{tier}_case.npz")
    td = {"adj": dms, "service_times": s, "clss": clss, "demand": demands}
    nv = len(M)
    ev = SolutionEvaluator(variant="P")
    print(f"\n=== {tier} (M={nv}) ===")
    for alg in ["ils", "ea", "aco", "harmony"]:
        Ts, lines = parse_sol(f"{CASE}/{alg}_{tier}.sol")
        if alg == "harmony":
            # each line = one vehicle; interior 0s separate trips
            vehicles = []
            for ln in lines:
                trips, cur = [], []
                for a in ln:
                    if a == 0:
                        if cur:
                            trips.append(np.array(cur, dtype=np.int64))
                        cur = []
                    else:
                        cur.append(a)
                if cur:
                    trips.append(np.array(cur, dtype=np.int64))
                if trips:
                    vehicles.append(trips)
            ev._cache_td(td)
            T = ev._completion_times(vehicles, td)
        else:
            flat = []
            for ln in lines:
                flat.extend(a for a in ln if a != 0)
                flat.append(0)
            vehicles, T = ev(np.array(flat[:-1], dtype=np.int64), td, M=nv)
        ok = all(abs(T[k] - Ts[k]) < 1e-4 for k in range(3))
        stats = per_vehicle_stats(vehicles, ev, td, clss)
        c1s = ", ".join(f"{x[0]:.2f}" for x in stats)
        comps = [x[1] for x in stats]
        print(f"{alg:8s} reported={tuple(round(x,3) for x in Ts)} "
              f"recomputed=({T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}) match={'YES' if ok else 'NO'}")
        print(f"         per-vehicle class-1 completion: [{c1s}]  "
              f"class comp per vehicle [c1,c2,c3]: {comps}")
        with open(f"{CASE}/{alg}_{tier}_vehicles.json", "w") as fh:
            json.dump({"T": list(Ts),
                       "vehicles": [[t.tolist() for t in v] for v in vehicles]}, fh)
