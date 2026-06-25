"""Run RL (GRPO) inference on one instance and write the route log.

Output format mirrors scripts/ils_log.py.

Usage:
    uv run python scripts/rl_log.py --ckpt <path.ckpt>
                                    [--file <path.npz>]
                                    [--variant P|U]
                                    [--vehicles M]
                                    [--num_sample 100]
                                    [--log outputs/rl_route_log.txt]
    uv run python scripts/rl_log.py \
            --ckpt outputs/checkpoints/curriculum_small/last.ckpt \
            --file data/ood/osm_cityA/40/38_17_358.npz \
            --vehicles 2 \
            --num_sample 100 \
            --log outputs/rl_route_log.txt
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from solvers.meta import InsertCheapestHCARP
from solvers.scheduler import Scheduler
from utils.ops import batchify, import_instance


# ── reuse print_route_log from ils_log verbatim ────────────────────────────
def print_route_log(routes, req, Ts, solver):
    adj = solver.vars["adj"]
    arc_info = {
        i + 1: dict(
            tail=int(req[i, 0]),
            head=int(req[i, 1]),
            clss=int(req[i, 3]),
            service=float(req[i, 4]),
            demand=float(solver.demands[i + 1]),
        )
        for i in range(len(req))
    }

    T_str = "  ".join(f"T{k + 1}={v:.4f}" for k, v in enumerate(Ts))
    print(f"\n{T_str}\n")

    for vid, route in enumerate(routes):
        arcs = route[1:-1]
        total_load = solver.demands[arcs].sum()
        bar = "=" * 60
        print(bar)
        print(f"  VEHICLE {vid + 1}   load={total_load:.4f}/1.0000")
        print(bar)

        chain = ["depot"]
        prev_head = None
        for arc_idx in arcs:
            info = arc_info[arc_idx]
            t, h = info["tail"], info["head"]
            if prev_head is not None and prev_head != t:
                chain.append(f"~~{prev_head}->{t}~~")
            chain.append(f"{t}->{h}")
            prev_head = h
        chain.append("depot")

        line, indent = "  ", "    "
        for k, part in enumerate(chain):
            sep = " -> " if k > 0 else ""
            if len(line) + len(sep) + len(part) > 70:
                print(line)
                line = indent + part
            else:
                line += sep + part
        print(line)
        print()

        hdr = f"  {'seq':>3}  {'arc':>3}  {'edge':>7}  {'cls'}  {'demand':>7}  {'service':>7}  {'dead':>7}  {'cum_load':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        cum = 0.0
        for pos, arc_idx in enumerate(arcs):
            info = arc_info[arc_idx]
            prev_arc = arcs[pos - 1] if pos > 0 else 0
            dead = float(adj[prev_arc, arc_idx])
            cum += info["demand"]
            print(
                f"  {pos + 1:>3}  {arc_idx:>3}  "
                f"{info['tail']:>2}->{info['head']:>2}  "
                f"  {info['clss']}  "
                f"{info['demand']:>7.4f}  "
                f"{info['service']:>7.4f}  "
                f"{dead:>7.4f}  "
                f"{cum:>8.4f}"
            )
        print()

        for p in [1, 2, 3]:
            class_arcs = [
                (pos, a) for pos, a in enumerate(arcs) if arc_info[a]["clss"] == p
            ]
            if not class_arcs:
                continue
            class_load = sum(arc_info[a]["demand"] for _, a in class_arcs)
            print(f"  Class {p}  ({len(class_arcs)} arcs, load={class_load:.4f})")
            for pos, arc_idx in class_arcs:
                info = arc_info[arc_idx]
                prev_arc = arcs[pos - 1] if pos > 0 else 0
                dead = float(adj[prev_arc, arc_idx])
                print(
                    f"    [seq {pos + 1:2d}] arc {arc_idx:2d}  "
                    f"{info['tail']:2d} -> {info['head']:2d}   "
                    f"demand={info['demand']:.4f}  "
                    f"service={info['service']:.4f}  "
                    f"dead={dead:.4f}"
                )
            print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="GRPO checkpoint (.ckpt)")
    parser.add_argument("--file", default="data/ood/osm_cityB/40/34_13_632.npz")
    parser.add_argument("--variant", default="P", choices=["P", "U"])
    parser.add_argument(
        "--vehicles",
        type=int,
        default=None,
        help="Override fleet size M (default: from instance)",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=100,
        help="Rollout samples; best is picked lexicographically",
    )
    parser.add_argument("--log", default="outputs/rl_route_log.txt")
    parser.add_argument("--solution", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load policy ────────────────────────────────────────────────────────
    from env.env import CARPEnv
    from policy.policy import AttentionModelPolicy

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    policy = AttentionModelPolicy(
        embed_dim=hp.get("embed_dim", 128),
        num_encoder_layers=hp.get("num_encoder_layers", 12),
        num_heads=hp.get("num_heads", 8),
    )
    sd = {
        k[len("policy.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("policy.")
    }
    policy.load_state_dict(sd, strict=True)
    policy = policy.to(device).eval()

    # ── load instance into RL env ──────────────────────────────────────────
    es = np.load(args.file)
    dms, P, M, demands, clss, s, d, _ = import_instance(args.file, M=args.vehicles)
    M = args.vehicles or M

    env = CARPEnv(
        num_loc=hp.get("num_loc", 40),
        num_arc=hp.get("num_arc", 80),
        variant=args.variant,
        reward_mode="vector",
    )

    from tensordict import TensorDict

    td = TensorDict(
        {
            "clss": torch.tensor(clss[None, :], dtype=torch.int64),
            "demands": torch.tensor(demands[None, :], dtype=torch.float32),
            "service_times": torch.tensor(s[None, :], dtype=torch.float32),
            "traversal_times": torch.tensor(d[None, :], dtype=torch.float32),
            "adj": torch.tensor(dms[None, :], dtype=torch.float32),
            "num_vehicle": torch.tensor([[M]], dtype=torch.int64),
        },
        batch_size=[1],
    )
    td = env.reset(td)
    td = td.to(device)

    # ── run K rollouts, pick lex-best ──────────────────────────────────────
    from time import time

    t0 = time()
    td_k = batchify(td, args.num_sample)
    with torch.inference_mode():
        out = policy(td_k, env=env, phase="test")

    Ts_all = out["reward"].cpu().numpy()  # (K, P)
    idx = int(np.lexsort((Ts_all[:, 2], Ts_all[:, 1], Ts_all[:, 0]))[0])
    best_T = Ts_all[idx]
    best_actions = out["actions"][idx].cpu().numpy().astype(np.int32)
    elapsed = time() - t0

    # ── use Scheduler to assign arcs → M vehicles (policy is M-agnostic;
    #    Scheduler owns the M partition from the arc order) ─────────────────
    scheduler = Scheduler(variant=args.variant)
    td_single = td[0]  # (1,...) → single-instance td
    vehicles, best_T = scheduler(best_actions, td_single, M=M)

    # Convert Scheduler output (list of trips per vehicle) → route format
    # [0, arc1, arc2, ..., 0] matching print_route_log expectation.
    best_routes = []
    for trips in vehicles:
        route = [0]
        for trip in trips:
            route.extend(int(a) for a in trip)
            route.append(0)
        if len(route) > 2:  # skip empty vehicles
            best_routes.append(np.array(route, dtype=np.int32))

    # ── load instance into ILS solver (for adj / demands used in log) ──────
    ils = InsertCheapestHCARP()
    ils.import_instance(args.file, M=M)

    # ── print & save ───────────────────────────────────────────────────────
    req = es["req"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Instance  : {args.file}")
        print(f"Checkpoint: {args.ckpt}")
        print(
            f"P={P} classes | M={M} vehicles | "
            f"{len(req)} required arcs | {args.num_sample} samples | "
            f"solved in {elapsed:.3f}s\n"
        )
        print_route_log(best_routes, req, best_T, ils)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w") as fh:
        fh.write(text)
    print(text)
    print(f"Saved -> {args.log}")

    sol_path = args.solution or (os.path.splitext(args.log)[0] + ".sol")
    with open(sol_path, "w") as fh:
        fh.write(f"instance: {args.file}\n")
        fh.write(f"checkpoint: {args.ckpt}\n")
        fh.write(f"variant: {args.variant}\n")
        fh.write(f"vehicles: {M}\n")
        fh.write(f"T1: {best_T[0]:.6f}  T2: {best_T[1]:.6f}  T3: {best_T[2]:.6f}\n")
        for i, r in enumerate(best_routes):
            fh.write(f"route {i + 1}: {' '.join(str(a) for a in r)}\n")
    print(f"Saved -> {sol_path}")


if __name__ == "__main__":
    main()
