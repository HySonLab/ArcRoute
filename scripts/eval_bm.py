"""Compare GRPO vs PPO checkpoints on the same held-out test set.

Called by bm_trainer.sh after both models finish training. Reports mean T1,
T2, T3 (lower = better) and the lex scalar under the same vector-reward env.

Usage
-----
  uv run python scripts/eval_bm.py \\
      --grpo_ckpt outputs/bm/ckpt/grpo/best.ckpt \\
      --ppo_ckpt  outputs/bm/ckpt/ppo/best.ckpt  \\
      [--num_loc 40] [--num_arc 80] [--fleet 3]   \\
      [--embed_dim 128] [--num_encoder_layers 6] [--num_heads 8] \\
      [--num_test 512] [--batch_size 128]
"""

import argparse
import numpy as np
import torch

from env.env import CARPEnv
from policy.policy import AttentionModelPolicy


_LEX_C = 1.0e3


def _lex_scalar(T):
    """(N, P) -> (N,) Horner lex scalar; lower T is better, higher scalar is better."""
    s = -T[:, 0].copy()
    for j in range(1, T.shape[1]):
        s = s * _LEX_C + (-T[:, j])
    return s


def _load_policy(ckpt_path, policy):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[len("policy."):]: v
          for k, v in ckpt["state_dict"].items()
          if k.startswith("policy.")}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
    return policy


@torch.no_grad()
def _eval(policy, env, num_test, batch_size, device):
    policy.eval().to(device)
    ds = env.dataset(num_test, batch_size=batch_size, shuffle=False, num_workers=0,
                     data="data/eval_bm_tmp.data")
    T_chunks = []
    for batch in ds:
        td = env.reset(batch)
        td = td.to(device)
        out = policy(td.clone(), env, phase="test")
        T_chunks.append(out["reward"].cpu().numpy())   # (B, P)
    return np.concatenate(T_chunks, axis=0)            # (N, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpo_ckpt",           required=True, help="GRPO checkpoint")
    ap.add_argument("--ppo_ckpt",            required=True, help="PPO checkpoint")
    ap.add_argument("--num_loc",            type=int, default=40)
    ap.add_argument("--num_arc",            type=int, default=80)
    ap.add_argument("--fleet",              type=int, default=3)
    ap.add_argument("--embed_dim",          type=int, default=128)
    ap.add_argument("--num_encoder_layers", type=int, default=6)
    ap.add_argument("--num_heads",          type=int, default=8)
    ap.add_argument("--num_test",           type=int, default=512)
    ap.add_argument("--batch_size",         type=int, default=128)
    ap.add_argument("--variant",            type=str, default="P")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = CARPEnv(
        num_loc=args.num_loc,
        num_arc=args.num_arc,
        num_vehicle=args.fleet,
        variant=args.variant,
        reward_mode="vector",
    )

    candidates = [("GRPO", args.grpo_ckpt), ("PPO", args.ppo_ckpt)]

    results = {}
    for name, ckpt_path in candidates:
        print(f"\nEvaluating {name} from {ckpt_path} ...")
        policy = AttentionModelPolicy(
            embed_dim=args.embed_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_heads=args.num_heads,
        )
        _load_policy(ckpt_path, policy)
        T = _eval(policy, env, args.num_test, args.batch_size, device)
        results[name] = T
        P = T.shape[1]
        print("  " + "  ".join(f"T{j+1}={T[:,j].mean():.4f}" for j in range(P)))

    # ── report ────────────────────────────────────────────────────────────────
    T_grpo, T_ppo = results["GRPO"], results["PPO"]
    P = T_grpo.shape[1]
    width = 62
    print()
    print("=" * width)
    print(f"  GRPO vs PPO — {args.num_test} instances "
          f"({args.num_loc}×{args.num_arc}, M={args.fleet})")
    print("=" * width)
    header = f"{'':>8}" + "".join(f"{'T'+str(j+1):>10s}" for j in range(P)) + f"{'lex':>12s}"
    print(header)

    def fmt_row(name, T):
        lex = _lex_scalar(T).mean()
        vals = "".join(f"{T[:,j].mean():10.4f}" for j in range(P))
        return f"{name:>8}{vals}{lex:12.2f}"

    for name in ["GRPO", "PPO"]:
        print(fmt_row(name, results[name]))
    print("-" * width)

    print(f"\nPer-objective delta (GRPO vs PPO, negative = GRPO better):")
    wins = 0
    for j in range(P):
        g, p = T_grpo[:, j].mean(), T_ppo[:, j].mean()
        delta_pct = (g - p) / (p + 1e-9) * 100
        winner = "GRPO" if g < p else "PPO"
        print(f"  T{j+1}: GRPO={g:.4f}  PPO={p:.4f}  delta={delta_pct:+.1f}%  → {winner} wins")
        if g < p:
            wins += 1
    lex_g = _lex_scalar(T_grpo).mean()
    lex_p = _lex_scalar(T_ppo).mean()
    print(f"  lex: GRPO={lex_g:.2f}  PPO={lex_p:.2f}  → {'GRPO' if lex_g > lex_p else 'PPO'} wins")
    print(f"\nOverall: {'GRPO' if wins > P // 2 else 'PPO'} ({wins}/{P} objectives)")
    print("=" * width)


if __name__ == "__main__":
    main()
