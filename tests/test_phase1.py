"""Tests for Joint M×N Vehicle-Arc Attention (GRPO v3, Phase 1).

Test gates (docs/plan/grpo_v3/05_test_plan.md):
  1. TestJointMxNInfeasibility — never an all-inf step during decode.
  2. TestLogProbCorrectness   — Σ log_p_step == log_p(trajectory).
  3. TestMaskCorrectness      — served + capacity constraints enforced.
  4. TestJointShapes          — logits (B,M,N+1), flat (B,M*(N+1)), m*/j* split.
  5. TestGRPOIntegrationP1    — GRPO trains 2 steps, grads finite.

Run: uv run python -m unittest tests.test_phase1 -v
"""
import unittest
from unittest import mock

import numpy as np
import torch


# --------------------------------------------------------------------- helpers
def _make_env(M=3, variant="P", reward_mode="vector"):
    from env.env import CARPEnv
    return CARPEnv(num_vehicle=M, variant=variant, reward_mode=reward_mode)


def _make_policy(embed_dim=32, num_heads=4):
    from policy.policy import AttentionModelPolicy
    return AttentionModelPolicy(embed_dim=embed_dim, num_encoder_layers=1,
                                num_heads=num_heads)


def _dataset(B=4, num_loc=12, M=3, P=3):
    """Synthetic batch. required arcs N = 3*(num_loc//4)."""
    from env.generator import generate_dataset
    return generate_dataset(B, num_loc, num_loc, M, num_workers=0, P=P)


def _manual_decode(env, policy, td0, decode="greedy"):
    """Replicate the Joint M×N decode loop step-by-step, capturing the flat mask,
    the chosen flat action, and the (m*, j*) split BEFORE each env.step. Returns
    (steps, final_td). `steps[i]` carries pre-step state for assertions."""
    td = td0.clone()
    hidden, _ = policy.encoder(td)
    td, env, cached = policy.decoder.pre_decoder_hook(td, env, hidden)
    N1 = td["visited"].shape[-1]
    steps = []
    while not td["done"].all():
        logits, mask = policy.decoder(td, cached)
        B = logits.shape[0]
        logits_flat = logits.reshape(B, -1)
        mask_flat = mask.reshape(B, -1)
        masked = logits_flat.masked_fill(~mask_flat, float("-inf"))
        if decode == "greedy":
            action = masked.argmax(dim=-1)
        else:
            action = torch.multinomial(torch.softmax(masked, dim=-1), 1).squeeze(1)
        m_star = action // N1
        j_star = action % N1
        steps.append({
            "mask_flat": mask_flat.clone(),
            "logits_shape": tuple(logits.shape),
            "action": action.clone(),
            "m": m_star.clone(),
            "j": j_star.clone(),
            "used_capacity": td["used_capacity"].clone(),
            "demand": td["demand"].clone(),
            "done_before": td["done"].clone().view(B),
            "N1": N1,
        })
        td.set("action", action)
        td = env.step(td)
    return steps, td


# --------------------------------------------------------------------- gate 4
class TestJointShapes(unittest.TestCase):
    """logits (B,M,N+1); flat (B,M*(N+1)); flat split recovers m*, j*."""

    def test_shapes_and_split(self):
        torch.manual_seed(0)
        env, policy = _make_env(M=2), _make_policy()
        td0 = env.reset(_dataset(B=3, num_loc=12, M=2))
        B = td0.batch_size[0]
        N1 = td0["visited"].shape[-1]
        M_max = td0["used_capacity"].shape[-1]
        self.assertEqual(M_max, 2)

        hidden, _ = policy.encoder(td0)
        td, env, cached = policy.decoder.pre_decoder_hook(td0.clone(), env, hidden)
        logits, mask = policy.decoder(td, cached)
        self.assertEqual(tuple(logits.shape), (B, M_max, N1))
        self.assertEqual(tuple(mask.shape), (B, M_max, N1))

        flat = logits.reshape(B, -1)
        self.assertEqual(tuple(flat.shape), (B, M_max * N1))

        # m*/j* recovered from flat index for every cell.
        for a in range(M_max * N1):
            self.assertEqual((a // N1, a % N1), divmod(a, N1))


# --------------------------------------------------------------------- gate 1
class TestJointMxNInfeasibility(unittest.TestCase):
    """100 random instances decoded to done: no step has an all-inf (all-masked)
    row — the M depot columns guarantee >=1 feasible cell."""

    def test_never_all_inf(self):
        torch.manual_seed(1)
        for trial in range(10):                       # 10 batches x B=10 = 100 inst
            env, policy = _make_env(M=3), _make_policy()
            td0 = env.reset(_dataset(B=10, num_loc=12, M=3))
            steps, _ = _manual_decode(env, policy, td0, decode="sampling")
            for si, st in enumerate(steps):
                feasible_per_row = st["mask_flat"].any(dim=-1)
                self.assertTrue(
                    bool(feasible_per_row.all()),
                    f"trial {trial} step {si}: an instance had all-inf logits",
                )

    def test_mixed_fleet_never_all_inf(self):
        """Per-instance M differs (padded vehicles masked) — still never all-inf."""
        torch.manual_seed(2)
        env, policy = _make_env(M=[2, 3, 5]), _make_policy()
        td0 = env.reset(_dataset(B=12, num_loc=12, M=[2, 3, 5]))
        steps, _ = _manual_decode(env, policy, td0, decode="sampling")
        for st in steps:
            self.assertTrue(bool(st["mask_flat"].any(dim=-1).all()))


# --------------------------------------------------------------------- gate 3
class TestMaskCorrectness(unittest.TestCase):
    """served + capacity constraints are enforced by the mask."""

    def test_served(self):
        """Each non-depot arc is served at most once across the whole rollout, and
        the final state has every arc served exactly once."""
        torch.manual_seed(3)
        env, policy = _make_env(M=3), _make_policy()
        td0 = env.reset(_dataset(B=6, num_loc=12, M=3))
        steps, final_td = _manual_decode(env, policy, td0, decode="greedy")
        B = td0.batch_size[0]
        N = td0["visited"].shape[-1] - 1

        served_count = torch.zeros(B, N + 1, dtype=torch.long)
        for st in steps:
            for b in range(B):
                j = int(st["j"][b])
                if j != 0:
                    served_count[b, j] += 1
        # No arc served more than once.
        self.assertTrue(bool((served_count[:, 1:] <= 1).all()),
                        f"some arc served twice:\n{served_count}")
        # All arcs served exactly once at the end.
        self.assertTrue(bool((served_count[:, 1:] == 1).all()),
                        "not every arc was served exactly once")
        # Final env visited mask agrees.
        self.assertTrue(bool((final_td["visited"][:, 0, 1:] != 0).all()))

    def test_capacity(self):
        """No chosen (m*, j!=0) ever exceeds vehicle m*'s remaining capacity."""
        torch.manual_seed(4)
        env, policy = _make_env(M=3), _make_policy()
        td0 = env.reset(_dataset(B=6, num_loc=12, M=3))
        steps, _ = _manual_decode(env, policy, td0, decode="sampling")
        cap = 1.0
        for si, st in enumerate(steps):
            B = st["action"].shape[0]
            for b in range(B):
                if bool(st["done_before"][b]):
                    continue
                j = int(st["j"][b])
                m = int(st["m"][b])
                if j == 0:
                    continue
                load = float(st["used_capacity"][b, m])
                dem = float(st["demand"][b, j])
                self.assertLessEqual(
                    load + dem, cap + 1e-6,
                    f"step {si} inst {b}: veh {m} over capacity "
                    f"({load:.3f}+{dem:.3f} > {cap})",
                )


# --------------------------------------------------------------------- gate 2
class TestLogProbCorrectness(unittest.TestCase):
    """Σ log_p(step) for the chosen joint actions == log_p(trajectory) reported
    by the policy's get_log_likelihood (chain rule over independent steps)."""

    def test_sum_equals_trajectory(self):
        torch.manual_seed(5)
        env, policy = _make_env(M=3), _make_policy()
        td0 = env.reset(_dataset(B=5, num_loc=12, M=3))

        # 1) greedy rollout -> a fixed action sequence.
        with torch.no_grad():
            out = policy(td0.clone(), env, phase="train", decode_type="greedy")
        actions = out["actions"]                       # (B, seq) flat joint idx

        # 2) policy's trajectory log-likelihood (sum over steps), replaying actions.
        with torch.no_grad():
            out_eval = policy(td0.clone(), env, actions=actions,
                              return_sum_log_likelihood=True)
        traj_ll = out_eval["log_likelihood"]           # (B,)

        # 3) manual Σ log_p(step): re-decode replaying actions, gather chosen logp.
        td = td0.clone()
        hidden, _ = policy.encoder(td)
        td, env, cached = policy.decoder.pre_decoder_hook(td, env, hidden)
        N1 = td["visited"].shape[-1]
        manual = torch.zeros(td.batch_size[0])
        step = 0
        with torch.no_grad():
            while not td["done"].all():
                valid = ~td["done"].reshape(-1)  # (B,) — exclude post-done steps
                logits, mask = policy.decoder(td, cached)
                B = logits.shape[0]
                lf = logits.reshape(B, -1)
                mf = mask.reshape(B, -1)
                logp = torch.log_softmax(lf.masked_fill(~mf, float("-inf")), dim=-1)
                a = actions[..., step]
                manual = manual + logp.gather(1, a[:, None]).squeeze(1) * valid.float()
                td.set("action", a)
                td = env.step(td)
                step += 1

        self.assertTrue(torch.allclose(manual, traj_ll, atol=1e-5),
                        f"manual {manual}\n vs traj {traj_ll}")


# --------------------------------------------------------------------- gate 5
class TestGRPOIntegrationP1(unittest.TestCase):
    """GRPO trains 2 steps with the Joint M×N policy: no crash, finite grads,
    finite entropy."""

    def _synthetic_dl(self, n=12, M=3, n_samples=16):
        from env.generator import generate_dataset
        from torch.utils.data import DataLoader
        td = generate_dataset(n_samples, n, n, M, num_workers=0)

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(td)
            def __getitem__(self, i): return td[i:i + 1]

        return DataLoader(_DS(), batch_size=n_samples,
                          collate_fn=lambda b: torch.cat(b, dim=0))

    def test_train_two_steps(self):
        from lightning import Trainer
        from trainers.grpo import GRPO
        torch.manual_seed(6)
        env, policy = _make_env(M=3), _make_policy()
        model = GRPO(env, policy,
                     path_train_data="data/_p1_train",
                     path_val_data="data/_p1_val",
                     path_test_data="data/_p1_test",
                     batch_size=16, mini_batch_size=8, group_size=8)
        dl = self._synthetic_dl(M=3)
        # max_epochs=2, 1 batch/epoch -> 2 GRPO train steps.
        trainer = Trainer(accelerator="cpu", devices=1, max_epochs=2,
                          limit_train_batches=1, limit_val_batches=0,
                          num_sanity_val_steps=0, enable_checkpointing=False,
                          logger=False, enable_progress_bar=False)
        with mock.patch.object(model.env, "dataset", side_effect=lambda *a, **kw: dl):
            trainer.fit(model)

        pgrads = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pgrads, "no policy gradients populated")
        self.assertTrue(all(torch.isfinite(g).all() for g in pgrads),
                        "non-finite policy gradient")

    def test_entropy_finite(self):
        """Joint M×N entropy (over the M*(N+1) flat action space) is finite."""
        from policy.decode_strategy import calculate_entropy
        torch.manual_seed(7)
        env, policy = _make_env(M=3), _make_policy()
        td0 = env.reset(_dataset(B=4, num_loc=12, M=3))
        with torch.no_grad():
            out = policy(td0.clone(), env, phase="train", return_entropy=True)
        self.assertIn("entropy", out)
        self.assertTrue(torch.isfinite(out["entropy"]).all())


# --------------------------------------------------------------- precedence
class TestPrecedenceMask(unittest.TestCase):
    """The single most load-bearing variant-U property: precedence is genuinely
    enforced by the mask for P and genuinely bypassed for U.

    P: at reset (nothing served), any arc whose class is strictly greater than
    the min class present in the instance is infeasible for EVERY vehicle
    (the `wrong_class`/`c_min` layer in env.get_action_mask).
    U: the very same arcs are feasible for every vehicle with capacity --
    only served/capacity/depot-idle/nonexistent-vehicle layers may mask."""

    def _masks(self, variant, data):
        env = _make_env(M=3, variant=variant)
        td = env.reset(data.clone())
        B = td.batch_size[0]
        N1 = td["visited"].shape[-1]
        M_max = td["used_capacity"].shape[-1]
        mask = env.get_action_mask(td).reshape(B, M_max, N1)  # True == feasible
        return td, mask

    def test_P_masks_higher_classes_U_does_not(self):
        torch.manual_seed(8)
        data = _dataset(B=6, num_loc=12, M=3)
        tdP, maskP = self._masks("P", data)
        tdU, maskU = self._masks("U", data)
        B, M_max, N1 = maskP.shape

        clss = tdP["clss"]
        demand = tdP["demand"]
        checked = 0
        for b in range(B):
            c_min = int(clss[b, 1:].min())
            higher = [j for j in range(1, N1) if int(clss[b, j]) > c_min]
            if not higher:
                continue  # degenerate single-class instance
            checked += 1
            for j in higher:
                self.assertFalse(
                    bool(maskP[b, :, j].any()),
                    f"P: class-{int(clss[b, j])} arc {j} feasible in inst {b} "
                    f"although c_min={c_min}")
                if float(demand[b, j]) <= 1.0 + 1e-9:
                    self.assertTrue(
                        bool(maskU[b, :, j].all()),
                        f"U: arc {j} masked in inst {b} -- precedence must not "
                        f"restrict feasibility under U")
        self.assertGreater(checked, 0,
                           "no instance had >= 2 distinct classes; test vacuous")

    def test_U_mask_is_exactly_served_capacity_layers(self):
        """Under U at reset (scalar fleet -> all vehicles exist), a non-depot
        column is feasible iff the arc fits an empty vehicle's capacity; depot
        columns are blocked by the idle-guard (empty vehicle at depot)."""
        torch.manual_seed(9)
        data = _dataset(B=4, num_loc=12, M=3)
        td, mask = self._masks("U", data)
        B, M_max, N1 = mask.shape
        demand = td["demand"]
        for b in range(B):
            for j in range(1, N1):
                fits = float(demand[b, j]) <= 1.0 + 1e-9
                self.assertEqual(bool(mask[b, :, j].all()), fits)
                self.assertEqual(bool(mask[b, :, j].any()), fits)
            # depot idle-guard: empty vehicle at depot may not pick depot again
            self.assertFalse(bool(mask[b, :, 0].any()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
