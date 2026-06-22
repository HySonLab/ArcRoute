"""D2 Phase 3: GRPO core (rl/grpo.py) — centered lexicographic-rank advantage,
critic-free, with rl/ppo.py left UNCHANGED.

Run: uv run python -m unittest tests.test_grpo -v
"""
import subprocess
import unittest

import numpy as np
import torch

from common.ops import batchify, unbatchify
from rl.grpo import GRPO, centered_lex_rank
from rl.ppo import PPO


def _make_env(reward_mode="vector", M=3, n=15):
    from env.env import CARPEnv
    return CARPEnv(num_loc=n, num_arc=n, num_vehicle=M, variant="P",
                   reward_mode=reward_mode)


def _make_policy():
    from policy.policy import AttentionModelPolicy
    return AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)


def _make_model(cls, env, policy, tag="grpo", **extra):
    return cls(env, policy,
               path_train_data=f"data/_{tag}_train.data",
               path_val_data=f"data/_{tag}_val.data",
               path_test_data=f"data/_{tag}_test.data",
               batch_size=16, mini_batch_size=8,
               train_data_size=16, val_data_size=16, test_data_size=16,
               dataloader_num_workers=0, reload_train_dataloader=999,
               **extra)


def _fit_one_step(model, tag):
    """Run exactly ONE train batch through a real (CPU) Trainer so Lightning's
    optimizer + logging are wired; grads persist on params afterwards (1 batch,
    not zeroed again). Cleans up the generated tiny data files."""
    import glob
    import os
    from lightning import Trainer
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                      limit_train_batches=1, limit_val_batches=0,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      logger=False, enable_progress_bar=False)
    try:
        trainer.fit(model)
    finally:
        for f in glob.glob(f"data/_{tag}_*.data"):
            try:
                os.remove(f)
            except OSError:
                pass


class TestRankFunction(unittest.TestCase):
    def test_hand_example(self):
        """⭐ [[5,9,9],[5,2,9],[5,2,1],[7,0,0]] (K=4): best lex is [5,2,1] (idx 2)
        -> max adv (+1); worst is [7,0,0] (idx 3) -> min adv (-1); mean ~ 0."""
        T = np.array([[5, 9, 9], [5, 2, 9], [5, 2, 1], [7, 0, 0]], dtype=float)
        adv = centered_lex_rank(T)
        self.assertEqual(int(np.argmax(adv)), 2)      # [5,2,1] best
        self.assertEqual(int(np.argmin(adv)), 3)      # [7,0,0] worst
        self.assertAlmostEqual(float(adv[2]), 1.0, places=6)
        self.assertAlmostEqual(float(adv[3]), -1.0, places=6)
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)

    def test_k1_is_zero(self):
        self.assertTrue(np.allclose(centered_lex_rank(np.array([[1, 2, 3]])), 0.0))

    def test_zero_mean_and_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            K = int(rng.integers(2, 12))
            T = rng.random((K, 3))
            adv = centered_lex_rank(T)
            self.assertTrue(np.all(np.isfinite(adv)))
            self.assertLessEqual(np.abs(adv).max(), 1.0 + 1e-9)
            self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)


class TestBatchifyLayout(unittest.TestCase):
    def test_unbatchify_roundtrips_per_instance(self):
        """⭐ unbatchify(batchify(x,K),K) == x broadcast over the K axis: each
        instance's K copies are grouped on the right axis (no instance mixing)."""
        K = 8
        B = 3
        x = torch.arange(B * 3, dtype=torch.float32).reshape(B, 3)
        xb = batchify(x, K)                  # (B*K, 3)
        xu = unbatchify(xb, K)              # (B, K, 3)
        self.assertEqual(tuple(xu.shape), (B, K, 3))
        for b in range(B):
            for k in range(K):
                self.assertTrue(torch.equal(xu[b, k], x[b]))

    def test_advantage_reflatten_matches_td_layout(self):
        """The (B,K) advantage re-flattened with permute(1,0).reshape lands on the
        SAME (B*K,) layout batchify produced (so adv aligns with each rollout)."""
        K, B = 4, 3
        # tag[b] unique per instance; batchify replicates each instance K times.
        tag = torch.arange(B, dtype=torch.float32).reshape(B, 1)
        tag_flat = batchify(tag, K).reshape(-1)            # td layout (B*K,)
        # adv_bk[b,k] = which instance -> reflatten must equal tag_flat.
        adv_bk = torch.arange(B, dtype=torch.float32).reshape(B, 1).expand(B, K).clone()
        reflat = adv_bk.permute(1, 0).reshape(B * K)
        self.assertTrue(torch.equal(reflat, tag_flat))


class TestGRPOStep(unittest.TestCase):
    """⭐ backward gives policy grads but NOT critic grads; B*K >= 10."""

    def _build(self, variant="P", tag="grpo"):
        torch.manual_seed(0)
        env = _make_env(); env.variant = variant
        policy = _make_policy()
        model = _make_model(GRPO, env, policy, tag=tag, group_size=8)
        return env, policy, model

    def test_advantage_finite_zero_mean(self):
        env, policy, model = self._build()
        T = torch.rand(2, 8, 3)
        adv = model._group_advantage(T)        # (2,8)
        self.assertEqual(tuple(adv.shape), (2, 8))
        self.assertTrue(torch.isfinite(adv).all())
        self.assertLessEqual(adv.abs().max().item(), 1.0 + 1e-6)
        self.assertTrue(torch.allclose(adv.mean(1), torch.zeros(2), atol=1e-6))

    def test_backward_policy_grad_no_critic_grad(self):
        # batch_size=16, group_size=8 -> B=2 instances, B*K=16 >= 10.
        env, policy, model = self._build(tag="grpo_bw")
        _fit_one_step(model, "grpo_bw")
        pgrads = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pgrads and all(torch.isfinite(g).all() for g in pgrads))
        # critic excluded from loss + optimizer -> no grad.
        cgrads = [p.grad for p in model.critic.parameters() if p.grad is not None]
        self.assertEqual(len(cgrads), 0)

    def test_smoke_P_then_U(self):
        for variant in ("P", "U"):
            with self.subTest(variant=variant):
                env, policy, model = self._build(variant=variant,
                                                 tag=f"grpo_sm_{variant}")
                _fit_one_step(model, f"grpo_sm_{variant}")


class TestPPOUnchanged(unittest.TestCase):
    def test_ppo_py_unchanged(self):
        """⭐ git diff rl/ppo.py must be empty (PPO untouched)."""
        out = subprocess.run(["git", "diff", "--stat", "--", "rl/ppo.py"],
                             capture_output=True, text=True, cwd=_repo_root())
        self.assertEqual(out.stdout.strip(), "", f"rl/ppo.py changed:\n{out.stdout}")

    def test_ppo_still_trains_critic_gets_grad(self):
        """⭐ The old PPO path still trains and the critic DOES get grad."""
        torch.manual_seed(0)
        env = _make_env(reward_mode="scalar")
        policy = _make_policy()
        model = _make_model(PPO, env, policy, tag="ppo_ck")
        _fit_one_step(model, "ppo_ck")
        cgrads = [p.grad for p in model.critic.parameters() if p.grad is not None]
        self.assertTrue(cgrads and all(torch.isfinite(g).all() for g in cgrads))


def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    unittest.main()
