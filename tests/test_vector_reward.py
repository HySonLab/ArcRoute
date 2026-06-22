"""D2 Phase 2: vector reward mode — env.get_reward exposes the (B,3) T-vector.

`reward_mode='scalar'` (default) keeps the old -(T . w) reward byte-identical;
`reward_mode='vector'` returns the raw (B,3) T-vector for the GRPO path (Phase 3).

Run: uv run python -m unittest tests.test_vector_reward -v
"""
import unittest

import numpy as np
import torch

from env.env import CARPEnv
from env.generator import generate_dataset
from policy.policy import AttentionModelPolicy


def _rollout(env, B=12, n=15, M=3, seed=0):
    """env.reset -> greedy policy rollout -> (td, actions). B>=10 so the
    run_parallel(num_epochs=10) split inside get_reward stays non-empty.
    (calc_reward=True so the policy populates td['reward']; we re-evaluate
    get_reward on `env` ourselves to probe the mode.)"""
    torch.manual_seed(seed)
    batch = generate_dataset(B, n, n, M, num_workers=0)
    td = env.reset(batch)
    policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
    out = policy(td.clone(), env, decode_type="greedy")
    return td, out["actions"], policy


class TestVectorReward(unittest.TestCase):
    def test_shape_scalar(self):
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P",
                      reward_mode="scalar")
        td, actions, _ = _rollout(env)
        r = env.get_reward(td, actions)
        self.assertEqual(tuple(r.shape), (12, 1))

    def test_shape_vector(self):
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P",
                      reward_mode="vector")
        td, actions, _ = _rollout(env)
        r = env.get_reward(td, actions)
        self.assertEqual(tuple(r.shape), (12, 3))

    def test_scalar_equals_weighted_vector(self):
        """⭐ scalar reward == -(vector . obj_weights): same meaning, vector just
        exposes more. Same actions/td feed both modes."""
        w = [1.0, 1e-2, 1e-4]
        td, actions, _ = _rollout(
            CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P"))
        env_s = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P",
                        obj_weights=w, reward_mode="scalar")
        env_v = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P",
                        obj_weights=w, reward_mode="vector")
        scalar = env_s.get_reward(td, actions).detach().cpu().numpy().reshape(-1)
        vector = env_v.get_reward(td, actions).detach().cpu().numpy()
        expected = -(vector * np.array(w)).sum(-1)
        np.testing.assert_allclose(scalar, expected, atol=1e-5)

    def test_vector_finite_and_nonneg(self):
        """⭐ every (B,3) entry finite and >= 0 (completion times)."""
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P",
                      reward_mode="vector")
        td, actions, _ = _rollout(env)
        r = env.get_reward(td, actions)
        self.assertTrue(torch.isfinite(r).all())
        self.assertTrue((r >= 0).all())

    def test_default_is_scalar(self):
        """No reward_mode kwarg -> scalar (B,1): safe rollback."""
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant="P")
        self.assertEqual(env.reward_mode, "scalar")
        td, actions, _ = _rollout(env)
        self.assertEqual(tuple(env.get_reward(td, actions).shape), (12, 1))


class TestRolloutSmokeScalar(unittest.TestCase):
    """⭐ The OLD scalar PPO path still works: reset -> policy(calc_reward=True)
    -> reward (B,1) -> backward, finite. Variant P then U."""

    def _run(self, variant):
        torch.manual_seed(0)
        env = CARPEnv(num_loc=15, num_arc=15, num_vehicle=3, variant=variant)
        batch = generate_dataset(12, 15, 15, 3, num_workers=0)
        td = env.reset(batch)
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
        out = policy(td.clone(), env, phase="train")
        reward = out["reward"]
        self.assertEqual(reward.shape[-1], 1)
        self.assertTrue(torch.isfinite(reward).all())
        (-out["log_likelihood"].sum()).backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_rollout_smoke_PU(self):
        for variant in ("P", "U"):
            with self.subTest(variant=variant):
                self._run(variant)


if __name__ == "__main__":
    unittest.main()
