"""D2 Phase 4: train config — --algo/--group_size/--reward_mode wiring + GRPO
per-objective metrics. Parses args and builds models; does NOT run a full train.

Run: uv run python -m unittest tests.test_train_config -v
"""
import sys
import unittest
from unittest import mock

import torch

import train
from rl.grpo import GRPO
from rl.ppo import PPO


def _parse(argv):
    with mock.patch.object(sys, "argv", ["train.py"] + argv):
        return train.parse_args()


def _auto_reward_mode(args):
    return args.reward_mode or ("vector" if args.algo == "grpo" else "scalar")


def _make_env(reward_mode, variant="P", n=15, M=3):
    from env.env import CARPEnv
    return CARPEnv(num_loc=n, num_arc=n, num_vehicle=M, variant=variant,
                   reward_mode=reward_mode)


def _make_policy():
    from policy.policy import AttentionModelPolicy
    return AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)


def _make_model(cls, env, policy, tag, metrics, **extra):
    return cls(env, policy,
               path_train_data=f"data/_tc_{tag}_train.data",
               path_val_data=f"data/_tc_{tag}_val.data",
               path_test_data=f"data/_tc_{tag}_test.data",
               batch_size=16, mini_batch_size=8,
               train_data_size=16, val_data_size=16, test_data_size=16,
               dataloader_num_workers=0, reload_train_dataloader=999,
               metrics=metrics, **extra)


def _fit_one_step(model, tag):
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
        for f in glob.glob(f"data/_tc_{tag}_*.data"):
            try:
                os.remove(f)
            except OSError:
                pass


GRPO_METRICS = {"train": ["reward", "loss", "surrogate_loss", "entropy",
                          "T1_mean", "T2_mean", "T3_mean"]}
PPO_METRICS = {"train": ["reward", "loss", "surrogate_loss", "value_loss", "entropy"]}


class TestParseArgs(unittest.TestCase):
    def test_algo_and_group_size(self):
        a = _parse(["--algo", "grpo", "--group_size", "16"])
        self.assertEqual(a.algo, "grpo")
        self.assertEqual(a.group_size, 16)

    def test_defaults(self):
        a = _parse([])
        self.assertEqual(a.algo, "ppo")
        self.assertEqual(a.group_size, 8)
        self.assertIsNone(a.reward_mode)

    def test_auto_reward_mode(self):
        self.assertEqual(_auto_reward_mode(_parse(["--algo", "grpo"])), "vector")
        self.assertEqual(_auto_reward_mode(_parse(["--algo", "ppo"])), "scalar")
        # explicit override is respected
        self.assertEqual(
            _auto_reward_mode(_parse(["--algo", "grpo", "--reward_mode", "scalar"])),
            "scalar")


class TestBuildAndStep(unittest.TestCase):
    def test_grpo_build_and_step(self):
        """⭐ algo=grpo path -> CARPEnv(vector) + GRPO(group_size=8); one train
        step finite (B=2,K=8 -> 16 >= 10)."""
        torch.manual_seed(0)
        env = _make_env("vector")
        model = _make_model(GRPO, env, _make_policy(), "grpo", GRPO_METRICS,
                            group_size=8)
        _fit_one_step(model, "grpo")
        pg = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pg and all(torch.isfinite(g).all() for g in pg))

    def test_grpo_per_objective_metrics(self):
        """⭐ GRPO.shared_step puts finite T1_mean/T2_mean/T3_mean in `out`."""
        torch.manual_seed(0)
        from env.generator import generate_dataset
        env = _make_env("vector")
        model = _make_model(GRPO, env, _make_policy(), "grpom", GRPO_METRICS,
                            group_size=8)
        captured = {}
        orig = model.log_metrics

        def spy(out, phase, dataloader_idx=None):
            if phase == "train":
                captured.update({k: out.get(k) for k in
                                 ("T1_mean", "T2_mean", "T3_mean")})
            return orig(out, phase, dataloader_idx=dataloader_idx)

        model.log_metrics = spy
        _fit_one_step(model, "grpom")
        for k in ("T1_mean", "T2_mean", "T3_mean"):
            self.assertIn(k, captured)
            self.assertIsNotNone(captured[k])
            self.assertTrue(torch.isfinite(captured[k]).all())

    def test_ppo_build_still_works(self):
        """⭐ default ppo path still trains (A/B)."""
        torch.manual_seed(0)
        env = _make_env("scalar")
        model = _make_model(PPO, env, _make_policy(), "ppo", PPO_METRICS)
        _fit_one_step(model, "ppo")
        pg = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pg and all(torch.isfinite(g).all() for g in pg))


if __name__ == "__main__":
    unittest.main()
