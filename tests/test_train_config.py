"""D2 Phase 4: train config — --algo/--group_size/--reward_mode wiring + GRPO
per-objective metrics. Parses args and builds models; does NOT run a full train.

Run: uv run python -m unittest tests.test_train_config -v
"""
import os
import sys
import unittest
from unittest import mock

import torch

import importlib.util, pathlib
_src = pathlib.Path(__file__).parent.parent / "src" / "train.py"
_spec = importlib.util.spec_from_file_location("train", _src)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

from trainers.grpo import GRPO


def _parse(argv):
    with mock.patch.object(sys, "argv", ["train.py"] + argv):
        return train.parse_args()


def _auto_reward_mode(args):
    return args.reward_mode


def _make_env(reward_mode, variant="P", M=3, **_):
    from env.env import CARPEnv
    return CARPEnv(num_vehicle=M, variant=variant, reward_mode=reward_mode)


def _make_policy():
    from policy.policy import AttentionModelPolicy
    return AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)


def _make_model(cls, env, policy, tag, metrics, **extra):
    return cls(env, policy,
               path_train_data=f"data/_tc_{tag}_train",
               path_val_data=f"data/_tc_{tag}_val",
               path_test_data=f"data/_tc_{tag}_test",
               batch_size=16, mini_batch_size=8,
               metrics=metrics, **extra)


def _synthetic_dl(n=15, M=3, n_samples=16):
    from env.generator import generate_dataset
    from torch.utils.data import DataLoader
    td = generate_dataset(n_samples, n, n, M, num_workers=0)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(td)
        def __getitem__(self, i): return td[i:i+1]

    return DataLoader(_DS(), batch_size=n_samples, collate_fn=lambda b: torch.cat(b, dim=0))


def _fit_one_step(model, tag, n=15):
    from lightning import Trainer
    M = model.env.num_vehicle
    dl = _synthetic_dl(n, M)
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                      limit_train_batches=1, limit_val_batches=0,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      logger=False, enable_progress_bar=False)
    with mock.patch.object(model.env, 'dataset', side_effect=lambda *a, **kw: dl):
        trainer.fit(model)


GRPO_METRICS = {"train": ["reward", "loss", "T1_mean", "T2_mean", "T3_mean"]}


class TestParseArgs(unittest.TestCase):
    def test_group_size(self):
        a = _parse(["--group_size", "16"])
        self.assertEqual(a.group_size, 16)

    def test_defaults(self):
        a = _parse([])
        self.assertEqual(a.group_size, 8)
        self.assertEqual(a.reward_mode, "vector")

    def test_auto_reward_mode(self):
        self.assertEqual(_auto_reward_mode(_parse([])), "vector")
        # explicit override is respected
        self.assertEqual(
            _auto_reward_mode(_parse(["--reward_mode", "scalar"])),
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


class TestVariantHparam(unittest.TestCase):
    """The env's problem variant must be recorded in hparams (and hence in the
    checkpoint's `hyper_parameters`), because run_grid.py rebuilds the env from
    hp["variant"]. Regression: `save_hyperparameters(ignore=["env", ...])`
    silently dropped it."""

    def test_grpo_records_env_variant(self):
        for variant in ("P", "U"):
            env = _make_env("vector", variant=variant)
            model = _make_model(GRPO, env, _make_policy(), f"var{variant}",
                                GRPO_METRICS, group_size=8)
            self.assertEqual(model.hparams["variant"], variant)

    def test_variant_persists_into_checkpoint(self):
        """End-to-end: hparams["variant"] lands in ckpt["hyper_parameters"],
        the exact dict run_grid.py's load path reads."""
        import tempfile
        torch.manual_seed(0)
        env = _make_env("vector", variant="U")
        model = _make_model(GRPO, env, _make_policy(), "varckpt",
                            GRPO_METRICS, group_size=8)
        from lightning import Trainer
        dl = _synthetic_dl(15, model.env.num_vehicle)
        trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                          limit_train_batches=1, limit_val_batches=0,
                          num_sanity_val_steps=0, enable_checkpointing=False,
                          logger=False, enable_progress_bar=False)
        with mock.patch.object(model.env, "dataset", side_effect=lambda *a, **kw: dl):
            trainer.fit(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt_variant.ckpt")
            trainer.save_checkpoint(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        self.assertEqual(hp.get("variant"), "U")


if __name__ == "__main__":
    unittest.main()
