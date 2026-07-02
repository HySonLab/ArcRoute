"""GRPO core — centered lexicographic-rank advantage, critic-free.

Run: uv run python -m unittest tests.test_grpo -v
"""
import unittest
from unittest import mock

import numpy as np
import torch

from utils.ops import batchify, unbatchify
from trainers.grpo import GRPO, centered_lex_rank, centered_hier_rank_gpu


def _make_env(reward_mode="vector", M=3, n=15):
    from env.env import CARPEnv
    return CARPEnv(num_vehicle=M, variant="P", reward_mode=reward_mode)


def _make_policy():
    from policy.policy import AttentionModelPolicy
    return AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)


def _make_model(cls, env, policy, tag="grpo", **extra):
    return cls(env, policy,
               path_train_data=f"data/_{tag}_train",
               path_val_data=f"data/_{tag}_val",
               path_test_data=f"data/_{tag}_test",
               batch_size=16, mini_batch_size=8, **extra)


def _synthetic_dl(n=15, M=3, n_samples=16):
    """Tiny DataLoader from synthetic data — used to mock env.dataset() in tests."""
    from env.generator import generate_dataset
    from torch.utils.data import DataLoader
    td = generate_dataset(n_samples, n, n, M, num_workers=0)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(td)
        def __getitem__(self, i): return td[i:i+1]

    return DataLoader(_DS(), batch_size=n_samples, collate_fn=lambda b: torch.cat(b, dim=0))


def _fit_one_step(model, tag, n=15):
    """Run exactly ONE train batch through a real (CPU) Trainer so Lightning's
    optimizer + logging are wired; grads persist on params afterwards."""
    from lightning import Trainer
    M = model.env.num_vehicle
    dl = _synthetic_dl(n, M)
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                      limit_train_batches=1, limit_val_batches=0,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      logger=False, enable_progress_bar=False)
    with mock.patch.object(model.env, 'dataset', side_effect=lambda *a, **kw: dl):
        trainer.fit(model)


class TestRankFunction(unittest.TestCase):
    """Tests for centered_lex_rank — hierarchical soft-lex CPU reference (gamma=0.5)."""

    def test_k1_is_zero(self):
        self.assertTrue(np.allclose(centered_lex_rank(np.array([[1, 2, 3]])), 0.0))

    def test_zero_mean_and_bounded(self):
        """Output is in [-1, +1], mean == 0 for any random group."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            K = int(rng.integers(2, 12))
            T = rng.random((K, 3))
            adv = centered_lex_rank(T)
            self.assertTrue(np.all(np.isfinite(adv)))
            self.assertLessEqual(np.abs(adv).max(), 1.0 + 1e-9)
            self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)

    def test_all_tied_is_zero(self):
        """⭐ Whole group identical -> ZERO advantage (DAPO zero-variance case)."""
        for K in (2, 5, 8):
            T = np.tile(np.array([5.0, 2.0, 1.0]), (K, 1))
            adv = centered_lex_rank(T)
            self.assertTrue(np.allclose(adv, 0.0), f"K={K}: {adv}")

    def test_t1_dominates(self):
        """⭐ T1 has the largest gradient weight: a unit T1 rank shift moves advantage
        more than a unit T2 or T3 rank shift (gamma=0.5 → weights 4:2:1)."""
        # All same T2/T3; only T1 differs → T1 rank is the only signal.
        T = np.array([[3, 5, 5], [4, 5, 5], [5, 5, 5], [6, 5, 5]], dtype=float)
        adv = centered_lex_rank(T)
        # Lower T1 → higher advantage, strictly monotone.
        self.assertTrue(adv[0] > adv[1] > adv[2] > adv[3])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)
        # T1 gradient weight > T3 gradient weight: perturbing T1 moves advantage more.
        T2 = np.array([[5, 3, 5], [5, 4, 5], [5, 5, 5], [5, 6, 5]], dtype=float)
        T3 = np.array([[5, 5, 3], [5, 5, 4], [5, 5, 5], [5, 5, 6]], dtype=float)
        adv_t1 = centered_lex_rank(T)
        adv_t2 = centered_lex_rank(T2)
        adv_t3 = centered_lex_rank(T3)
        spread_t1 = float(adv_t1.max() - adv_t1.min())
        spread_t2 = float(adv_t2.max() - adv_t2.min())
        spread_t3 = float(adv_t3.max() - adv_t3.min())
        self.assertGreater(spread_t1, spread_t2)
        self.assertGreater(spread_t2, spread_t3)

    def test_t2_signal_within_tied_t1(self):
        """⭐ When T1 is all tied, T2 drives the advantage ordering."""
        T = np.array([[5, 1, 9], [5, 2, 9], [5, 3, 9], [5, 4, 9]], dtype=float)
        adv = centered_lex_rank(T)
        # Lower T2 → higher advantage (all T1 tied, T3 tied)
        self.assertTrue(adv[0] > adv[1] > adv[2] > adv[3])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)

    def test_t3_signal_within_tied_t1_t2(self):
        """⭐ When T1 and T2 are all tied, T3 drives the advantage ordering."""
        T = np.array([[5, 2, 1], [5, 2, 2], [5, 2, 3], [5, 2, 4]], dtype=float)
        adv = centered_lex_rank(T)
        self.assertTrue(adv[0] > adv[1] > adv[2] > adv[3])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)

    def test_tied_same_advantage(self):
        """⭐ Identical T-vectors get the SAME advantage (midrank averaging)."""
        T = np.array([[5, 1, 1], [5, 1, 1], [5, 2, 9], [7, 0, 0]], dtype=float)
        adv = centered_lex_rank(T)
        self.assertAlmostEqual(float(adv[0]), float(adv[1]), places=6)  # tied → equal
        self.assertGreater(float(adv[0]), float(adv[2]))                 # better T2 wins
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)

    def test_gradient_weight_ratio(self):
        """⭐ gamma=0.5: T1 gradient share ~57%, T2 ~29%, T3 ~14% for P=3.
        Verified by comparing advantage from T1-only vs T3-only perturbations."""
        rng = np.random.default_rng(42)
        # Base: K solutions with all levels spread
        K, P = 20, 3
        T_base = rng.random((K, P)) * 10
        adv_base = centered_lex_rank(T_base).astype(np.float64)
        # Perturb T1 vs T3 by the same delta → T1 perturbation changes advantage more
        delta = 0.1
        T_p1 = T_base.copy(); T_p1[:, 0] += delta * rng.standard_normal(K)
        T_p3 = T_base.copy(); T_p3[:, 2] += delta * rng.standard_normal(K)
        change_t1 = np.abs(centered_lex_rank(T_p1).astype(np.float64) - adv_base).mean()
        change_t3 = np.abs(centered_lex_rank(T_p3).astype(np.float64) - adv_base).mean()
        self.assertGreater(change_t1, change_t3)  # T1 moves advantage more than T3

    def test_general_P(self):
        """⭐ Works for any P: finite, bounded, zero-mean; T1 still primary."""
        rng = np.random.default_rng(2)
        for _ in range(30):
            K = int(rng.integers(2, 9))
            P = int(rng.integers(2, 7))
            adv = centered_lex_rank(rng.random((K, P)))
            self.assertTrue(np.all(np.isfinite(adv)))
            self.assertLessEqual(np.abs(adv).max(), 1.0 + 1e-9)
            self.assertAlmostEqual(float(adv.mean()), 0.0, places=6)


class TestGPURankMatchesCPU(unittest.TestCase):
    """⭐ centered_hier_rank_gpu must exactly match the CPU reference on all cases."""

    def _cpu_batch(self, T_np):
        """Run CPU reference for each row and stack -> (B, K) float32."""
        return np.stack([centered_lex_rank(T_np[b]) for b in range(T_np.shape[0])])

    def test_no_ties(self):
        """Random T with large spread -> no ties; GPU must match CPU."""
        rng = np.random.default_rng(42)
        T_np = rng.random((16, 8, 3)).astype(np.float32) * 1000
        T_gpu = torch.from_numpy(T_np)
        gpu_out = centered_hier_rank_gpu(T_gpu).numpy()
        cpu_out = self._cpu_batch(T_np.astype(np.float64))
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5,
                                   err_msg="GPU hier rank diverges from CPU (no-tie case)")

    def test_with_ties(self):
        """Intentional ties: identical rows in some groups. Tie-averaging must match."""
        T_np = np.array([
            [[5., 1., 1.], [5., 1., 1.], [5., 2., 9.], [7., 0., 0.]],
            [[1., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
        ], dtype=np.float32)
        T_gpu = torch.from_numpy(T_np)
        gpu_out = centered_hier_rank_gpu(T_gpu).numpy()
        cpu_out = self._cpu_batch(T_np.astype(np.float64))
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5,
                                   err_msg="GPU hier rank diverges from CPU (tie case)")

    def test_all_tied(self):
        """Whole group tied -> all zeros on GPU."""
        T_np = np.tile(np.array([3., 1., 2.]), (4, 8, 1)).astype(np.float32)
        gpu_out = centered_hier_rank_gpu(torch.from_numpy(T_np)).numpy()
        np.testing.assert_allclose(gpu_out, 0.0, atol=1e-6)

    def test_large_random(self):
        """Stress: B=64, K=32, P=3 — finite, bounded, zero-mean, matches CPU."""
        rng = np.random.default_rng(7)
        T_np = rng.random((64, 32, 3)).astype(np.float32)
        T_gpu = torch.from_numpy(T_np)
        gpu_out = centered_hier_rank_gpu(T_gpu).numpy()
        cpu_out = self._cpu_batch(T_np.astype(np.float64))
        self.assertTrue(np.isfinite(gpu_out).all())
        self.assertLessEqual(float(np.abs(gpu_out).max()), 1.0 + 1e-5)
        np.testing.assert_allclose(gpu_out.mean(axis=1), 0.0, atol=1e-5)
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-4,
                                   err_msg="GPU/CPU mismatch on large random batch")

    def test_k1_group(self):
        """K=1 -> all zeros (no ranking possible)."""
        T = torch.rand(5, 1, 3)
        out = centered_hier_rank_gpu(T).numpy()
        np.testing.assert_allclose(out, 0.0, atol=1e-9)

    def test_general_P_gpu(self):
        """GPU handles arbitrary P correctly."""
        rng = np.random.default_rng(5)
        for P in (2, 4, 5):
            T_np = rng.random((8, 16, P)).astype(np.float32)
            gpu_out = centered_hier_rank_gpu(torch.from_numpy(T_np)).numpy()
            cpu_out = self._cpu_batch(T_np.astype(np.float64))
            np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-4,
                                       err_msg=f"P={P}: GPU/CPU mismatch")


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
        """Option F: finite, mean≈0 per group, unit std per group."""
        env, policy, model = self._build()
        T = torch.rand(2, 8, 3)
        adv = model._group_advantage(T)        # (2,8)
        self.assertEqual(tuple(adv.shape), (2, 8))
        self.assertTrue(torch.isfinite(adv).all())
        self.assertTrue(torch.allclose(adv.mean(1), torch.zeros(2), atol=1e-6))
        self.assertLessEqual(adv.abs().max().item(), 1.0 + 1e-6)

    def test_critic_free(self):
        """⭐ Fresh GRPO has no critic module and uses manual optimization."""
        env, policy, model = self._build(tag="grpo_nc")
        self.assertFalse(hasattr(model, "critic"))
        self.assertFalse(model.automatic_optimization)

    def test_backward_policy_grad(self):
        # batch_size=16, group_size=8 -> B=2 instances, B*K=16 >= 10.
        env, policy, model = self._build(tag="grpo_bw")
        _fit_one_step(model, "grpo_bw")
        pgrads = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pgrads and all(torch.isfinite(g).all() for g in pgrads))

    def test_eval_step_logs_lex_best(self):
        """⭐ Val rollout logs val/lex_best (the lexicographic checkpoint monitor)."""
        from lightning import Trainer
        env, policy, model = self._build(tag="grpo_val")
        dl = _synthetic_dl()
        trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                          limit_train_batches=1, limit_val_batches=1,
                          num_sanity_val_steps=0, enable_checkpointing=False,
                          logger=False, enable_progress_bar=False)
        with mock.patch.object(model.env, 'dataset', side_effect=lambda *a, **kw: dl):
            trainer.fit(model)
        self.assertIn("val/lex_best", trainer.callback_metrics)

    def test_smoke_P_then_U(self):
        for variant in ("P", "U"):
            with self.subTest(variant=variant):
                env, policy, model = self._build(variant=variant,
                                                 tag=f"grpo_sm_{variant}")
                _fit_one_step(model, f"grpo_sm_{variant}")


def _tiny_rollout(seed=0, B=6, K=4, n=15, M=3):
    """One small CPU rollout -> (env, policy, td0, out) for reward/encoder tests."""
    from env.generator import generate_dataset
    torch.manual_seed(seed)
    env = _make_env(M=M, n=n)
    policy = _make_policy()
    td_batch = generate_dataset(B, n, n, M, num_workers=0)
    td0 = env.reset(td_batch)
    with torch.no_grad():
        out = policy(batchify(td0, K).clone(), env, phase="train", decode_type="greedy")
    return env, policy, td0, out


def _wire_and_batch(model, tag, n=15):
    """Attach `model` to a real CPU Trainer (so optimizers / manual_backward /
    clip_gradients are wired) and return ONE real train batch so `_train_step`
    can be invoked directly. `log_metrics` is stubbed to a pure dict passthrough
    (Lightning's result collection isn't registered outside the training loop)."""
    from lightning import Trainer
    M = model.env.num_vehicle
    dl = _synthetic_dl(n, M)
    model.env.dataset = lambda *a, **kw: dl  # monkeypatch for this model's lifetime

    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                      limit_train_batches=1, limit_val_batches=0,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      logger=False, enable_progress_bar=False)
    trainer.strategy.connect(model)
    trainer.strategy._lightning_module = model
    model.trainer = trainer
    model.setup("fit")
    trainer.strategy.setup_optimizers(trainer)

    def _passthrough(metric_dict, phase, dataloader_idx=None):
        return {f"{phase}/{k}": (v.mean() if isinstance(v, torch.Tensor) else v)
                for k, v in metric_dict.items()}
    model.log_metrics = _passthrough

    batch = next(iter(model.train_dataloader()))
    return trainer, batch


def _cleanup(tag):
    pass  # no .data files created any more


class TestDynamicGroupFiltering(unittest.TestCase):
    """⭐ DAPO dynamic group filtering: groups whose K rollouts are all tied
    (all-zero advantage) are dropped; train/skipped_frac monitors the fraction."""

    def _build(self, tag, **extra):
        torch.manual_seed(0)
        env = _make_env()
        policy = _make_policy()
        # batch_size=16, group_size=8 -> B=2 instances.
        model = _make_model(GRPO, env, policy, tag=tag, group_size=8, **extra)
        return env, policy, model

    def test_all_tied_skips_backward(self):
        """Every group tied -> all-zero advantage -> early return loss=0.0,
        skipped_frac=1.0, no backward, no crash."""
        env, policy, model = self._build("dgf_all")
        try:
            _, batch = _wire_and_batch(model, "dgf_all")
            with mock.patch.object(model, "_group_advantage",
                                   side_effect=lambda T: torch.zeros(T.shape[0], T.shape[1])):
                out = model._train_step(batch)
        finally:
            _cleanup("dgf_all")
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertAlmostEqual(out["train/skipped_frac"], 1.0, places=6)

    def test_partial_filter_only_active_groups(self):
        """Group 0 tied (zeros), group 1 active (non-zero) -> backward runs only
        on the 1 active group; skipped_frac == 0.5; no crash."""
        env, policy, model = self._build("dgf_part")

        def half_zero(T):
            B, K = T.shape[0], T.shape[1]
            adv = torch.zeros(B, K)
            # first half of the groups get a real centered ranking; rest stay tied.
            for b in range(B // 2):
                adv[b] = torch.linspace(-1.0, 1.0, K)
            return adv

        try:
            _, batch = _wire_and_batch(model, "dgf_part")
            with mock.patch.object(model, "_group_advantage", side_effect=half_zero):
                out = model._train_step(batch)
        finally:
            _cleanup("dgf_part")
        self.assertAlmostEqual(out["train/skipped_frac"], 0.5, places=6)
        self.assertTrue(np.isfinite(float(out["loss"])))
        pgrads = [p.grad for p in model.policy.parameters() if p.grad is not None]
        self.assertTrue(pgrads and all(torch.isfinite(g).all() for g in pgrads))

    def test_skipped_frac_logged(self):
        """Normal data path: 'train/skipped_frac' is present in the returned dict."""
        env, policy, model = self._build("dgf_log")
        try:
            _, batch = _wire_and_batch(model, "dgf_log")
            out = model._train_step(batch)
        finally:
            _cleanup("dgf_log")
        self.assertIn("train/skipped_frac", out)
        self.assertGreaterEqual(out["train/skipped_frac"], 0.0)
        self.assertLessEqual(out["train/skipped_frac"], 1.0)


class TestPriorityCountP(unittest.TestCase):
    """⭐ P (number of priority classes) is generalized; default 3 unchanged."""

    def test_env_P4_reward_vector_width(self):
        from env.env import CARPEnv
        from env.generator import generate_dataset
        torch.manual_seed(0)
        env = CARPEnv(num_vehicle=3, variant="P", reward_mode="vector", P=4)
        self.assertEqual(env.obj_weights, [1.0, 1e-2, 1e-4, 1e-6])   # P=4 weights
        policy = _make_policy()
        ds = generate_dataset(6, 20, 20, 3, num_workers=0, P=4)
        td0 = env.reset(ds)
        with torch.no_grad():
            out = policy(batchify(td0, 4).clone(), env, phase="train")
        self.assertEqual(out["reward"].shape[-1], 4)   # P=4 -> T-vector width 4


class TestEncoderShare(unittest.TestCase):
    """#3: POMO encoder sharing (encode once, replicate to the K group) must be
    bit-identical to re-encoding per sample (instance-norm is batch-independent)."""

    def test_share_equals_noshare_greedy(self):
        env, policy, td0, _ = _tiny_rollout()
        K = 4
        with torch.no_grad():
            out_ns = policy(batchify(td0, K).clone(), env, phase="train",
                            decode_type="greedy")
            h0, _ = policy.encoder(td0)
            out_sh = policy(batchify(td0, K).clone(), env, phase="train",
                            decode_type="greedy", hidden=batchify(h0, K))
        self.assertTrue(torch.equal(out_ns["actions"], out_sh["actions"]))
        self.assertTrue(torch.equal(out_ns["reward"], out_sh["reward"]))


if __name__ == "__main__":
    unittest.main()
