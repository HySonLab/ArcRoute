"""Unit tests for SizeBucketBatchSampler and multi-fleet generate_dataset.

Run:
    uv run python -m unittest tests.test_multisize -v
"""
import unittest

import torch
from torch.utils.data import DataLoader

from env.generator import SizeBucketBatchSampler, generate_dataset


# Hand-crafted bucket_ranges for sampler tests — no data generation needed.
BUCKET_RANGES = [(0, 20), (20, 40), (40, 60)]   # 3 buckets of 20 items each


class TestSizeBucketSampler(unittest.TestCase):
    def test_length_deterministic(self):
        sampler = SizeBucketBatchSampler(BUCKET_RANGES, batch_size=8, shuffle=False)
        self.assertEqual(len(sampler), len(list(iter(sampler))))

    def test_length_shuffle(self):
        sampler = SizeBucketBatchSampler(BUCKET_RANGES, batch_size=8, shuffle=True)
        self.assertEqual(len(sampler), len(list(iter(sampler))))

    def test_no_batch_crosses_bucket_boundary(self):
        sampler = SizeBucketBatchSampler(BUCKET_RANGES, batch_size=8, shuffle=True)
        for batch in iter(sampler):
            owners = [bi for bi, (st, en) in enumerate(BUCKET_RANGES)
                      if st <= min(batch) and max(batch) < en]
            self.assertEqual(len(owners), 1, f"batch {batch} crosses buckets")

    def test_all_indices_covered(self):
        sampler = SizeBucketBatchSampler(BUCKET_RANGES, batch_size=8, shuffle=False)
        all_idx = sorted(i for batch in iter(sampler) for i in batch)
        self.assertEqual(all_idx, list(range(60)))


class TestFleetPerInstance(unittest.TestCase):
    """generate_dataset with a list M sweeps num_vehicle per-instance."""

    def test_fleet_list_mixes_M(self):
        ds = generate_dataset(64, 30, 30, [2, 3, 5, 7, 10], num_workers=0)
        nv = ds["num_vehicle"]
        self.assertEqual(tuple(nv.shape), (64,))
        self.assertGreaterEqual(len(set(nv.tolist())), 2)

    def test_single_fleet(self):
        ds = generate_dataset(8, 30, 30, 3, num_workers=0)
        self.assertEqual(sorted(set(ds["num_vehicle"].tolist())), [3])

    def test_reset_per_instance_M(self):
        import torch
        from env.env import CARPEnv
        from policy.policy import AttentionModelPolicy

        torch.manual_seed(0)
        env = CARPEnv(num_vehicle=[2, 3, 7], variant="P", reward_mode="vector")
        ds = generate_dataset(12, 30, 30, [2, 3, 7], num_workers=0)
        td = env.reset(ds)
        self.assertEqual(tuple(td["num_vehicle"].shape), (12, 1))
        policy = AttentionModelPolicy(embed_dim=32, num_encoder_layers=1, num_heads=4)
        out = policy(td, env, phase="train")
        r = out["reward"]
        self.assertEqual(tuple(r.shape), (12, 3))
        self.assertTrue(r.isfinite().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
