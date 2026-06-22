"""Unit tests for Phase 6 multi-size bucketed training
(env.generator.MultiSizeCARPGenerator + SizeBucketBatchSampler). The contract:
every collated batch is single-size, yet sizes vary across batches.

Run from the project root:
    uv run python -m unittest tests.test_multisize -v
"""
import unittest

import torch
from torch.utils.data import DataLoader

from env.generator import MultiSizeCARPGenerator, SizeBucketBatchSampler


SIZES = [(20, 40), (30, 60), (40, 80)]     # |A_r| = 30, 45, 60


class TestMultiSizeDataset(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ds = MultiSizeCARPGenerator(num_samples=60, sizes=SIZES, num_vehicle=3,
                                         num_workers=0)

    def test_length_and_buckets(self):
        self.assertEqual(len(self.ds), 60)            # 20 per bucket * 3
        self.assertEqual(len(self.ds.bucket_ranges), 3)

    def test_each_bucket_is_single_size(self):
        for bi, (st, en) in enumerate(self.ds.bucket_ranges):
            sizes = {self.ds[g]["adj"].shape[1] for g in range(st, en)}
            self.assertEqual(len(sizes), 1, f"bucket {bi} mixes sizes: {sizes}")

    def test_expected_sizes_present(self):
        all_n = {self.ds[g]["adj"].shape[1] for g in range(len(self.ds))}
        # |A_r|+1 = 31, 46, 61
        self.assertEqual(all_n, {31, 46, 61})


class TestSizeBucketSampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ds = MultiSizeCARPGenerator(num_samples=60, sizes=SIZES, num_vehicle=3,
                                         num_workers=0)

    def test_every_batch_single_size(self):
        sampler = SizeBucketBatchSampler(self.ds.bucket_ranges, batch_size=8, shuffle=True)
        loader = DataLoader(self.ds, batch_sampler=sampler, num_workers=0,
                            collate_fn=self.ds.collate_fn)
        seen_sizes = set()
        for batch in loader:
            n = batch["adj"].shape[1]                  # collate succeeded -> single-size
            self.assertEqual(batch["adj"].shape[2], n)
            seen_sizes.add(n)
        self.assertEqual(seen_sizes, {31, 46, 61})     # all ladder sizes appear

    def test_sampler_length_matches_iteration(self):
        sampler = SizeBucketBatchSampler(self.ds.bucket_ranges, batch_size=8, shuffle=False)
        self.assertEqual(len(sampler), len(list(iter(sampler))))

    def test_no_batch_crosses_bucket_boundary(self):
        sampler = SizeBucketBatchSampler(self.ds.bucket_ranges, batch_size=8, shuffle=True)
        starts = [st for st, _ in self.ds.bucket_ranges]
        ends = [en for _, en in self.ds.bucket_ranges]
        for batch in iter(sampler):
            # all indices in a batch fall within one [start, end) range
            owner = [bi for bi, (st, en) in enumerate(self.ds.bucket_ranges)
                     if st <= min(batch) and max(batch) < en]
            self.assertEqual(len(owner), 1, f"batch {batch} crosses buckets")


class TestFleetPerInstance(unittest.TestCase):
    """Phase 3: a fleet LIST sweeps M per-instance (M is a scalar -> mixes freely
    in one batch, unlike size). Reward is computed per-instance under each M."""

    def test_fleet_list_mixes_M_per_instance(self):
        from env.generator import generate_dataset
        ds = generate_dataset(64, 30, 30, [2, 3, 5, 7, 10], num_workers=0)
        nv = ds["num_vehicle"]
        self.assertEqual(tuple(nv.shape), (64,))               # scalar per instance
        self.assertGreaterEqual(len(set(nv.tolist())), 2)      # genuinely mixed

    def test_single_fleet_backward_compatible(self):
        from env.generator import generate_dataset
        ds = generate_dataset(8, 30, 30, 3, num_workers=0)
        self.assertEqual(sorted(set(ds["num_vehicle"].tolist())), [3])

    def test_reset_carries_per_instance_M_and_reward_runs(self):
        import numpy as np
        from env.env import CARPEnv
        from env.generator import generate_dataset
        from common.cal_reward import calc_reward

        env = CARPEnv(num_loc=30, num_arc=30, num_vehicle=[2, 3, 7], variant="P")
        ds = generate_dataset(12, 30, 30, [2, 3, 7], num_workers=0)
        td = env.reset(ds)
        self.assertEqual(tuple(td["num_vehicle"].shape), (12, 1))
        # per-instance reward uses that instance's M (run_parallel slices td[i]).
        n = int((td[0]["demand"] > 0).sum())
        action = np.arange(1, n + 1)
        for i in range(12):
            r = calc_reward(action, td[i], return_numpy=True)
            self.assertEqual(r.shape, (3,))
            self.assertTrue(np.all(np.isfinite(r)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
