"""OSM training dataset for HDCARP — M-agnostic, drop-in for MultiSizeCARPGenerator.

Reads .npz files produced by scripts/gen_test_instances.py from
data/osm_train/<city>/<bucket>/*.npz and exposes them as a PyTorch Dataset
that the existing SizeBucketBatchSampler can consume directly.

Key design points (docs/strategy_data.md §5):
  - M-agnostic storage: .npz carries raw demands q; each __getitem__ samples
    M ∈ vehicle_choices at random and recomputes C = Σq/M + 0.5.
  - Adj (Floyd-Warshall shortest paths over the sparse arc set) is computed
    once at load time and cached in RAM — it is M-independent.
  - Instances are grouped by exact |A_r| = 3*(|A|//4) so every batch served
    by SizeBucketBatchSampler has a uniform TensorDict shape.
  - bucket_ranges / collate_fn match MultiSizeCARPGenerator's interface exactly.
  - Optional on-disk cache (cache_path=...) skips Floyd-Warshall on re-runs.

TensorDict layout (identical to env.generator.generate()):
    clss            (|A_r|+1,)   int64   — depot=0, then priority class 1/2/3
    demands         (|A_r|+1,)   float32 — normalized by C; depot=0
    capacity        scalar       float32 — always 1.0 (demands already normalised)
    service_times   (|A_r|+1,)   float32 — depot=0
    traversal_times (|A_r|+1,)   float32 — depot=0
    adj             (|A_r|+1, |A_r|+1)   float32 — Floyd-Warshall + dist_edges
    num_vehicle     scalar       int64   — sampled M

Usage:
    from env.osm_dataset import OSMMultiSizeDataset
    from env.generator import SizeBucketBatchSampler
    from torch.utils.data import DataLoader

    ds = OSMMultiSizeDataset("data/osm_train")
    sampler = SizeBucketBatchSampler(ds.bucket_ranges, batch_size=128)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=ds.collate_fn)
"""
import os
import random

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.ops import dist_edges_from_file


class OSMMultiSizeDataset(Dataset):
    """M-agnostic OSM training dataset.

    Parameters
    ----------
    data_dir:
        Root directory containing .npz files, e.g. data/osm_train/.
        The directory is walked recursively; only .npz files are read.
    vehicle_choices:
        Pool of M values to sample uniformly on each __getitem__ call.
        Defaults to {2, 3, 5, 7, 10} (strategy_data.md §3).
    cache_path:
        If provided, save the processed in-RAM cache to this .pt file on
        first load and reload from it on subsequent runs (skips Floyd-Warshall).
    """

    VEHICLE_CHOICES: list[int] = [2, 3, 5, 7, 10]

    def __init__(
        self,
        data_dir: str = "data/osm_train",
        vehicle_choices: list[int] | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.vehicle_choices = vehicle_choices or self.VEHICLE_CHOICES

        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._load_from_dir()
            if cache_path:
                self._save_to_cache(cache_path)

    # ---------------------------------------------------------------------- #
    # Loading                                                                  #
    # ---------------------------------------------------------------------- #
    def _load_from_dir(self) -> None:
        """Walk data_dir, load every .npz, compute adj, group by |A_r|."""
        npz_files = sorted(
            os.path.join(root, fname)
            for root, _, files in os.walk(self.data_dir)
            for fname in files
            if fname.endswith(".npz")
        )
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz files found under {self.data_dir!r}. "
                "Run scripts/gen_osm_train.sh first."
            )

        # Group by |A_r| so same-size instances are contiguous.
        items_by_nreq: dict[int, list[dict]] = {}

        for fpath in tqdm(npz_files, desc="Loading OSM instances", unit="inst"):
            try:
                item = _load_npz(fpath)
            except Exception as exc:
                print(f"[OSMMultiSizeDataset] skipping {fpath}: {exc}")
                continue
            n_req = item["n_req"]
            items_by_nreq.setdefault(n_req, []).append(item)

        # Build flat item list sorted by |A_r|, record bucket_ranges.
        self._items: list[dict] = []
        self.bucket_ranges: list[tuple[int, int]] = []
        start = 0
        for n_req in sorted(items_by_nreq):
            group = items_by_nreq[n_req]
            self._items.extend(group)
            end = start + len(group)
            self.bucket_ranges.append((start, end))
            start = end

        print(
            f"[OSMMultiSizeDataset] loaded {len(self._items)} instances, "
            f"{len(self.bucket_ranges)} size bucket(s)."
        )

    def _save_to_cache(self, cache_path: str) -> None:
        """Persist the in-RAM cache to disk for fast reloads."""
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {"items": self._items, "bucket_ranges": self.bucket_ranges},
            cache_path,
        )
        print(f"[OSMMultiSizeDataset] cache saved to {cache_path}")

    def _load_from_cache(self, cache_path: str) -> None:
        """Reload from a previously saved .pt cache (skips Floyd-Warshall)."""
        payload = torch.load(cache_path, weights_only=False)
        self._items = payload["items"]
        self.bucket_ranges = payload["bucket_ranges"]
        print(
            f"[OSMMultiSizeDataset] loaded {len(self._items)} instances from "
            f"cache {cache_path} ({len(self.bucket_ranges)} bucket(s))."
        )

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                         #
    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> TensorDict:
        item = self._items[idx]

        # Sample M and recompute capacity (M-agnostic storage, strategy_data §0).
        M = random.choice(self.vehicle_choices)
        q: torch.Tensor = item["raw_q"]            # (|A_r|,) raw (unnormalized) demands
        C = float(q.sum().item() / M + 0.5)

        td = TensorDict(
            {
                "clss": item["clss"],                          # (|A_r|+1,) int64
                "demands": torch.cat([torch.zeros(1), q]) / C, # (|A_r|+1,) float32
                "capacity": 1,
                "service_times": item["service_times"],        # (|A_r|+1,) float32
                "traversal_times": item["traversal_times"],    # (|A_r|+1,) float32
                "adj": item["adj"],                            # (|A_r|+1, |A_r|+1) float32
                "num_vehicle": M,
            },
        )
        td = td.unsqueeze(0)
        td.batch_size = torch.Size([1])
        return td

    @staticmethod
    def collate_fn(batch: list[TensorDict]) -> TensorDict:
        """Concatenate a list of single-item TensorDicts into a batch."""
        return torch.cat(batch, dim=0)


# --------------------------------------------------------------------------- #
# Per-file loader (pure function, easy to unit-test)
# --------------------------------------------------------------------------- #
def _load_npz(fpath: str) -> dict:
    """Load one .npz, compute adj, return a cacheable dict.

    The dict contains:
        n_req          int   — |A_r| (used for bucket grouping)
        adj            Tensor (|A_r|+1, |A_r|+1) float32
        clss           Tensor (|A_r|+1,) int64    — depot=0
        raw_q          Tensor (|A_r|,)   float32  — un-normalised demands
        service_times  Tensor (|A_r|+1,) float32  — depot=0
        traversal_times Tensor(|A_r|+1,) float32  — depot=0
    """
    es = np.load(fpath)
    req: np.ndarray = es["req"]       # (|A_r|, 6)
    nonreq: np.ndarray = es["nonreq"] # (|A_nr|, 6)
    n_req = len(req)

    # Floyd-Warshall over the full sparse graph → required-arc distance matrix.
    adj_np = dist_edges_from_file({"req": req, "nonreq": nonreq})  # (n_req+1, n_req+1) float32

    z1 = torch.zeros(1)
    return {
        "n_req": n_req,
        "adj": torch.from_numpy(adj_np),                                   # float32
        "clss": torch.cat([
            torch.zeros(1, dtype=torch.int64),
            torch.from_numpy(req[:, 3].astype(np.int64)),
        ]),
        "raw_q": torch.from_numpy(req[:, 2].astype(np.float32)),           # unnormalized
        "service_times": torch.cat([
            z1, torch.from_numpy(req[:, 4].astype(np.float32))
        ]),
        "traversal_times": torch.cat([
            z1, torch.from_numpy(req[:, 5].astype(np.float32))
        ]),
    }
