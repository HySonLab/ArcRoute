import torch
import numpy as np
from torch.distributions import Uniform, Normal
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os

from utils.ops import dist_edges_from_file

# Plan Phase 1: hard cap |A_r| <= 100 (n <= 101) so every config fits one 4090
# without AMP/checkpointing. With ¼-split |A_r| = 3*floor(|A|/4), this means
# |A| <= 135 (3*floor(135/4)=99). Configs above raise (fail-fast).
MAX_REQUIRED = 100


def required_arcs(num_arc):
    """|A_r| = 3*floor(|A|/4) (paper F2 ¼-split)."""
    return 3 * (num_arc // 4)


def _pick(v):
    """Resolve a hyperparameter into a concrete int. Convention:
      - scalar        -> itself
      - tuple (lo,hi) -> uniform int in the INCLUSIVE range [lo, hi]
      - list [...]    -> uniform DISCRETE choice (e.g. fleet M in {1,2,3,5,7,10})
    """
    if isinstance(v, tuple):
        return int(torch.randint(int(v[0]), int(v[1]) + 1, (1,)))
    if isinstance(v, list):
        return int(v[int(torch.randint(len(v), (1,)))])
    return int(v)


def get_sampler(
    distribution: str,
    low: float = 0.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    **kwargs,
):
    if distribution == "uniform":
        return Uniform(low, high)
    elif distribution == "normal":
        return Normal(mean, std)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

# ---------------------------------------------------------------------------- #
# Paper F1 graph (Hà 2024): a SPARSE, strongly-connected directed graph on the
# unit square. NOTE (plan Phase 0 §0.6): the old generator used `cdist` over all
# nodes -> `adj` was the COMPLETE-Euclidean metric (straight-line teleport),
# whereas test (utils.ops.import_instance) uses floyd-warshall over the SPARSE
# arc set. That train/test metric gap is closed here: we build a real sparse
# strongly-connected graph and compute `adj` with the SAME pipeline as test.
# ---------------------------------------------------------------------------- #
def sample_arcs(num_loc, num_arc):
    """Sparse strongly-connected directed graph on `num_loc` nodes (depot = 0)
    with exactly `num_arc` distinct arcs. A random directed Hamiltonian cycle
    guarantees strong connectivity; extra distinct, non-self-loop arcs are added
    up to `num_arc`. Returns an (num_arc, 2) int tensor of (tail, head).
    """
    assert num_arc >= num_loc, (
        f"num_arc({num_arc}) must be >= num_loc({num_loc}): a Hamiltonian cycle "
        f"needs num_loc arcs (paper |A|=n*d, d>=1.5 always satisfies this)."
    )
    perm = torch.randperm(num_loc)
    cycle = torch.stack([perm, torch.roll(perm, -1)], dim=1)   # (num_loc, 2)
    edges = [(int(a), int(b)) for a, b in cycle.tolist()]
    seen = set(edges)
    while len(edges) < num_arc:
        u = int(torch.randint(num_loc, (1,)))
        v = int(torch.randint(num_loc, (1,)))
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    return torch.tensor(edges, dtype=torch.long)

def sample_priority_classes(num_arc, P=3):
    """Paper F2 generalized: each of the P priority classes {1..P} gets exactly
    floor(num_arc/(P+1)) arcs; the remainder are non-required (class 0). The split
    is 1/(P+1) so #required ≈ P/(P+1) of all arcs (P=3 -> the paper's ¼-split).
    Returns a per-arc class vector of length `num_arc` (NO depot row — the depot is
    a prepended row in `generate`, mirroring utils.ops.import_instance).
    """
    per_class = num_arc // (P + 1)
    clss = torch.zeros(num_arc, dtype=torch.int64)
    perm = torch.randperm(num_arc)
    for c in range(1, P + 1):
        idx = perm[(c - 1) * per_class: c * per_class]
        clss[idx] = c
    return clss

def sample_service_time(traversal_time):
    # Paper F4: service = 2 * traversal.
    return traversal_time * 2

def sample_demand(traversal_time):
    # Paper F4: q_a = 0.5 * d_a + 0.5 (required arcs only).
    return traversal_time * 0.5 + 0.5

def sample_vehicle_capacity(q_req):
    # Paper F5: Q = (sum over required arcs of q_a) / 3 + 0.5  (add 0.5 ONCE).
    return q_req.sum() / 3 + 0.5

def build_sparse_adj(edges, d, req_mask):
    """Shortest-path adjacency over the SPARSE arc set, restricted to required
    arcs (+ a prepended depot row 0). Reuses utils.ops.dist_edges_from_file so
    the train metric is byte-identical to test (import_instance). `adj` only
    depends on (tail, head, traversal), so demand/class/service columns are 0.
    Returns a (|A_r|+1, |A_r|+1) float tensor.
    """
    e = edges.numpy().astype(np.float64)
    dd = d.numpy().astype(np.float64)
    m = np.asarray(req_mask)

    def _cols(mask):
        k = int(mask.sum())
        z = np.zeros(k)
        return np.column_stack([e[mask, 0], e[mask, 1], z, z, z, dd[mask]])

    dms = dist_edges_from_file({"req": _cols(m), "nonreq": _cols(~m)})
    return torch.from_numpy(dms).float()

def _pick_density(density, num_loc):
    """Pick a density d (paper F1: d in {1.5,2,2.5,3}) from a scalar or list.
    From a list, prefer values whose |A|=round(n*d) stays within the cap."""
    if isinstance(density, (tuple, list)):
        valid = [d for d in density
                 if required_arcs(round(num_loc * d)) <= MAX_REQUIRED]
        choices = valid if valid else list(density)
        return float(choices[int(torch.randint(len(choices), (1,)))])
    return float(density)


def generate(num_loc, num_arc=None, num_vehicle=3, density=None, P=3):
    # Phase 1: each arg may be a scalar or an inclusive (lo, hi) range. A single
    # generate() call resolves to ONE concrete size (the dataloader must keep a
    # batch single-size; see save_cache + the encoder's `assert mask is None`).
    num_loc = _pick(num_loc)
    num_vehicle = _pick(num_vehicle)
    # Phase 2: if a density d is given, |A| = round(n*d) (paper F1). d overrides
    # num_arc; report/sweep d in {1.5,2,2.5,3}.
    if density is not None:
        num_arc = round(num_loc * _pick_density(density, num_loc))
    else:
        num_arc = _pick(num_arc)
    assert num_arc is not None, "pass either num_arc or density"
    assert required_arcs(num_arc) <= MAX_REQUIRED, (
        f"|A_r|={required_arcs(num_arc)} (|A|={num_arc}) exceeds the hard cap "
        f"{MAX_REQUIRED}; pick a smaller num_arc (|A| <= 135)."
    )

    coord_sampler = get_sampler("uniform", low=0, high=1)
    coords = coord_sampler.sample((num_loc, 2))
    edges = sample_arcs(num_loc, num_arc)

    # Per-arc Euclidean traversal time, normalized by the max (paper d_a=d'/d'max).
    d_eucl = (coords[edges[:, 0]] - coords[edges[:, 1]]).pow(2).sum(-1).sqrt()
    d = d_eucl / d_eucl.max()

    clss_e = sample_priority_classes(num_arc, P)
    req_mask = clss_e > 0

    d_req = d[req_mask]
    clss_req = clss_e[req_mask]
    s_req = sample_service_time(d_req)
    q_req = sample_demand(d_req)
    C = sample_vehicle_capacity(q_req)

    # Sparse shortest-path adjacency, identical pipeline to import_instance.
    adj = build_sparse_adj(edges, d, req_mask.numpy())

    # Prepend the depot row (all zeros) -> mirror utils.ops.import_instance,
    # which does `vstack([[0]*6, req])`. So index 0 is the depot.
    z1 = torch.zeros(1)
    td = TensorDict(
            {
                'clss': torch.cat([torch.zeros(1, dtype=torch.int64), clss_req]),
                "demands": torch.cat([z1, q_req]) / C,
                "capacity": 1,
                "service_times": torch.cat([z1, s_req]),
                "traversal_times": torch.cat([z1, d_req]),
                "adj": adj,
                "num_vehicle": num_vehicle
            },
        )
    td = td.unsqueeze(0)
    td.batch_size=torch.Size([1])
    return td

def generate_dataset(num_samples, num_loc, num_arc, num_vehicle, num_workers=24,
                     progress=False, P=3):
    """Generate `num_samples` instances of ONE size in parallel; returns a single
    batched TensorDict (all the same size, so torch.cat is valid)."""
    # Size is fixed ONCE per dataset (a batch must stay single-size for torch.cat),
    # but num_vehicle is left as-is (scalar OR fleet list) and resolved PER-INSTANCE
    # inside generate() -> so a fleet like [3,5,7,10] mixes M across instances
    # (dynamic_plan Phase 3). M is a scalar field, it does not change tensor shapes.
    num_loc, num_arc = _pick(num_loc), _pick(num_arc)

    class WrapDataset(Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return generate(num_loc, num_arc, num_vehicle, P=P)

        @staticmethod
        def collate_fn(batch):
            return torch.cat(batch, dim=0)

    dataloader = DataLoader(
        WrapDataset(), batch_size=128, shuffle=False,
        num_workers=num_workers, collate_fn=WrapDataset.collate_fn,
    )
    it = tqdm(dataloader) if progress else dataloader
    return torch.cat([td for td in it], dim=0)


def save_cache(num_sample, num_loc, num_arc, num_vehicle, path_data="carp_data.pt", P=3):
    # Bucketing (Phase 1 §1.3): the encoder cannot mix sizes in one batch
    # (collate = torch.cat of (1,n,n); encoder asserts mask is None). So resolve
    # any ranges to ONE concrete size here -> a cache file is a single bucket.
    tds = generate_dataset(num_sample, num_loc, num_arc, num_vehicle, progress=True, P=P)

    parent_dir = os.path.dirname(path_data)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(tds, path_data)
    print(f"Saved dataset to {path_data}...")


class SizeBucketBatchSampler:
    """Phase 6: yields batches whose indices all live in the SAME size bucket, so
    every collated batch is single-size (the encoder cannot mix sizes). Indices
    within a bucket and the batch order are reshuffled each epoch when shuffle."""

    def __init__(self, bucket_ranges, batch_size, shuffle=True):
        self.bucket_ranges = list(bucket_ranges)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._len = sum((en - st + batch_size - 1) // batch_size
                        for st, en in self.bucket_ranges)
        # Lightning introspects `batch_sampler.sampler` to detect shuffling; expose
        # one matching `shuffle` so the multi-size (--sizes) path works under the
        # installed Lightning. The real batching is done in __iter__.
        n = max((en for _, en in self.bucket_ranges), default=0)
        self.sampler = RandomSampler(range(n)) if shuffle else SequentialSampler(range(n))

    def __iter__(self):
        batches = []
        for st, en in self.bucket_ranges:
            idx = torch.arange(st, en)
            if self.shuffle:
                idx = idx[torch.randperm(len(idx))]
            idx = idx.tolist()
            for i in range(0, len(idx), self.batch_size):
                batches.append(idx[i:i + self.batch_size])
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return iter(batches)

    def __len__(self):
        return self._len


class MultiSizeCARPGenerator(Dataset):
    """Phase 6: a training dataset spanning several sizes (the |A_r| ladder). Data
    is grouped into per-size buckets; pair with SizeBucketBatchSampler to keep
    each batch single-size. `sizes` is a list of (num_loc, num_arc) pairs."""

    def __init__(self, num_samples, sizes, num_vehicle, num_workers=24, data=None, P=3):
        if isinstance(data, str) and os.path.exists(data):
            payload = torch.load(data, weights_only=False)
            self.buckets, self.bucket_ranges = payload["buckets"], payload["ranges"]
        else:
            per = max(1, num_samples // len(sizes))
            self.buckets, self.bucket_ranges, start = [], [], 0
            for (nl, na) in sizes:
                tds = generate_dataset(per, nl, na, num_vehicle, num_workers, P=P)
                self.buckets.append(tds)
                self.bucket_ranges.append((start, start + len(tds)))
                start += len(tds)
            if isinstance(data, str):
                parent = os.path.dirname(data)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                torch.save({"buckets": self.buckets, "ranges": self.bucket_ranges}, data)
        self._map = [(bi, li) for bi, (st, en) in enumerate(self.bucket_ranges)
                     for li in range(en - st)]

    def __len__(self):
        return len(self._map)

    def __getitem__(self, gidx):
        bi, li = self._map[gidx]
        return self.buckets[bi][li:li + 1]

    @staticmethod
    def collate_fn(batch):
        return torch.cat(batch, dim=0)

class CARPGenerator(Dataset):
    def __init__(self, num_samples=None, num_loc=None, num_arc=None, num_vehicle=None, data="carp_data.pt", P=3):
        if isinstance(data, str):
            if not os.path.exists(data):
                print(f"Don't have {data}, generating...")
                save_cache(num_samples, num_loc, num_arc, num_vehicle, data, P=P)
            self.data = torch.load(data, weights_only=False)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx:idx+1]

    @staticmethod
    def collate_fn(batch):
        return torch.cat(batch, dim=0)
