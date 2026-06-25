"""Generate fixed val benchmark datasets for curriculum phase evaluation.

Train data is regenerated on-the-fly by the trainer. Only val data is fixed
so evaluation is comparable across checkpoints.

  data/bench_small_val.data   — 20:40        × 10k
  data/bench_medium_val.data  — 30:60 + 40:80 × 5k each
  data/bench_large_val.data   — 50:100 + 40:120 × 5k each

Run: uv run python scripts/gen_data_benchmark.py
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.generator import generate_dataset

FLEET = [2, 3, 5, 7, 10]
SEED  = 42

# (name, [(num_loc, num_arc, n_val), ...])
GROUPS = [
    ("bench_small_val",  [(20,  40, 10000)]),
    ("bench_medium_val", [(30,  60,  5000), (40,  80, 5000)]),
    ("bench_large_val",  [(50, 100,  5000), (40, 120, 5000)]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(SEED)

    for name, sizes in GROUPS:
        path = os.path.join(args.out_dir, f"{name}.data")
        buckets, ranges, start = [], [], 0
        for num_loc, num_arc, n in sizes:
            print(f"  generating {n} x ({num_loc}:{num_arc}) ...", flush=True)
            td = generate_dataset(n, num_loc, num_arc, FLEET,
                                  num_workers=args.num_workers, progress=True)
            buckets.append(td)
            ranges.append((start, start + n))
            start += n
        torch.save({"buckets": buckets, "ranges": ranges}, path)
        sizes_str = " + ".join(f"{nl}:{na}(×{n})" for nl, na, n in sizes)
        print(f"[{name}] {start} instances ({sizes_str}) → {path}")


if __name__ == "__main__":
    main()
