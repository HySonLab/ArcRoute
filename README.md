# Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Test Benchmark Grid](#generate-test-benchmark-grid)
  - [Run Meta Heuristic Algorithms](#run-meta-heuristic-algorithms)
  - [Run Exact Method](#run-exact-method)
  - [Evaluate the RL Policy](#evaluate-the-rl-policy-best-of-k-grid)
  - [RL Training](#rl-training)
- [Results](#results)
- [Weight and Data](#weight_and_data)
- [License](#license)
- [Contact](#contact)

## Overview

This project provides a solution to the **Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)**.

The project implements multiple approaches:

- **Exact methods** for solving smaller instances.
- **Meta Heuristic algorithms** (ea, aco, ils).
- **Reinforcement Learning** — an attention encoder/decoder policy trained with **GRPO** (critic-free, lexicographic-rank advantage).
- **Hybrid algorithm combining Reinforcement Learning (RL) and heuristics**.

## Key Components

The library lives under `src/` (an installed, editable package); runnable entry points are in
`scripts/`, and generated artifacts go to `data/` (inputs) and `outputs/` (checkpoints, logs).

```
ArcRoute/
├── src/
│   ├── env/
│   │   ├── env.py             # Environment (CARPEnv) for the routing problem
│   │   └── generator.py       # Training-instance generator
│   ├── policy/                # attention model (M-agnostic): encoder/decoder/policy/context/init
│   ├── trainers/
│   │   ├── base.py            # BaseRL — shared Lightning scaffolding
│   │   ├── ppo.py             # PPO (clipped surrogate + critic)
│   │   └── grpo.py            # GRPO (critic-free REINFORCE, lexicographic-rank advantage)
│   ├── solvers/
│   │   ├── scheduler.py       # Scheduler Φ: arc order + M -> routes + (T1,T2,T3)
│   │   ├── cal_reward.py      # objective / reward
│   │   ├── aco.py ea.py ils.py# metaheuristics
│   │   ├── lp.py              # exact (gurobi)
│   │   └── rl_hyb.py          # HRDA hybrid (RL + local search)
│   ├── eval/                  # run_grid.py, stats.py
│   └── utils/                 # ops, consts, local_search
├── scripts/
│   ├── train.py  train.sh     # RL training entry point + launcher
│   └── gen_data.py            # benchmark-instance generator
├── tests/                     # unittest suite
├── configs/  docs/
├── data/                      # inputs/artifacts (gitignored)   outputs/  # checkpoints + logs
├── pyproject.toml  setup.sh  uv.lock  README.md  CLAUDE.md  LICENSE
```


## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package/environment manager. Just run:

```bash
./setup.sh
```

This installs uv (if missing), creates a `.venv`, and installs all dependencies from `pyproject.toml`. Then either activate the env or prefix commands with `uv run`:

```bash
source .venv/bin/activate        # then: python scripts/train.py ...
# or
uv run python scripts/train.py --help
```

`uv sync` builds the project as an editable package, so the `src/` modules import directly
(e.g. `from trainers.grpo import GRPO`, `from solvers.scheduler import Scheduler`).

## Usage

### Generate Test Benchmark Grid

Test instances are written as `.npz` files under `data/ood/<topology>/<|A|>/` (one
folder per total-arc bucket `|A|`). Each instance follows the paper's F1–F5 formulas
(¼-split required arcs, balanced priority classes, `Q = Σq/3 + 0.5`). The fleet size
`M` is a **solve-time** parameter (stored as a nominal value, overridden at eval), so
arcs are generated **once** — not per `M`.

**Synthetic topologies (no internet / osmnx required):**
```bash
# unit_square (in-distribution, paper F1)  -> data/ood/unit_square/<|A|>/*.npz
uv run python scripts/gen_data.py --topology unit_square \
    --density 1.5 2.0 2.5 3.0 --per_bucket 20 --min_arc 40 --seed 6868

# cluster (OOD, Gaussian clusters)         -> data/ood/cluster/<|A|>/*.npz
uv run python scripts/gen_data.py --topology cluster \
    --density 1.5 2.0 2.5 3.0 --per_bucket 20 --min_arc 40 --seed 6868
```

**Real road networks from OpenStreetMap (requires `uv add osmnx`):**
```bash
# city A = Da Nang  -> data/ood/osm_cityA/<|A|>/*.npz
uv run python scripts/gen_data.py --topology osm --out data/ood/osm_cityA \
    --per_bucket 20 --min_arc 40 --tol 5 --seed 6868 \
    --bbox 16.0741 16.0591 108.2187 108.1972

# city B = Hanoi    -> data/ood/osm_cityB/<|A|>/*.npz
uv run python scripts/gen_data.py --topology osm --out data/ood/osm_cityB \
    --per_bucket 20 --min_arc 40 --tol 6 --seed 777 \
    --bbox 21.0450 21.0180 105.8650 105.8350
```

Key flags: `--topology {unit_square,cluster,osm}`, `--density` (sweeps `d=|A|/|V|`),
`--per_bucket` (instances per `|A|` bucket), `--min_arc` (smallest `|A|`, use 40 for the
full size ladder), `--m_nominal` (nominal fleet stored in the `.npz`), `--tol`
(accepted `|A|` deviation), `--bbox N S E W` (OSM only). Re-running tops up existing
buckets, so it is safe to resume. Training data is generated on the fly (see
[RL Training](#rl-training)) and does not need this script.
### Run Meta Heuristic Algorithms
```bash
    uv run python -m solvers.ils --data_path "data/instances/30/61_20.npz"
    uv run python -m solvers.ea  --data_path "data/instances/30/61_20.npz"
    uv run python -m solvers.aco --data_path "data/instances/30/61_20.npz"
```

### Run Exact Method
```bash
    uv run python -m solvers.lp
```
### Evaluate the RL Policy (best-of-K grid)
```bash
    uv run python -m eval.run_grid \
    --ckpt "outputs/checkpoints/.../best.ckpt" \
    --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 \
    --out outputs/grid.csv
```

### RL Training

Use the launcher (`scripts/train.sh`, runs in the background via `nohup`):
```bash
    MODE=validate ALGO=grpo ./scripts/train.sh      # GRPO (critic-free, default)
    MODE=full     ALGO=ppo  ./scripts/train.sh      # PPO baseline
```

Or call the entry point directly. `--algo {ppo,grpo}` selects the trainer (GRPO auto-uses the
vector reward + the `val/lex_best` checkpoint monitor; `--group_size K` sets the group size):
```bash
    uv run python scripts/train.py \
    --algo grpo --group_size 8 \
    --seed 6868 \
    --max_epoch 1000 \
    --batch_size 4096 \
    --mini_batch_size 512 \
    --train_data_size 100000 \
    --val_data_size 10000 \
    --embed_dim 128 \
    --num_encoder_layers 12 \
    --num_heads 8 \
    --num_loc 20 \
    --num_arc 20 \
    --variant P \
    --checkpoint_dir outputs/checkpoints/cl123 \
    --accelerator gpu \
    --devices 1
```

## Results

## Weight_and_Data
[Weight of HDRA](https://drive.google.com/drive/folders/16R3gudbrDpMo9sZn-iLxHVXkXTHUCVQi)

[Data](https://drive.google.com/drive/folders/1JAgkaH1TnPz7zls_oCbsjJP2nGpkTRPm)

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Please cite our work!

```bibtex
@misc{nguyen2025hybridisingreinforcementlearningheuristics,
      title={Hybridising Reinforcement Learning and Heuristics for Hierarchical Directed Arc Routing Problems}, 
      author={Van Quang Nguyen and Quoc Chuong Nguyen and Thu Huong Dang and Truong-Son Hy},
      year={2025},
      eprint={2501.00852},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.00852}, 
}
```

## Contact
For questions or collaboration inquiries, please reach out to Truong-Son Hy at thy@uab.edu
