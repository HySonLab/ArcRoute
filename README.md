# Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Test Benchmark Grid](#generate-test-benchmark-grid)
  - [Run Benchmarks](#run-benchmarks)
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
source .venv/bin/activate        # then: python src/train.py ...
# or
uv run python src/train.py --help
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

### Run Benchmarks

Use `scripts/bm.sh` as the single benchmark entry point for all solvers. The benchmark
directory must contain instances in `<size>/*.npz` subdirectories, for example:

```text
data/5m/
├── 40/
│   └── instance_01.npz
└── 50/
    └── instance_01.npz
```

Set `SOLVER` to `ea`, `ils`, `aco`, `lp`, or `rl`. By default the script reads
`data/5m`, uses priority variant `P` and seed `6868`, and writes output to
`logs/bm_<solver>_<variant>_<timestamp>.out`.

```bash
# Evolutionary Algorithm (EAHCARP)
SOLVER=ea bash scripts/bm.sh

# Iterated Local Search (ILSHCARP)
SOLVER=ils bash scripts/bm.sh

# Ant Colony Optimisation (ACOHCARP)
SOLVER=aco bash scripts/bm.sh

# Exact MILP solver (SCIP)
SOLVER=lp bash scripts/bm.sh

# RL policy (GRPO or PPO checkpoint)
SOLVER=rl CKPT=outputs/checkpoints/best.ckpt bash scripts/bm.sh

# Hybrid RL + local search
SOLVER=rl CKPT=outputs/checkpoints/best.ckpt HYBRID=1 \
    NUM_SAMPLE=400 SHORTLIST=25 TOPK_T1=100 ORACLE_HEAD=1 \
    bash scripts/bm.sh
```

Common overrides apply to every solver:

```bash
DATA_DIR=data/ood/unit_square VARIANT=U SEED=777 M=5 \
    SOLVER=ils bash scripts/bm.sh
```

| Solver | Solver-specific environment variables |
|---|---|
| EA | `N_EPOCH` (100), `N_POPULATION` (200) |
| ACO | `N_EPOCH` (100), `N_ANT` (50) |
| ILS | `MAX_ITER` (200), `NUM_INIT_SAMPLE` (5), `STRENGTH` (3), `ACCEPT_MODE` (`best` or `sa`) |
| LP | `TIME_LIMIT` (3600 seconds), `THREADS` (8; currently ignored by community SCIP) |
| RL | `CKPT` (required), `NUM_SAMPLE` (100), `EPS` (0.005) |
| RL hybrid | `HYBRID=1`, `SHORTLIST` (10), `TOPK_T1` (0), `ORACLE_HEAD` (0) |

Examples with solver-specific settings:

```bash
SOLVER=ea DATA_DIR=data/2m VARIANT=U N_EPOCH=200 N_POPULATION=300 \
    bash scripts/bm.sh

SOLVER=ils MAX_ITER=500 NUM_INIT_SAMPLE=10 STRENGTH=4 ACCEPT_MODE=sa \
    bash scripts/bm.sh

SOLVER=aco N_EPOCH=200 N_ANT=100 bash scripts/bm.sh

SOLVER=lp TIME_LIMIT=300 bash scripts/bm.sh
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
    uv run python src/train.py \
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
