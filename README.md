# Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Test Benchmark Grid](#generate-test-benchmark-grid)
  - [Run Heuristic Algorithms](run-heuristic-algorithms)
  - [Run Exact Method](#train-reinforcement-learning-model)
  - [Run RL method](#evaluate-performance)
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
- **Hybrid algorithm combining Reinforcement Learning (RL) and heuristics**.

## Key Components

The project is organized into several directories:

```
hdcarp/
├── baseline/
│   ├── aco.py                 # aco algorithm
│   ├── ea.py                  # ea algorithm
│   ├── ils.py                 # ea algorithm
│   ├── rl_hyb.py              # HRDA algorithm
│   ├── lp.py                  # exact method
│   ├── meta.py                # implemented code of Meta Heuristic algorithms
├── common/
├── env/
│   ├── env.py                # Environment setup for the routing problem
│   ├── generator.py          # Problem instance generator
├── policy/
│   ├── context.py            # Contextual features for the RL model
│   ├── encoder.py            # Encoding components
│   ├── decoder.py            # Decoding components
│   ├── init.py               # Model initialization functions
│   ├── policy.py             # Policy network for RL
├── rl/
│   ├── critic.py
│   ├── ppo.py                # Proximal Policy Optimization (PPO) algorithm
│   ├── policy.py             # Policy network for RL
│   ├── trainer.py
├── .gitignore                # Git ignore file
├── pyproject.toml            # Python dependencies (managed with uv)
├── setup.sh                  # One-step environment setup via uv
├── train.py                  # Main script to start training the models
├── README.md                 # Project documentation
├── LICENSE                   # License information
```


## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package/environment manager. Just run:

```bash
./setup.sh
```

This installs uv (if missing), creates a `.venv`, and installs all dependencies from `pyproject.toml`. Then either activate the env or prefix commands with `uv run`:

```bash
source .venv/bin/activate        # then: python train.py ...
# or
uv run python train.py --help
```

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
uv run python data/gen.py --topology unit_square \
    --density 1.5 2.0 2.5 3.0 --per_bucket 20 --min_arc 40 --seed 6868

# cluster (OOD, Gaussian clusters)         -> data/ood/cluster/<|A|>/*.npz
uv run python data/gen.py --topology cluster \
    --density 1.5 2.0 2.5 3.0 --per_bucket 20 --min_arc 40 --seed 6868
```

**Real road networks from OpenStreetMap (requires `uv add osmnx`):**
```bash
# city A = Da Nang  -> data/ood/osm_cityA/<|A|>/*.npz
uv run python data/gen.py --topology osm --out data/ood/osm_cityA \
    --per_bucket 20 --min_arc 40 --tol 5 --seed 6868 \
    --bbox 16.0741 16.0591 108.2187 108.1972

# city B = Hanoi    -> data/ood/osm_cityB/<|A|>/*.npz
uv run python data/gen.py --topology osm --out data/ood/osm_cityB \
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
```python
    python3 baseline/ils.py --data_path "data/instances/30/61_20.npz"
    python3 baseline/ea.py --data_path "data/instances/30/61_20.npz"
    python3 baseline/aco.py --data_path "data/instances/30/61_20.npz"
```

### Run Exact Method
```python
    python3 baseline/lp.py
```
### Run RL Method
```python
    python3 baseline/rl_infer.py \
    --checkpoint_path "best.ckpt" \
    --data_path "data/30/61_20.npz"
```

### RL Training
```python
    python3 train.py \
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
    --checkpoint_dir /home/project/checkpoints/cl123 \
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
