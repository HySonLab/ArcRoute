# ArcRoute

ArcRoute solves the **Hierarchical Directed Capacitated Arc Routing Problem
(HDCARP)** using:

- Evolutionary Algorithm (EA)
- Iterated Local Search (ILS)
- Ant Colony Optimisation (ACO)
- Exact MILP (SCIP)
- Reinforcement Learning (GRPO)
- Hybrid RL and local search

## Installation

```bash
./setup.sh
```

The setup script installs `uv`, creates `.venv`, and installs the project dependencies.

## Benchmark data

Benchmark files must follow this structure:

```text
data/5m/
├── 40/*.npz
├── 50/*.npz
└── 60/*.npz
```

Generate synthetic data with:

```bash
uv run python scripts/gen_data.py --topology unit_square \
    --density 1.5 2.0 2.5 3.0 --per_bucket 20 --min_arc 40 --seed 6868
```

## Run benchmarks

Use `scripts/bm.sh` for every solver:

```bash
SOLVER=ea  bash scripts/bm.sh
SOLVER=ils bash scripts/bm.sh
SOLVER=aco bash scripts/bm.sh
SOLVER=lp  bash scripts/bm.sh

SOLVER=rl CKPT=outputs/checkpoints/best.ckpt bash scripts/bm.sh
```

The default dataset is `data/5m`. Override settings with environment variables:

```bash
SOLVER=ils DATA_DIR=data/ood/unit_square VARIANT=P M=5 \
    MAX_ITER=500 bash scripts/bm.sh
```

Run the hybrid RL solver with:

```bash
SOLVER=rl CKPT=outputs/checkpoints/best.ckpt HYBRID=1 \
    NUM_SAMPLE=400 SHORTLIST=25 TOPK_T1=100 bash scripts/bm.sh
```

Results are saved in `logs/`.

## Train an RL model

```bash
MODE=validate ALGO=grpo ./scripts/train.sh
MODE=full ALGO=grpo ./scripts/train.sh
```

## Pretrained weights and data

- [Download pretrained weights and data](https://drive.google.com/drive/folders/1IIvJWIM-3676Nl30gROam8qQsNdRwBPU?usp=sharing)

## Citation

```bibtex
@misc{nguyen2025hybridisingreinforcementlearningheuristics,
  title={Hybridising Reinforcement Learning and Heuristics for Hierarchical Directed Arc Routing Problems},
  author={Van Quang Nguyen and Quoc Chuong Nguyen and Thu Huong Dang and Truong-Son Hy},
  year={2025},
  eprint={2501.00852},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2501.00852}
}
```

## License

MIT License. See [LICENSE](LICENSE).

Contact: Truong-Son Hy at thy@uab.edu.
