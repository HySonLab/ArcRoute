# CLAUDE.md

Guidance for working in this repository.

## Project

**HDCARP** — Hierarchical Directed Capacitated Arc Routing Problem. Solves arc-routing
with priority classes via a hybrid of Reinforcement Learning (PPO + attention policy)
and classical heuristics. Paper: arXiv:2501.00852.

Three solver families live here:
- **RL** — attention encoder/decoder policy trained with PPO (`policy/`, `rl/`, `train.py`).
- **Metaheuristics** — EA, ACO, ILS (`baseline/`, `common/`).
- **Exact** — LP/MIP via gurobi (`baseline/lp.py`).
- **Hybrid (HRDA)** — RL policy + local search at inference (`baseline/rl_hyb.py`).

## Environment & commands

**Training runs go through `train.sh`** — it launches `train.py` via `nohup` (in the background)
and writes the log to `logs/train_<timestamp>.out`. To check on a run, read that file
(e.g. `tail -f logs/...`); do **not** redirect to ad-hoc paths like `/tmp/train_test.log`.

**Tests** (built-in `unittest`, no pytest):

    uv run python -m unittest discover -s tests -p "test_*.py"

## Data generators

Two separate paths, both follow the paper's "Create HDCARP instances" formulas
(traversal `d=d'/d'_max`, service `=2d`, demand `=0.5d+0.5`, capacity `Q=Σ(q/3+0.5)`):

- `env/generator.py` — synthetic **training** data (random uniform coords). Cached to `data/*.data`.
- `data/gen.py` — **benchmark** instances from real OSM road graphs (OSMnx). Writes
  `data/<M>m/<|A|>/*.npz`. Pure physics is in `build_instance` (unit-tested); the OSMnx
  step needs `osmnx` installed (`uv add osmnx`). Run: `uv run python data/gen.py --vehicles {2,5}`.

`.npz` schema: `req`/`nonreq` columns `[tail, head, demand, clss, service, traversal]`,
plus `P=3, M, C`; loaded by `common/ops.import_instance` (shortest-path adjacency built
via Floyd-Warshall at load, demands normalized by `C`).
