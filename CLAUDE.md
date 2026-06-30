# CLAUDE.md

Guidance for working in this repository.

## Project

**HDCARP** — Hierarchical Directed Capacitated Arc Routing Problem. Solves arc-routing
with priority classes via a hybrid of Reinforcement Learning (PPO/GRPO + attention policy)
and classical heuristics. Paper: arXiv:2501.00852.

**Layout** — `src/` is an installed (editable) package root; entry-point scripts live in
`scripts/`, generated artifacts in `data/` (inputs) and `outputs/` (checkpoints, lightning_logs).
Three solver families live here:
- **RL** — attention encoder/decoder policy trained with PPO or GRPO (`src/policy/`,
  `src/trainers/` = `base.py`+`ppo.py`+`grpo.py`, `src/train.py`).
- **Metaheuristics** — EA, ACO, ILS (`src/solvers/`, `src/utils/`).
- **Exact** — LP/MIP via gurobi (`src/solvers/lp.py`).
- **Hybrid (HRDA)** — RL policy + local search at inference (`src/solvers/rl_hyb.py`).

`src/trainers/`: `PPO` and `GRPO` are siblings, both inherit `BaseRL` (shared Lightning
scaffolding). PPO = clipped surrogate + critic; GRPO = critic-free REINFORCE with a
lexicographic-rank group advantage. `src/solvers/scheduler.py` maps a policy's arc order →
routes + (T₁,T₂,T₃). The Scheduler/policy/env machinery is adapted from rl4co patterns.

## Environment & commands

**Imports** resolve via the installed package (`uv sync` builds it editable), e.g.
`from trainers.grpo import GRPO`, `from solvers.scheduler import Scheduler`, `from utils.ops import ...`.

**Training runs go through `scripts/train.sh`** — it launches `src/train.py` via `nohup`
(in the background) and writes the log to `logs/train_<timestamp>.out`. To check on a run, read that file
(e.g. `tail -f logs/...`); do **not** redirect to ad-hoc paths like `/tmp/train_test.log`.

**Tests** (built-in `unittest`, no pytest):

    uv run python -m unittest discover -s tests -p "test_*.py"

## Data generators

Two separate paths, both follow the paper's "Create HDCARP instances" formulas
(traversal `d=d'/d'_max`, service `=2d`, demand `=0.5d+0.5`, capacity `Q=Σ(q/3+0.5)`):

- `src/env/generator.py` — synthetic **training** data (random uniform coords). Cached to `data/*.data`.
- `scripts/gen_data.py` — **benchmark** instances from real OSM road graphs (OSMnx). Writes
  `data/<M>m/<|A|>/*.npz`. Pure physics is in `build_instance` (unit-tested); the OSMnx
  step needs `osmnx` installed (`uv add osmnx`). Run: `uv run python scripts/gen_data.py --vehicles {2,5}`.

`.npz` schema: `req`/`nonreq` columns `[tail, head, demand, clss, service, traversal]`,
plus `P=3, M, C`; loaded by `utils.ops.import_instance` (shortest-path adjacency built
via Floyd-Warshall at load, demands normalized by `C`).
