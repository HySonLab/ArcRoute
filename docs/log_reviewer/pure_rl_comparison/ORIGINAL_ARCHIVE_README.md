# Pure-RL vs. Hybrid Time-Matched Experiment — 2026-07-15

Archived raw logs and analysis for the experiment reported in
`main.tex` §V "Is a well-resourced pure-RL decode enough?" (`sec:purerl-comparison`,
table `tab:purerl`) of the HARMONY paper (`~/projects/HDCARP_with_ML`). Full
write-up with interpretation: `~/projects/HDCARP_with_ML/docs/spec_exp_results.md`.
Original design spec: `~/projects/HDCARP_with_ML/docs/spec_exp.md`.

These logs were originally produced under a temporary session scratchpad
(cleaned up automatically) and archived here on 2026-07-15 so the numbers in
the paper remain reproducible/traceable.

## Layout

- `full_runs/` — the 13 complete 100-instance runs the paper's numbers come
  from directly:
  - `prod_{small,medium,large}.out` — fresh production HARMONY
    (`HYBRID=1 ORACLE_HEAD=1 ORACLE_K=3 NUM_SAMPLE=150 SHORTLIST=25 TOPK_T1=100`)
    on the corrected-environment checkpoints
    (`outputs/checkpoints/grpo_reload_fix/{small,medium,large}/epoch={044,037,041}.ckpt`).
  - `pure_{tier}_{ns}.out` — pure-RL-only decode (`HYBRID=0 ORACLE_HEAD=0`,
    same checkpoints) at each swept `NUM_SAMPLE` value: small
    {8000,16000,32000,64000}, medium {16000,32000,40000}, large
    {8000,12000,16000}.
  - All on `data/test_bm/{tier}.npz`, VARIANT=P, the same 100 held-out
    test-city instances behind the paper's `tab:planned`.
- `calibration/` — smaller (10-instance) probe runs used to pick the sweep
  grid above and to establish per-tier timing. Includes the runs that
  triggered `torch.OutOfMemoryError` on the large tier at
  `NUM_SAMPLE >= ~20000` (`calib3_large_*.out`, `calib4_large_{24000,28000}.out`)
  — this is the primary evidence for the memory-ceiling finding disclosed in
  the paper's `tab:purerl` caption.
- `scripts/` — `run_prod.sh`, `run_sweep.sh`, `run_large.sh` (the exact driver
  commands used) and `analyze.py`/`parse.py` (parses the `.out` logs and
  reproduces every number in `tab:purerl`, using this codebase's own
  `src/eval/stats.py::pairwise_wilcoxon` for the paired significance tests).
  Re-run with:
  ```
  cd logs/purerl_experiment_20260715/scripts
  /home/quangnm/projects/ArcRoute/.venv/bin/python analyze.py
  ```
- `prod_driver.log`, `sweep_driver.log`, `large_driver.log` — timestamps for
  when each stage of the run started/finished.

## Checkpoints used (for reproducibility if re-run later)

| Tier | Checkpoint |
|---|---|
| small | `outputs/checkpoints/grpo_reload_fix/small/epoch=044.ckpt` |
| medium | `outputs/checkpoints/grpo_reload_fix/medium/epoch=037.ckpt` |
| large | `outputs/checkpoints/grpo_reload_fix/large/epoch=041.ckpt` |

Note: `logs/`, `outputs/`, and `*.out` are gitignored in this repo (see
`.gitignore`) — this archive is local-filesystem persistence only, not
git-tracked. If these checkpoints or `data/test_bm/*.npz` are ever
regenerated, this archive's numbers should be re-verified against a fresh run
before being trusted for a submission.
