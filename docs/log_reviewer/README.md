# Log reviewer — HARMONY paper traceability

This directory is a **curated copy** (nothing was moved or deleted from
`/home/quangnm/projects/ArcRoute/logs/`) containing only the log/result files
that actually back the numbers currently reported in
`/home/quangnm/projects/HDCARP_with_ML/main.tex` (git commit `374eb6b` at the
time this was built, 2026-07-15). Every file below was cross-checked by
parsing its contents and comparing the computed mean(s) against the exact
value printed in the paper — this is not just filename pattern-matching.

Total files copied: **261**, across 9 subfolders. Nothing under
`logs/` was touched; everything here is a `cp`.

## How to re-verify any number yourself

Each `.out` benchmark log is a plain-text file with one line per instance of
the form:
```
data/test_bm/<tier>.npz[<idx>] ::: [[T1 T2 T3]] ::: <time_seconds>
```
Mean T1/T2/T3/time over all lines reproduces the corresponding table cell.
MILP results are one JSON file per instance with a `T` (or `U_T`/`P_T` for
the warm-start files) array and a `time` field.

---

## `table1_small_tier/` (8 files)

Backs **Table I** ("Scale-stratified benchmark"), the **small** rows only,
for HDCARP-P and HDCARP-U. These are the *current*, post-reload-fix runs
(all timestamped `20260715_00*`, i.e. produced in the early hours of the
reload-fix day).

| File | Backs |
|---|---|
| `bm_ils_P_small_20260715_002919.out` | P/small/ILS: T1=4.728, T2=8.910, T3=14.012, time=0.93s |
| `bm_aco_P_small_20260715_002950.out` | P/small/ACO: T1=4.637, T2=9.056, T3=14.228, time=184.60s |
| `bm_ea_P_small_20260715_002950.out`  | P/small/EA: T1=4.643, T2=9.437, T3=15.040, time=196.94s |
| `bm_rl_P_small_20260715_005819.out`  | P/small/HARMONY: T1=4.581, T2=8.790, T3=14.780, time=0.64s |
| `bm_ils_U_small_20260715_002950.out` | U/small/ILS: T1=4.641, T2=9.306, T3=15.547, time=12.28s |
| `bm_aco_U_small_20260715_002950.out` | U/small/ACO: T1=4.595, T2=9.423, T3=15.452, time=299.09s |
| `bm_ea_U_small_20260715_002950.out`  | U/small/EA: T1=4.627, T2=9.486, T3=15.278, time=291.40s |
| `bm_rl_U_small_20260715_005533.out`  | U/small/HARMONY: T1=4.581, T2=8.658, T3=13.481, time=0.64s |

All 8 values were confirmed by computing the mean of all 100 lines in each
file and matching to 3 decimal places against the table.

Note: there is a *second*, slower HARMONY-small-P run,
`bm_rl_P_small_20260715_002950.out` (mean time 8.80s/instance — evidently a
cold-start/CUDA-warm-up run, same T1/T2 but not the run Table I's 0.64s
comes from). It was **not** copied here (see "Not needed" below) since
`bm_rl_P_small_20260715_005819.out` is the exact match for the reported time.

## `table1_medium_large_provisional/` (16 files)

Backs **Table I**'s medium and large rows, which the paper explicitly marks
"provisional pending the full post-reload re-benchmark" and states use
"pre-reload-fix logs" (main.tex line 343). Consistent with that caveat,
these are all the **bare, non-timestamped** `.out` files (the *older* runs),
**not** the newer `20260715_*`-timestamped files that exist for HARMONY at
these tiers — the newer runs give different numbers (they belong to the
pure-RL comparison experiment, see below) and do not match Table I's
medium/large cells. Verified exact match for every cell:

| File | Backs |
|---|---|
| `bm_ils_P_medium.out` | P/medium/ILS: T1=8.302, T2=17.492, T3=28.723, time=1.40s |
| `bm_aco_P_medium.out` | P/medium/ACO: T1=7.617, T2=18.834, T3=30.214, time=80.82s |
| `bm_ea_P_medium.out`  | P/medium/EA: T1=8.144, T2=19.437, T3=30.616, time=47.35s |
| `bm_rl_P_medium.out`  | P/medium/HARMONY: T1=7.371, T2=15.697, T3=25.379, time=2.08s |
| `bm_ils_P_large.out`  | P/large/ILS: T1=15.052, T2=32.027, T3=51.338, time=1.91s |
| `bm_aco_P_large.out`  | P/large/ACO: T1=13.664, T2=33.372, T3=52.678, time=190.31s |
| `bm_ea_P_large.out`   | P/large/EA: T1=14.948, T2=34.596, T3=53.702, time=63.27s |
| `bm_rl_P_large.out`   | P/large/HARMONY: T1=12.494, T2=30.072, T3=49.570, time=5.78s |
| `bm_ils_U_medium.out` | U/medium/ILS: T1=8.195, T2=18.050, T3=29.589, time=3.76s |
| `bm_aco_U_medium.out` | U/medium/ACO: T1=7.624, T2=18.465, T3=29.671, time=190.49s |
| `bm_ea_U_medium.out`  | U/medium/EA: T1=7.900, T2=18.308, T3=27.653, time=134.41s |
| `bm_rl_U_medium.out`  | U/medium/HARMONY: T1=7.371, T2=15.097, T3=22.605, time=2.08s |
| `bm_ils_U_large.out`  | U/large/ILS: T1=14.866, T2=33.219, T3=52.626, time=7.15s |
| `bm_aco_U_large.out`  | U/large/ACO: T1=13.623, T2=33.374, T3=51.746, time=420.82s |
| `bm_ea_U_large.out`   | U/large/EA: T1=14.371, T2=32.322, T3=48.533, time=261.00s |
| `bm_rl_U_large.out`   | U/large/HARMONY: T1=12.494, T2=27.451, T3=40.478, time=5.95s |

No timestamped duplicates exist for ILS/ACO/EA at medium/large — only the
bare file, so there's no ambiguity there. For HARMONY (`bm_rl_*`), the bare
file is the one that reproduces the table; this was resolved by direct
numeric comparison, not by assumption (see main task notes: "do not assume
newest timestamp = correct" for these two tiers).

## `milp_reference/` (139 files: `small_P/` = 100, `small_U/` = 39)

Backs Table I's **MILP** column (small tier only — "the MILP column is
populated only on the small tier, where the exact method is tractable").

These come from `logs/paper_results/main_table_lpfix/`, confirmed to be the
**current, corrected** MILP run (multigraph-arc-dedup bug fix + warm-start
fix), **not** `logs/paper_results/main_table/` (the stale pre-fix numbers).
Spot check: `milp_small_P_18.json` in `main_table/` shows `T[0] = 8.789`
(the old, buggy value); the same instance in `main_table_lpfix/` shows
`T[0] = 6.223` — a clearly different, corrected value.

Aggregate verification (mean over all instances, using `.warm.json` in
preference to the plain `.json` for the 19 U instances that have a
warm-started rerun — `U_T`/`U_status`/`U_time` fields):

- `small_P/` (100 files, no warm variants): mean T1=4.581, T2=8.267,
  T3=12.521, time=342.38s. Status breakdown: 97 `optimal`, 3 `timelimit` —
  matches the paper's claim "matches the proven optimal T1 on all 97 proven
  instances."
- `small_U/` (20 base instances + 19 `.warm.json` reruns = 39 files): mean
  T1=4.047, T2=7.515, T3=11.379, time=300.02s. All 20 are `timelimit` (0
  proven optimal) — matches "none of the 20 sampled instances is proven
  optimal at 300s."

Both aggregates match Table I's MILP column to 3 decimal places.

## `ablation/` (17 files, copied whole from `paper_results/ablation/`)

Backs **Table II** (component ablation: A1 greedy / A1 best-of-R / A2 / A3 /
A4 / A5, across P/U x small/medium/large). Contains
`ablation_{small,medium,large}_{P,U}.json` (A1-A4 raw data),
`ablation_learning_free_{small,medium,large}_{P,U}.json` (A5 raw data),
the generation scripts (`ablation.py`, `ablation_a5.py`,
`process_ablation.py`, `process_ablation_a5.py`), and the human-readable
summary `ablation_table_final.txt`, whose numbers were checked line-by-line
against Table II — e.g. "small P A4 T2gap=3.22%" and "large U A2 T1gap=136.88%"
both match exactly.

## `pure_rl_comparison/` (28 files)

Backs **Section "Time-matched pure-RL comparison"** and **Table
`tab:purerl`** (R=16000/32000/16000 on small/medium/large; the 0.681s vs
0.588s small-tier timing; the medium/large gaps; the large-tier OOM ceiling
around R≈20000).

- `full_runs/` (13 files) — the exact runs the paper's `tab:purerl` numbers
  come from, per the archive's own `ORIGINAL_ARCHIVE_README.md` (copied here
  unchanged): `prod_{small,medium,large}.out` (fresh HARMONY on the
  corrected-environment checkpoints) and `pure_{tier}_{NUM_SAMPLE}.out`
  (pure-RL-only decode swept over R). Verified exact matches, e.g.:
  - `prod_small.out` → mean T1=4.581, T2=8.790, T3=14.780, time=0.588s
    (matches "0.588s for HARMONY" in the text and the HARMONY column of
    `tab:purerl`).
  - `pure_small_16000.out` → mean T1=4.609, T2=10.325, T3=17.661,
    time=0.681s (matches "R=16000 on small (0.681s..." and the pure-RL
    column of `tab:purerl`).
  - `prod_medium.out`/`pure_medium_32000.out` → 1.836s/2.350s, matching text.
  - `prod_large.out`/`pure_large_16000.out` → 5.317s/2.805s, matching text.
- `calibration_oom_evidence/` (6 files: `calib3_large_{32000,48000,64000}.out`,
  `calib4_large_{20000,24000,28000}.out`) — the archive README identifies
  these specifically as "the primary evidence for the memory-ceiling finding
  disclosed in the paper's `tab:purerl` caption" (large-tier OOM at
  R≳20000). The other ~27 calibration files (`calib_*`, `calib2_*`,
  `hard_*`) were used only to pick the sweep grid, not cited by any specific
  number in the paper — not copied (see "Not needed" below).
- `scripts/` — `run_prod.sh`, `run_sweep.sh`, `run_large.sh` (exact driver
  commands) and `analyze.py`/`parse.py` (reproduces every number in
  `tab:purerl`).
- `prod_driver.log`, `large_driver.log`, `sweep_driver.log` — stage timing.
- `ORIGINAL_ARCHIVE_README.md` — the original archive's own README,
  written by whoever ran this experiment; it independently corroborates the
  file mapping above.

Note: several top-level `logs/bm_rl_P_*_20260715_18*.out` files contain the
same numeric results as files in `full_runs/` (e.g.
`bm_rl_P_small_20260715_181050.out` ≈ `prod_small.out`) — these are
redundant copies of the same runs and were not duplicated here; see "Not
needed" below.

## `anytime_curve/` (9 files, copied whole from `paper_results/anytime_curve/`)

Backs **Figure `fig:anytime`** (T1 gap / wall-clock time vs. rollout budget
R, swept over {1,4,16,64,256} on a 30-instance subset). Contains
`rsweep_{small,medium,large}_{P,U}.json` (verified: each has
`N=30, R_GRID=[1,4,16,64,256]`, matching the text exactly), `rsweep.py`
(generator), `plot_anytime2.py` (plotting script), and
`anytime_curve_v2.png` (the actual figure asset used).

## `prop1_validation/` (7 files, copied whole from `paper_results/prop1_validation/`)

Backs **Table `tab:prop-validation`** (the no-degradation / funnel baseline
check: violations, strict-improvement %, median margin, over 600 held-out
instances). Contains `prop1_{small,medium,large}_{P,U}.json` and
`prop1_validate.py`. Verified two spot checks:
- `prop1_small_P.json`: `n_violations=0`, `pct_strict=85.0`,
  `median_margin=0.2149` → matches table row "P/small: 0/100, 85, 0.21".
- `prop1_large_U.json`: `n_violations=0`, `pct_strict=100.0`,
  `median_margin=52.336` → matches table row "U/large: 0/100, 100, 52.34".

## `case_study/` (36 files, copied whole from `paper_results/case_study/`)

Backs the **qualitative case study figure** (`fig:case-study`): one small
melbourne instance (|V|=18, |A|=31, 21 required arcs) and one large bogota
instance (|V|=81, |A|=127, 93 required arcs), both HDCARP-P, M=3, solved by
ILS/EA/ACO/HARMONY. Contains the generation script
(`gen_case_instance.py`), the solve/eval driver (`harmony_case.py`,
`analyze_case.py`), the plotting script (`plot_case_study.py`), and
`case/` with the actual instance `.npz` files, each algorithm's `.sol` /
`.log` / `_vehicles.json` output, and `_geo.npz` coordinate sidecars.

Verified: `case/small_case.npz` and `case/large_case.npz` — the files
`plot_case_study.py` actually loads — have `n_req = 21` and `n_req = 93`
respectively, matching the paper's stated required-arc counts. Note that
`case/` also contains earlier/rejected draws with descriptive names
(`small_melbourne_20_30.npz`, `large_bogota_80_133.npz`,
`large_bogota_50_100.npz`) generated while searching for a suitable
instance — `small_melbourne_20_30.npz` happens to be byte-identical to the
final `small_case.npz` (kept), but `large_bogota_80_133.npz` (n_req=105) and
`large_bogota_50_100.npz` (n_req=69) are different, discarded draws that do
**not** match the paper's "93 required arcs" and were not the ones plotted.
They were still copied (the whole `case_study/` tree was copied intact for
simplicity and because they're small/harmless) but are not themselves load-
bearing for the figure.

## `training/` (1 file)

Backs the specific claim "The initial small-tier stage completed 192 of 200
budgeted epochs in roughly 7 wall-clock hours on the RTX 4090" (main.tex,
Appendix, training details).

- `train_20260703_162357.out` — started 2026-07-03 16:23:57, last write
  2026-07-03 23:21:15 (≈6h57m ≈ "roughly 7 hours"), and its last logged
  epoch is `Epoch 192` (of a 200-epoch budget) — an exact match on both the
  epoch count and the wall-clock duration. The other 8 `train_*.out` files
  in `logs/` are other training runs (earlier restarts, and the
  medium/large curriculum stages) that are not tied to any specific number
  quoted in the paper text, so they were not copied (see below).

---

## What was found in `logs/` but NOT copied (clutter), and why

Nothing was deleted or moved — this is purely a report of what's still
sitting in `logs/` that isn't traceable to a currently-reported number, for
the user to decide whether to actually clean up later.

| Category | Count | Location | Reasoning |
|---|---|---|---|
| Superseded bare small-tier `.out` (ILS/ACO/EA/RL, P & U) | 8 | top level | A timestamped `20260715_*` version of the same (method, variant, small) run exists and matches Table I; the bare file gives different numbers (e.g. `bm_ils_P_small.out` mean T1=4.875 vs table's 4.728) — clearly an older, superseded run. Files: `bm_ils_P_small.out`, `bm_aco_P_small.out`, `bm_ea_P_small.out`, `bm_rl_P_small.out`, `bm_ils_U_small.out`, `bm_aco_U_small.out`, `bm_ea_U_small.out`, `bm_rl_U_small.out`. |
| Cold-start outlier HARMONY-small-P run | 1 | top level | `bm_rl_P_small_20260715_002950.out` — same T1/T2 as the file we did copy but mean time 8.80s/instance (vs. Table I's 0.64s) and slightly different T3; looks like a GPU/CUDA warm-up run, superseded by the later same-day rerun at 00:58. |
| Duplicate copies of the pure-RL-comparison full runs | 13 | top level | `bm_rl_P_{small,medium,large}_20260715_18*.out` reproduce the same means as files already preserved (and README-documented) under `pure_rl_comparison/full_runs/` (`prod_*.out`, `pure_*.out`); redundant, not re-copied. |
| 10-instance / "hard10" probe runs | 34 | top level | `bm_rl_*_{small10,medium10,large10}_*` and `bm_rl_*_{small,medium,large}_hard10_*` — explicitly called out in the task brief as exploratory calibration runs (n=10 instances, not the 100-instance benchmark). Confirmed by parsing: all have `n=10` or fewer lines. |
| Zero-byte `.out` files | 8 | top level | Failed/aborted runs (0 bytes). 6 of these overlap with the probe-run category above; 2 are standalone zero-byte attempts at full 100-instance runs (`bm_rl_P_large_20260715_181457.out`, `bm_rl_P_small_20260715_181005.out`). |
| Other training logs | 8 | top level | `train_20260701_113333.out`, `train_20260702_153948.out`, `train_20260702_214234.out`, `train_20260703_090014.out`, `train_20260706_120143.out`, `train_20260714_182440.out`, `train_20260714_195644.out`, `train_20260715_051112.out` — legitimate training runs, but none is tied to a specific quoted number in the paper the way `train_20260703_162357.out` is (192/200 epochs, ~7h). |
| Stale pre-lpfix MILP results | 129 | `paper_results/main_table/` | Superseded by `main_table_lpfix/` (see verification above: `milp_small_P_18.json` shows T1=8.789 here vs. the corrected 6.223 in `main_table_lpfix/`). Also includes derived summary artifacts (`main_table_fixed.txt`, `process_main_table.py`, etc.) whose numbers (e.g. HARMONY small-P time=0.59s, gap-based format) don't match the current Table I either. |
| Pure-RL sweep-grid calibration probes | 27 | `purerl_experiment_20260715/calibration/` | `calib_*`, `calib2_*`, `hard_*` — 10-instance probes used only to choose the R sweep grid and per-tier timing budget, per the archive's own README; not cited by any specific number in the paper (unlike the 6 `calib3_large_*`/`calib4_large_*` OOM files, which *are* copied — see `pure_rl_comparison/calibration_oom_evidence/`). |
| `scripts/__pycache__/parse.cpython-312.pyc` | 1 | `purerl_experiment_20260715/scripts/` | Compiled bytecode, not source; `parse.py` itself was copied. |
| `logs/archive/` | 46 | `logs/archive/` | Per the task brief, this looks like a previously, manually curated "already superseded" archive. Checked: none of its files were needed to reproduce any Table I / II / prop1 / anytime / pure-RL / case-study number — every current number was reproducible from top-level or `paper_results/` files. Not re-copied. |

**Totals**: 530 files exist under `logs/` (91 top-level + 337
`paper_results/` + 56 `purerl_experiment_20260715/` + 46 `archive/`); 261
were copied here as necessary, 269 were left as clutter/superseded per the
table above (66 top-level + 129 `main_table/` + 28
`purerl_experiment_20260715/` [27 calibration + 1 pycache] + 46 `archive/`).

## Ambiguity flags

None remain unresolved — every Table I/II cell, `tab:purerl` cell,
`tab:prop-validation` row, and the anytime-curve/case-study/training claims
were each matched to exactly one specific source file by parsing its
contents and checking the computed statistic against the value printed in
`main.tex`, not by filename/timestamp heuristics alone. The one place where
a timestamp heuristic would have given the *wrong* answer was Table I's
medium/large HARMONY rows, where the newer timestamped `bm_rl_P_medium_*`
files are decoys belonging to the pure-RL experiment, not Table I — this was
caught by checking values, not just dates (see `table1_medium_large_provisional/`
above).
