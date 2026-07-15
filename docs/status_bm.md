# Benchmark status — RL vs ILS (HDCARP P/U)

Tracks the RL-vs-ILS comparison so results don't need to be re-run to check
where things stand. Update this file whenever a new benchmark config is run.

## CURRENT — paper-reported results (as of 2026-07-11)

This is the authoritative, up-to-date benchmark backing the paper's
`tab:planned` (main results table). Everything below "Benchmark:
`data/test_bm` (paper benchmark set)" is an earlier round of dev-history
benchmarking (different checkpoints, no K-candidate class-1 mechanism, no
Horner-scalar fix) and is superseded — kept only for archival context, not
current numbers.

**Config**: HARMONY = full pool-funnel pipeline (`K=3` class-1 candidates
on HDCARP-P, `R=150`, production `bm.sh` defaults; HDCARP-U answered via the
P-anchored `UPGRADE_TO_U=1` pipeline). Checkpoints: small tier →
`outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt`; medium/large tiers →
`outputs/checkpoints/curriculum_medium/last.ckpt`. ILS/EA/ACO use `bm.sh`
defaults (`max_iter=200`/`n_epoch=100 n_population=200`/`n_epoch=100 n_ant=50`).
100 held-out test-city instances per (tier, variant) cell, `data/test_bm/{small,medium,large}.npz`.

**2026-07-11 fix**: `src/solvers/meta.py` used to rank solutions via a
scalar Horner encoding of `(T1,T2,T3)` that could silently misorder
near-tied solutions (confirmed on real instances, ~6% of spot-checked runs
affected). Fixed to exact componentwise/tuple lexicographic comparison —
see commit `e28bfe8`. ACO and HARMONY were verified unaffected (byte-identical
before/after); **ILS and EA numbers below are post-fix reruns**. The same
class of bug existed in `src/trainers/grpo.py`'s checkpoint-selection metric
(`val/lex_best`, the literal `ModelCheckpoint` monitor) — fixed for future
training runs (`LexBestCheckpoint`, same commit), but the currently deployed
`grpo_osm_bac3/epoch=159.ckpt` was selected under the old flawed metric and
was **not** retrained/re-verified (no surviving training logs; accepted as a
known risk — see the paper's Limitations).

Gaps below are mean gap-to-best-found across ILS/ACO/EA/HARMONY on the same
100 instances (matches the paper's `tab:planned` exactly); times are mean
seconds/instance.

| Variant | Tier | Metric | ILS | ACO | EA | HARMONY |
|---|---|---|---|---|---|---|
| P | small | T1/T2/T3 gap % | 5.98 / 15.58 / 11.15 | 1.73 / 15.93 / 19.00 | 3.22 / 24.17 / 24.44 | 0.00 / 1.31 / 4.40 |
| P | small | time (s) | 0.79 | 41.87 | 47.68 | 0.59 |
| P | medium | T1/T2/T3 gap % | 12.02 / 14.44 / 16.42 | 3.19 / 23.37 / 22.40 | 9.98 / 27.65 / 24.69 | 0.00 / 1.85 / 2.28 |
| P | medium | time (s) | 1.40 | 80.82 | 47.35 | 2.08 |
| P | large | T1/T2/T3 gap % | 20.68 / 9.89 / 10.63 | 9.39 / 15.08 / 13.87 | 19.50 / 19.39 / 16.12 | 0.00 / 2.68 / 5.68 |
| P | large | time (s) | 1.91 | 190.31 | 63.27 | 5.78 |
| U | small | T1/T2/T3 gap % | 3.47 / 15.19 / 18.43 | 1.29 / 16.90 / 30.71 | 1.34 / 16.24 / 21.30 | 0.00 / 1.89 / 3.51 |
| U | small | time (s) | 2.21 | 96.07 | 89.04 | 0.59 |
| U | medium | T1/T2/T3 gap % | 10.69 / 20.15 / 31.43 | 3.29 / 23.02 / 33.05 | 6.82 / 22.58 / 22.98 | 0.00 / 0.54 / 0.78 |
| U | medium | time (s) | 3.76 | 190.49 | 134.41 | 2.08 |
| U | large | T1/T2/T3 gap % | 18.83 / 21.96 / 31.23 | 8.96 / 23.34 / 30.69 | 14.80 / 18.87 / 22.00 | 0.00 / 0.76 / 1.08 |
| U | large | time (s) | 7.15 | 420.82 | 261.00 | 5.95 |

HARMONY's mean T1 gap is 0.00% in every cell; paired Wilcoxon signed-rank
tests separate it from ILS/EA/ACO in all 18 comparisons at p ≤ 1.7e-7 (large
tiers saturate at p = 3.9e-18 against every baseline).

**Canonical logs** (renamed to drop timestamps — exactly one file per
(solver, variant, tier) cell now; everything else that used to live in
`logs/bm_*.out` was moved to `logs/archive/` rather than deleted):

```
logs/bm_{ils,ea,aco,rl}_{P,U}_{small,medium,large}.out   # rl.out = HARMONY
```

Regenerate with (ILS/EA only need rerunning if `src/solvers/meta.py` or the
instance data changes again — ACO/HARMONY logs are independently verified
unaffected by the 2026-07-11 fix and don't need rerunning):

```bash
for tier in small medium large; do for v in P U; do
  DATA_DIR=data/test_bm/${tier}.npz VARIANT=$v SOLVER=ils bash scripts/bm.sh
  DATA_DIR=data/test_bm/${tier}.npz VARIANT=$v SOLVER=ea  bash scripts/bm.sh
done; done
```

### Other paper tables/figures — raw data + scripts saved under `logs/paper_results/`

The main table above is backed by `logs/bm_*.out` (safe, see above). The
paper's other three results tables/figures were originally computed via
one-off scripts in a session scratchpad (not part of this repo's normal
tree, and not guaranteed to survive between sessions); their raw outputs and
the scripts that produced them have been copied into this repo under
`logs/paper_results/` so they don't need to be regenerated from scratch.
`logs/` is gitignored, so this is on-disk only, not committed — back it up
separately if you care about surviving a clone/reinstall.

| Paper artifact | Directory | Contents |
|---|---|---|
| Table 4 (Empirical validation, `tab:prop-validation`) | `logs/paper_results/prop1_validation/` | `prop1_{tier}_{variant}.json` (6, raw per-instance margins) + `prop1_validate.py` |
| Table 6 (Ablation, `tab:ablation-planned`) | `logs/paper_results/ablation/` | `ablation_{tier}_{variant}.json` (6) + `ablation_table_final.txt` (processed) + `ablation.py`/`process_ablation.py` |
| Figure 4 (anytime curve, `fig:anytime`) | `logs/paper_results/anytime_curve/` | `rsweep_{tier}_{variant}.json` (6, R-sweep raw) + `rsweep.py`/`plot_anytime2.py` + the rendered PNG |
| Figure 5 (case study, `fig:case-study`) | `logs/paper_results/case_study/case/` | the 2 case instances (`.npz` + `_geo.npz` sidecar), every solver's `.sol`/`.log`/`_vehicles.json` (post-Horner-fix versions — `ils_large.sol` here has `T2=24.46, T3=38.84`, matching the paper; verified against the pre-fix `.OLD` backup that used to live in the scratchpad) + `gen_case_instance.py`/`harmony_case.py`/`analyze_case.py`/`plot_case_study.py` |
| Table 5 (main table, `tab:planned`) processing | `logs/paper_results/main_table/` | `process_main_table.py` + `main_table_fixed.txt` (verified ground truth, matches the paper's Table 5 exactly) |

None of these four need rerunning unless the underlying algorithm code
changes again (same rule as ILS/EA above: only rerun what the code change
actually touches — e.g. a HARMONY-only code change never requires
re-running ACO). If `src/solvers/meta.py` or `src/solvers/rl.py` changes,
re-derive the affected artifact with its script above rather than assuming
these saved JSONs are still valid.

---

## Benchmark: `data/test_bm` (paper benchmark set)

**(Everything from here down is superseded dev-history — different
checkpoints, no K-candidate mechanism, pre-dates the Horner-scalar fix.
Kept for archival context only; see "CURRENT" section above for the
paper's actual numbers.)**

- **Source**: `data/osm_test/{bogota,melbourne}` — held-out OOD test cities
  (not in `osm_train`'s 20 cities nor `osm_val`'s chandigarh/nairobi).
- **Construction**: 100 instances per tier in `data/test_bm/{small,medium,large}.npz`,
  stratified evenly across every `(city, bucket)` combination in that
  curriculum tier (`small`: `20_30`+`20_40`, 4 groups × 25; `medium`:
  `30_45`+`30_60`+`40_60`+`50_75`+`40_80`, 10 groups × 10; `large`:
  `50_100`+`80_120`+`80_133`, 6 groups × 16-17) — fixed seed 6868,
  reproducible. Same `.npz` schema as every other bundled file in this repo
  (`req`/`nonreq` object arrays, `C`/`M`/`n_req`/`tau`).
- **Why 100/tier**: fixed, fast, citable paper benchmark. RL/ILS are cheap
  enough to run at any scale (GPU ~0.3-0.7s/instance, ILS ~1-2.4s/instance
  with the `multiprocessing.Pool` this repo's CLIs already use); EA/ACO are
  heavier (measured ~48-98s/instance on `small.npz`, see results below) and
  MILP is sequential/single-threaded with no realistic full-scale story at
  all — 100 stratified instances per tier is the point that stays tractable
  for every solver (full `small.npz` run: a few minutes for EA/ACO, under a
  minute for RL/ILS) while remaining stable enough for a mean-level comparison.
- **Known relevant fixes in the current codebase** (context for interpreting
  results below — see `docs/grpo.md` and git history for full detail, not
  reproduced here): `BaseHCARP.ls_lex_guarded` (`src/solvers/meta.py`) keeps
  ILS/EA/ACO's local search from lex-worsening T1 under variant U; `RLHybHCARP`
  (`src/solvers/rl.py`) carries a greedy-anchor row plus a P-arm local-search
  option for variant U so its funnel can't underperform a plain greedy decode.
- **No dedicated RL-U checkpoint is maintained.** A from-scratch GRPO policy
  was trained for variant U (200 epochs) and benchmarked, but it generalized
  to OOD test cities noticeably worse than the P checkpoint (T1=4.980,
  T2=11.048, T3=15.440 on `small.npz` — worse than ILS-U). Root cause: it's a
  single training run with no analogue of the multi-round tuning (phase-masked
  advantage, decoder upgrade, oracle head) the P checkpoint has had, and U's
  `phase` is a training-time constant so the phase-masked-advantage fix has no
  analogue for it at all. Separately, `RLHybHCARP.upgrade_to_u` (opt-in flag)
  was measured to strictly beat that dedicated U checkpoint with **zero extra
  training** — see below — so the checkpoint and its training logs
  (`outputs/checkpoints/grpo_osm_u{,_fromP}/`, `lightning_logs/version_{4,5,6}`)
  were deleted rather than kept as a maintained artifact. **The practical "RL
  for variant U" answer is `RL-P + upgrade_to_u`, not a separately-trained
  policy.**

### Results — `small.npz` (100 instances)

| Config | T1 | T2 | T3 | time/inst | log |
|---|---|---|---|---|---|
| ILS-P (`max_iter=200`, default config) | 4.892 | 9.973 | 15.772 | 1.03s | `bm_ils_P_small_20260706_111028.out` |
| ILS-U (same config) | 4.730 | 9.827 | 15.385 | 2.37s | `bm_ils_U_small_20260706_111035.out` |
| RL-P (`grpo_osm_bac3/epoch=159.ckpt`, production `bm.sh` defaults: `HYBRID=1 NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01 TOPK_T1=100 ORACLE_HEAD=1`) | 4.581 | 8.798 | 14.928 | 0.31s | `bm_rl_P_small_20260706_111054.out` |
| **RL-U** (`grpo_osm_bac3/epoch=159.ckpt`, decodes as P + `UPGRADE_TO_U=1`, zero extra training) | **4.581** | **8.659** | **13.398** | 0.32s | `bm_rl_U_small_20260706_112210.out` |
| EA-P (`n_epoch=100 n_population=200`, default config) | 4.743 | 10.755 | 17.669 | 49.9s | `bm_ea_P_small_20260706_112415.out` |
| EA-U (same config) | 4.658 | 9.865 | 16.424 | 97.8s | `bm_ea_U_small_20260706_112847.out` |
| ACO-P (`n_epoch=100 n_ant=50`, default config) | 4.663 | 10.114 | 17.203 | 48.0s | `bm_aco_P_small_20260706_113727.out` |
| ACO-U (same config) | 4.644 | 10.670 | 17.771 | 83.6s | `bm_aco_U_small_20260706_114147.out` |

`UPGRADE_TO_U` post-processes RL-P's own winning route with a move-level
lex-safe local search restricted to variant U's looser precedence (any
individual move is applied only if it doesn't lex-worsen the touched
vehicle's own objective — see `src/utils/local_search.py::u_route_opt_lex_safe`),
kept only if the result is lex-better than the pure-P answer. Fires on 77/100
instances here, zero T1 regressions on any instance, and beats every other
row in this table on all three objectives simultaneously — including the
deleted dedicated-U-checkpoint result above.

**EA-U cleanly beats EA-P on all three objectives** (T1/T2/T3 all lower),
matching ILS-U's pattern and the U ⊇ P feasibility guarantee, at ~2x the
runtime (the `ls_lex_guarded` two-arm cost). **ACO-U is mixed**: T1 is
(marginally) better than ACO-P, but T2/T3 are *worse* — not a guard failure
(`ls_lex_guarded` only guarantees the accepted local-search result is
lex-non-worse than its own pre-LS input at each step, not that a full
independent stochastic search — different pheromone trails, different random
walk — averages better than P's run on every individual T_k; a lex-dominant
objective can legitimately prefer a slightly-better-T1-but-worse-T2/T3
outcome). First real evidence in this session that the "U ⊇ P" guarantee
shows up cleanly in the *feasible optimum* and in guarded local search, but
not automatically in every finite-budget heuristic's empirical mean —
ACO's pheromone-guided construction explores differently enough from ILS/EA
that this can happen. See the `medium.npz` results below — this pattern
gets *more* pronounced there, and a second one (ILS-U vs ILS-P) joins it.

### Results — `medium.npz` (100 instances)

- **RL checkpoint**: `outputs/checkpoints/curriculum_medium/epoch=110.ckpt` — a
  different checkpoint from the `small.npz` table above (`grpo_osm_bac3/epoch=159.ckpt`),
  trained/selected for the medium curriculum tier. Not directly comparable
  across tiers for that reason; RL-P/RL-U rows below use this checkpoint only.

| Config | T1 | T2 | T3 | time/inst | log |
|---|---|---|---|---|---|
| ILS-P (`max_iter=200`, default config) | 8.296 | 17.452 | 28.680 | 1.39s | `bm_ils_P_medium_20260707_002906.out` |
| ILS-U (same config) | 8.171 | 17.965 | 29.293 | 3.78s | `bm_ils_U_medium_20260707_002957.out` |
| RL-P (`curriculum_medium/epoch=110.ckpt`, production `bm.sh` defaults) | 7.380 | 16.008 | 25.950 | 2.64s | `bm_rl_P_medium_20260707_003021.out` |
| **RL-U** (same ckpt, decodes as P + `UPGRADE_TO_U=1`, zero extra training) | **7.380** | **15.276** | **22.635** | 2.61s | `bm_rl_U_medium_20260707_003455.out` |
| EA-P (`n_epoch=100 n_population=200`, default config) | 8.139 | 19.553 | 30.889 | 49.9s | `bm_ea_P_medium_20260707_003934.out` |
| EA-U (same config) | 7.884 | 17.986 | 27.034 | 139.1s | `bm_ea_U_medium_20260707_004425.out` |
| ACO-P (`n_epoch=100 n_ant=50`, default config) | 7.615 | 18.963 | 30.336 | 98.3s | `bm_aco_P_medium_20260707_005704.out` |
| ACO-U (same config) | 8.153 | 21.276 | 33.539 | 184.9s | `bm_aco_U_medium_20260707_010602.out` |

`UPGRADE_TO_U` fires on 95/100 instances here (vs 77/100 on `small.npz`),
still zero T1 regressions, T2 improves on 72/100 and T3 on 92/100 — again
beats every other row on all three objectives simultaneously.

**Two patterns from `small.npz` do NOT hold on `medium.npz`:**
- **ILS-U is *worse* than ILS-P on T2/T3** here (17.965/29.293 vs 17.452/28.680),
  only T1 is (marginally) better — the reverse of `small.npz` where ILS-U beat
  ILS-P cleanly on all three. Same `ls_lex_guarded` machinery; the guard only
  promises no lex-worsening of the *local-search step itself*, not that a
  looser-precedence search's different exploration path averages better at
  this larger, denser instance size.
- **ACO-U is now worse than ACO-P on all three objectives** (7.615/18.963/30.336
  vs 8.153/21.276/33.539), not just T2/T3 as on `small.npz` — same explanation
  (stochastic pheromone-guided search, no monotonic guarantee across a finite
  budget), just a starker instance of it at this tier.
- EA-U still cleanly beats EA-P on all three (matches `small.npz`), at ~2.8x
  runtime here (vs ~2x on `small.npz`).
- RL-U (P checkpoint + `upgrade_to_u`) still strictly dominates every other
  solver/variant on all three objectives, same as `small.npz` — the one
  pattern that has held at both tiers so far.

### Results — `large.npz` (100 instances)

- **RL not benchmarked here** — no RL checkpoint has been trained on
  large-tier instances, so RL was initially skipped in the main sweep below
  rather than reporting a misleading number. **Update**: an ad-hoc zero-shot
  check with the `small.npz` checkpoint (`grpo_osm_bac3/epoch=159.ckpt`,
  never trained on large-tier instances) was run afterward — see the
  dedicated subsection after the table below. It generalizes well and is
  included for reference, but it's still not a large-tier-trained checkpoint.

| Config | T1 | T2 | T3 | time/inst | log |
|---|---|---|---|---|---|
| ILS-P (`max_iter=200`, default config) | 15.053 | 31.877 | 51.296 | 1.89s | `bm_ils_P_large_20260707_012513.out` |
| ILS-U (same config) | 14.814 | 32.836 | 52.506 | 7.37s | `bm_ils_U_large_20260707_012529.out` |
| EA-P (`n_epoch=100 n_population=200`, default config) | 14.816 | 34.210 | 53.398 | 65.5s | `bm_ea_P_large_20260707_012618.out` |
| EA-U (same config) | 14.403 | 31.892 | 48.119 | 285.0s | `bm_ea_U_large_20260707_013308.out` |
| ACO-P (`n_epoch=100 n_ant=50`, default config) | 13.760 | 33.858 | 53.444 | 237.5s | `bm_aco_P_large_20260707_015858.out` |
| ACO-U (same config) | 16.881 | 41.689 | 65.564 | 479.8s | `bm_aco_U_large_20260707_022016.out` |

**Patterns from `medium.npz` continue, and one sharpens further:**
- **ILS-U vs ILS-P**: T1 (marginally) better, T2/T3 worse — same as `medium.npz`.
- **EA-U still cleanly beats EA-P on all three** (matches both smaller tiers),
  now at ~4.35x runtime (up from ~2x on `small.npz`, ~2.8x on `medium.npz`) —
  the two-arm `ls_lex_guarded` cost keeps growing with instance size.
- **ACO-U is now worse than ACO-P on all three objectives by a wider margin**
  than `medium.npz`: +22.7% T1, +23.1% T2, +22.7% T3 here vs. +7%/+12%/+10.6%
  on `medium.npz` (and only T2/T3 worse, T1 slightly better, on `small.npz`).
  The degradation is monotonically worsening with instance size — at this
  scale ACO's finite-budget pheromone search under variant U's looser
  precedence is being clearly outcompeted by its own P-variant run, not just
  failing to beat it. Not a correctness bug (`ls_lex_guarded`'s per-step
  guarantee still holds), but worth flagging if ACO-U is ever proposed as a
  default: at `large.npz` scale it is strictly worse to run, not just no
  better.
- ACO-P/ACO-U are now the slowest configs in the whole benchmark (237s/480s
  per instance) — EA has fallen behind ILS/RL but is still far cheaper than
  ACO at this tier.

#### RL zero-shot size-generalization check (ad-hoc, `curriculum_medium/epoch=178.ckpt` on `large.npz`)

Ran `RL-P`/`RL-U` on `large.npz` using `curriculum_medium/epoch=178.ckpt` — a
checkpoint trained on the *medium* curriculum tier, **never trained on
large-tier instances** (`50_100`/`80_120`/`80_133` buckets are still OOD for
it, though its training instances are closer in size to large-tier than the
small-tier checkpoint used in an earlier, since-discarded run of this check).
Purely a "how well does it generalize" probe, not a recommendation to ship
this checkpoint for the large tier.

| Config | T1 | T2 | T3 | time/inst | log |
|---|---|---|---|---|---|
| RL-P (`curriculum_medium/epoch=178.ckpt`, OOD size, production `bm.sh` defaults) | 12.636 | 30.209 | 50.203 | 2.38s | `bm_rl_P_large_20260707_095032.out` |
| **RL-U** (same ckpt, decodes as P + `UPGRADE_TO_U=1`) | **12.632** | **27.353** | **40.260** | 2.26s | `bm_rl_U_large_20260707_095500.out` |

`UPGRADE_TO_U` fires on 100/100 instances, 0 T1 regressions, T2 improves on
90/100 and T3 on 99/100 — same shape as `small.npz`/`medium.npz`.

- **RL-P beats every other P-variant solver on all three objectives**
  (12.636/30.209/50.203 vs ILS-P 15.053/31.877/51.296, EA-P
  14.816/34.210/53.398, ACO-P 13.760/33.858/53.444) despite being ~2.5x out
  of its training size range — a stronger showing than the earlier
  (since-discarded) small-tier-checkpoint zero-shot run, which won T1 but
  was the worst of the four on T3. A medium-trained checkpoint's deeper-tier
  (T2/T3) decisions degrade less under size OOD, consistent with it having
  seen instances closer in size to `large.npz` during training.
- **RL-U (this checkpoint + zero-training upgrade_to_u) still strictly
  dominates every ILS/EA/ACO row on all three objectives** — beats even
  ACO-P/EA-P's T1, and comfortably beats every U-variant row on T2/T3 (e.g.
  T3=40.260 vs ILS-U 52.506, EA-U 48.119, ACO-U 65.564). The pattern that
  held at `small.npz`/`medium.npz` ("RL-P + upgrade_to_u beats every
  metaheuristic on every objective") extends to `large.npz` even with a
  checkpoint that has never seen an instance this size — `upgrade_to_u`'s
  local search appears to compensate for exactly the T2/T3 degradation noted
  above.
- Runtime-wise RL is unchanged from its `small.npz`/`medium.npz` profile
  (~2-2.5s/inst) and remains by far the cheapest solver at this tier — 25-260x
  faster than ILS-U/EA-P and 2 orders of magnitude faster than ACO.
- This is *not* evidence that no large-tier-specific RL training is needed —
  a checkpoint trained on large-tier curriculum would likely still be a
  meaningfully better base for the T1/T2 legs even after upgrade_to_u.

#### Hardest-instance slice within `large.npz` (`|Ar| >= 100`, n=23/100)

`large.npz`'s `req` arrays range from 69 to 105 required arcs (buckets
`50_100`/`80_120`/`80_133`); 23 of the 100 instances hit `|Ar| >= 100`
(102 or 105). Re-aggregating every config above over just those 23 —
i.e. the largest, hardest slice of the tier — sharpens every pattern already
seen on the full 100:

| Config | T1 | T2 | T3 | time/inst |
|---|---|---|---|---|
| ILS-P | 16.005 | 34.182 | 54.868 | 2.03s |
| ILS-U | 16.202 | 35.972 | 57.312 | 8.58s |
| EA-P | 15.534 | 35.230 | 56.095 | 67.2s |
| EA-U | 15.447 | 33.580 | 50.727 | 328.9s |
| ACO-P | 14.613 | 35.538 | 57.431 | 288.2s |
| ACO-U | 18.832 | 46.593 | 70.896 | 590.1s |
| RL-P (`curriculum_medium/epoch=178.ckpt`, OOD size) | 13.230 | 32.837 | 55.801 | 2.77s |
| **RL-U** (+ upgrade_to_u) | **13.230** | **29.464** | **44.074** | 2.82s |

- **ILS-U flips to worse than ILS-P on all three objectives here, including
  T1** (16.202 vs 16.005) — on the full 100-instance `large.npz` average
  ILS-U was still marginally ahead on T1. The guard's per-step
  non-lex-worsening promise costs more than it recovers as instances get
  bigger/harder.
- **ACO-U's regression vs ACO-P widens further**: +28.8% T1, +31.2% T2,
  +23.4% T3 on this slice vs +22.7%/+23.1%/+22.7% on the full `large.npz` —
  the size-driven degradation trend keeps extending into the hardest
  instances, not just being an artifact of averaging over a mixed tier.
- EA-U still cleanly beats EA-P on all three (pattern holds), runtime ratio
  ~4.9x here (vs ~4.35x on the full tier).
- **RL-P is still the best T1 of any P-variant** (13.230) and its T3
  (55.801) sits between ILS-P (54.868, slightly better) and EA-P/ACO-P
  (56.095/57.431, both worse) — a much smaller OOD-size penalty on the
  deeper tiers than the earlier (since-discarded) small-tier-checkpoint run
  showed on this same slice, consistent with the medium-tier checkpoint
  generalizing better to large instances.
- **RL-U still wins all three objectives outright** (13.230/29.464/44.074,
  each the lowest in the table, beating EA-U's 15.447/33.580/50.727) —
  `upgrade_to_u` fires on 23/23 instances here (100%, matching the full
  tier's 100/100), 0 T1 regressions, T2 improves on 22/23 and T3 on 23/23.
  The local-search upgrade continues to close most of the remaining
  raw-decode gap even with a better-generalizing base checkpoint.

### Repro commands

```bash
SOLVER=ils VARIANT=P bash scripts/bm.sh     # DATA_DIR defaults to data/test_bm
SOLVER=ils DATA_DIR=data/test_bm/small.npz VARIANT=U bash scripts/bm.sh
CKPT=outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt DATA_DIR=data/test_bm/small.npz SOLVER=rl bash scripts/bm.sh
CKPT=outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt DATA_DIR=data/test_bm/small.npz SOLVER=rl UPGRADE_TO_U=1 bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/small.npz VARIANT=P bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/small.npz VARIANT=U bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/small.npz VARIANT=P bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/small.npz VARIANT=U bash scripts/bm.sh

# medium.npz — note the different RL checkpoint (curriculum_medium, not grpo_osm_bac3)
SOLVER=ils DATA_DIR=data/test_bm/medium.npz VARIANT=P bash scripts/bm.sh
SOLVER=ils DATA_DIR=data/test_bm/medium.npz VARIANT=U bash scripts/bm.sh
CKPT=outputs/checkpoints/curriculum_medium/epoch=110.ckpt DATA_DIR=data/test_bm/medium.npz SOLVER=rl bash scripts/bm.sh
CKPT=outputs/checkpoints/curriculum_medium/epoch=110.ckpt DATA_DIR=data/test_bm/medium.npz SOLVER=rl UPGRADE_TO_U=1 bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/medium.npz VARIANT=P bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/medium.npz VARIANT=U bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/medium.npz VARIANT=P bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/medium.npz VARIANT=U bash scripts/bm.sh

# large.npz — no checkpoint trained on large-tier instances
SOLVER=ils DATA_DIR=data/test_bm/large.npz VARIANT=P bash scripts/bm.sh
SOLVER=ils DATA_DIR=data/test_bm/large.npz VARIANT=U bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/large.npz VARIANT=P bash scripts/bm.sh
SOLVER=ea  DATA_DIR=data/test_bm/large.npz VARIANT=U bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/large.npz VARIANT=P bash scripts/bm.sh
SOLVER=aco DATA_DIR=data/test_bm/large.npz VARIANT=U bash scripts/bm.sh
# ad-hoc RL zero-shot size-generalization check (curriculum_medium checkpoint, OOD size)
CKPT=outputs/checkpoints/curriculum_medium/epoch=178.ckpt DATA_DIR=data/test_bm/large.npz SOLVER=rl bash scripts/bm.sh
CKPT=outputs/checkpoints/curriculum_medium/epoch=178.ckpt DATA_DIR=data/test_bm/large.npz SOLVER=rl UPGRADE_TO_U=1 bash scripts/bm.sh
```

(`scripts/bm.sh`'s RL defaults are now the production config above, and
`DATA_DIR` accepts a single `.npz` file directly — no more symlink wrapping
needed; see the script's own header comment for the full option list.)

## Other logs in this directory

- `train_*.out` — GRPO training run logs, see `outputs/lightning_logs/` for
  the corresponding metrics.
