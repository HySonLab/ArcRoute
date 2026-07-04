# GRPO — architecture, topology & current status

Ground truth as of **2026-07-03**, read directly from code (`src/trainers/grpo.py`,
`src/policy/*.py`, `src/env/env.py`, `src/train.py`) plus the latest training run
and OOD benchmark.

## 1. Problem being solved

HDCARP-P: fleet of M vehicles, arcs split into P=3 priority classes, class-k
arcs may only be serviced after ALL class-(k-1) arcs are done (linear
precedence). Objective: lexicographically minimize `(T1, T2, T3)` where
`T_k` = max completion time over class-k arcs across all vehicles.

## 2. Data flow (topology)

### 2.1 Shared building block: one decode step

Both training and inference bottom out in the same encoder → decoder → env
loop; what differs is what wraps around it (§2.2 vs §2.3).

```
.npz instance (data/osm_train|val|test)
        │  utils.ops.import_instance / iter_instances
        ▼
  TensorDict  { clss, demands, service_times, traversal_times, adj, num_vehicle }
        │  env.CARPEnv.reset()
        ▼
  td (state)  { current_node(B,M), used_capacity(B,M), completion(B,M),
                completion_at_arc(B,N+1), visited(B,1,N+1), action_mask, ... }
        │
        ▼                                                    ┌─────────────┐
  policy.encoder(td)  ──►  hidden (B,N+1,D)  ─────────────────┤ once/instance│
        │                                                    └─────────────┘
        ▼
  decoder.pre_decoder_hook → PrecomputedCache (glimpse_k/v, logit_k, graph_ctx)
        │
        ▼   ┌──────────────────────────── decode loop (while not done) ────────────┐
        │   │ JointMVContext(cache.node_embeddings, td) → veh_q (B,M,D)            │
        │   │   (bậc-3: individual query + masked cross-vehicle self-attention     │
        │   │    + fuse, §3/§8 — identical module at train and inference time)     │
        │   │ PointerAttention(veh_q, glimpse_k, glimpse_v, logit_k, mask)          │
        │   │        → logits (B,M,N+1)                                            │
        │   │ flatten → (B, M*(N+1)) joint action space                            │
        │   │ decode_strategy.step (greedy/sampling/evaluate) → flat action         │
        │   │ env.step(td): action = m*(N+1)+j → splits (vehicle m*, arc j*)        │
        │   │        updates current_node, used_capacity, completion,              │
        │   │        completion_at_arc[j*], visited; recomputes action_mask        │
        │   └─────────────────────────────────────────────────────────────────────┘
        ▼
  env.get_reward(td) : T_k = max over class-k arcs of completion_at_arc  (GPU, no CPU Scheduler)
```

Key point: the policy decides **(vehicle, arc) jointly at every step** — it is
not "decide arc order, then assign to vehicles" (that was the pre-GRPO-v3
design; see §7).

### 2.2 Training-time pipeline (`src/train.py` + `src/trainers/grpo.py`)

No funnel, no local search, no oracle head — those are all inference-only
additions in `src/solvers/rl.py` and are never invoked from the training
code path. Training only ever runs the §2.1 loop once per rollout, purely
to produce a gradient signal:

```
batch of B instances (curriculum "small": 20_30/20_40, M random in {2,3,5,7,10})
      │
      ▼  encoder (§2.1), once per instance
      ▼  batchify K=16 (group_size, POMO-style — instance-norm makes the K
      │  copies' embeddings identical, so this is free, no extra encode)
      ▼
┌──────────── rollout, torch.no_grad(), decode_type='sampling' ─────────────┐
│ §2.1 decode loop, B*K copies in parallel, decoder = current bậc-3 module  │
│ NO local search, NO oracle head → actions + reward (B,K,3) = (T1,T2,T3)  │
└────────────────────────────────────────────────────────────────────────────┘
      │
      ▼  centered_hier_rank_by_phase(T, gamma=0.5)  (§4, §7.2)
      │    per-level centered midrank across the K group, combined with
      │    geometric weights; each decode step only receives advantage from
      │    priority levels it could causally still influence (its phase)
      ▼  DAPO dynamic group filtering: all-K-tied groups → zero advantage,
      │  dropped before the gradient pass (train/skipped_frac)
      ▼
┌──────── mini-batch replay, WITH gradient, decode_type='evaluate' ─────────┐
│ replay the stored actions in chunks of mini_batch_size, fresh log-probs, │
│ REINFORCE surrogate (no PPO ratio/clip — θ is unchanged since rollout)   │
│ weighted by per-step phase-gathered advantage − entropy bonus            │
└────────────────────────────────────────────────────────────────────────────┘
      │
      ▼  gradient accumulated across chunks → one backward + AdamW step
      ▼  (periodic) eval step: greedy rollout on val set → val/T{k}_best,
         val/lex_best → drives ModelCheckpoint selection
```

### 2.3 Inference-time pipeline (current best config)

Checkpoint `epoch=159.ckpt`, `HYBRID=1 NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01
TOPK_T1=100 ORACLE_HEAD=1`.

`src/solvers/rl.py::RLHybHCARP`. Same encoder/decoder as §2.2 (weights
frozen, `eval()` mode) — everything below is inference-time search wrapped
around it, none of it changes what the policy learned:

```
.npz instance
      │
      ▼  encoder (§2.1), once per instance
      │
      ├──────────────────────── ARM A: pool funnel (P1, §9) ───────────────────┐
      │  batchify K=400 → §2.1 decode loop, decode_type='sampling'             │
      │  → eps_lex_select(eps=0.01, shortlist=25)   [near-best-T1 candidates]  │
      │  → ∪ top-100-by-raw-T1 (topk_t1=100)        [T1-dedicated candidates]  │
      │  → local search (intra/inter route) per candidate, keep only if       │
      │    strictly lex-better than pre-LS                                    │
      │  → strict-lex-best  =>  pool_best (T1,T2,T3)                          │
      └──────────────────────────────────────────────────────────────────────────┘
      │
      ├──────────────────── ARM B: oracle head (P2, §10, variant P only) ──────┐
      │  t1_oracle_argmin (`solvers/oracle.py`): Held-Karp + partition DP over │
      │    class-1 arcs → exact T1 optimum + witnessing per-vehicle prefix     │
      │    (+ optional exact tiebreak among T1-optimal ties)                   │
      │  → force prefix into decode (`_decode_forced`, bypasses the normal     │
      │    sampling entrypoint to inject specific actions)                     │
      │  → complete classes 2/3: oracle_r=64 sampled + 1 greedy                │
      │  → shortlist by T2-eps-band (oracle_shortlist=8) → local search        │
      │  → arm-internal pick (`t2_band_pick`)  =>  head_best (T1,T2,T3)        │
      │  (guard: n1 > max_n1=16 or infeasible → returns None, arm contributes  │
      │   nothing, pool_best passes through unchanged)                        │
      └──────────────────────────────────────────────────────────────────────────┘
      │
      ▼  oracle_union_pick(pool_best, head_best, eps2=0.02):
      │    T1 strictly better on one side → take it; T1 tied → eps-band on T2,
      │    then min T3
      ▼  re-evaluate the union winner via the float64 SolutionEvaluator
      │  (same precision path local search already uses — the T1 guarantee
      │   holds at full precision, not just decode-time float32)
      ▼
  output: (T1, T2, T3) + winning route
```

Summary: one encode, two parallel candidate-finding arms (one searches
broadly via sampling + local search, the other solves the T1 subproblem
exactly and lets the policy complete the rest), merged lexicographically,
then re-verified at full precision. Everything past the encoder/decoder is
inference-time compute, not learned weights — §2.2's training never sees
any of it.

## 3. Policy architecture

### Encoder (`src/policy/encoder.py`)

- `ARPInitEmbedding`: per-node features `[demand, clss, service_time,
  traversal_time]` → linear projection to `embed_dim` (depot gets its own
  linear layer).
- `GraphAttentionNetwork`: stack of `num_encoder_layers` (12 in the full
  config) `MultiHeadAttentionLayer`s — standard MHA + instance-norm + MLP
  feedforward + instance-norm, skip connections throughout (Kool et al.
  Transformer-encoder style).
- `MultiHeadAttention` has a **learnable per-head distance bias**
  (`dist_bias`, initialized -0.1): the Floyd-Warshall arc-to-arc adjacency
  matrix `adj` (normalized to `[0,1]` per instance) is added as a negative
  additive bias to attention logits, so nearby arcs attend to each other more
  by default — a spatial-locality inductive prior on top of content attention.
- Output: `hidden (B, N+1, D)` node embeddings, encoded **once per instance**
  regardless of `group_size` K (instance-norm is batch-independent, so K
  identical copies of one instance produce identical embeddings — GRPO
  exploits this to skip redundant encoding, see §4).

### Decoder (`src/policy/decoder.py`) — "Joint M×N" (GRPO v3) + bậc-3 upgrade (§8)

- `JointMVContext`: for each of the M vehicles builds a **local** feature
  `[node_embedding[current_node_m], remaining_capacity_m, completion_m /
  max_m(completion), dist(current_node_m, depot), phase_m/P, per-class
  remaining-arc-fraction_m (P values)]` (7 scalars + the D-dim node
  embedding for P=3), linearly projected to `embed_dim` → `veh_q_individual`
  ("h¹"). This individual query then passes through a **masked self-attention
  over the M vehicle tokens** (`veh_attn_qkv`/`veh_attn_out`, masked by
  `valid_veh = arange(M) < num_vehicle` so padded "ghost" vehicle slots in a
  mixed-fleet-size batch can't pollute real vehicles — see §8.2) to produce
  a **collaborative** query `veh_q_collab` ("h²"), and the two are fused
  (`self.fuse`, `Linear(2D→D)`) into the final `veh_q` used downstream. Fully
  vectorized over M — no Python loop. (Pre-bậc-3, `veh_q` was just the local
  feature projected directly, with no phase/class-progress features and no
  cross-vehicle signal — see §7.3/§8 for why that was a gap.)
- `PointerAttention`: standard glimpse/logit pointer mechanism, but scores
  the **M vehicle queries** against the **N+1 arc keys** in one batched
  matmul → `logits (B, M, N+1)`. Flattened to `(B, M*(N+1))` for the decoding
  strategy, so at every step the policy picks one `(vehicle, arc)` pair from
  a single categorical over the whole joint space.
- No `dynamic_embedding` beyond the static cache (`StaticEmbedding` — a
  placeholder returning 0); all step-varying information enters through
  `JointMVContext`'s vehicle-state features, not through re-projected node
  keys.
- Feasibility mask (`env.get_action_mask`, `src/env/env.py:89-143`): three
  layers combined — (1) arc already served, (2) `(m,j)` capacity would
  exceed 1.0, (3) precedence — arc's class > current global min unserved
  class (`c_min`, computed once per step from `td['clss']` and `visited`).
  Plus a depot idle-guard (an empty vehicle already at the depot can't
  re-select depot) and a nonexistent-vehicle guard for `m ≥ M_i` in the
  padded batch.

### Decoding strategies (`src/policy/decode_strategy.py`)

`greedy` (val/test), `sampling` (train, and inference-time K-rollout),
`evaluate` (replay externally-provided actions for GRPO's mini-batch
gradient pass — recomputes log-probs without re-sampling). Standard
temperature / top-k / top-p knobs exist but training doesn't use them
(`temperature=1.0`, `tanh_clipping=10` at train time per `src/train.py`, `0`
for the plain `AttentionModelPolicy` default).

### What the decoder sees, pre- vs post-bậc-3

Pre-bậc-3 (§6.1/§6.2's checkpoints), `JointMVContext`'s vehicle state was
only `[remaining_cap, completion_norm, dist_to_depot]` — no signal for "how
many arcs of the current/next priority class remain," no cross-vehicle
awareness at all. §7.3/§8 diagnosed both gaps; bậc 3 (§8, trained and
benchmarked, §8.5-8.6) adds class-progress features and cross-vehicle
self-attention to close them. §6.2's benchmarked checkpoints (v4-v6 in
`logs/status_bm.md`) predate this change; bậc-3's own benchmark (v7, and
v8 with the §9 funnel widening) is in §8.6/§9.

## 4. GRPO trainer (`src/trainers/grpo.py`)

Critic-free: **group-relative rank IS the advantage**, no value network.

### Rollout (`_train_step`)

1. Encode each of the B instances **once** (`policy.encoder(td0)`).
2. `batchify` the shared hidden K times (POMO-style, `k*B+b` layout) and roll
   out the full `B*K` group inside `torch.no_grad()` — no activation graph,
   so the big `B*K` forward is memory-cheap.
3. Compute the group advantage from the resulting `(B,K,3)` T-vectors (below).
4. **DAPO dynamic group filtering**: a group whose K rollouts are all
   T-tied gives an all-zero advantage (zero gradient) — such groups are
   dropped before the gradient pass; `train/skipped_frac` logs how much was
   skipped.
5. **Mini-batch replay with gradient**: chunks of `mini_batch_size // K`
   instances at a time, `decode_type='evaluate'` replays the stored actions
   to get fresh log-likelihoods at the current policy — since θ hasn't
   changed since the rollout, this is a REINFORCE surrogate, not a PPO
   ratio/clip (see §4's Loss subsection) — gradient accumulated across
   chunks, **one** optimizer step per training step
   (`automatic_optimization=False`).

This two-phase (no-grad full rollout, then grad-enabled mini-batch replay)
is what keeps memory bounded: encode-once-with-grad at B×K=1024 sequences on
120-arc instances OOM'd at ~22-23GB; this design peaks around 5.7GB.

### Advantage: hierarchical soft lexicographic rank

`centered_hier_rank_gpu(T, gamma=0.5)` (`grpo.py:82-99`):

- For each priority level `p` independently, compute a **centered midrank**
  over the K rollouts (`_centered_midrank`, tie-aware — exact ties get the
  averaged rank, so identical T-vectors get identical advantage).
- Combine levels with geometric weights `gamma^(p-1) / Σ gamma^q`.
- For P=3, `gamma=0.5` → **gradient share T1=57%, T2=29%, T3=14%**
  (K-independent, scales to any P without re-tuning).

This replaced an earlier "hard lex rank" (commit `0baf47b`) that was
mathematically equivalent to pure-T1 rank for continuous T-vectors — with
K=50 sampling, exact ties on T1 essentially never happen, so T2/T3 got zero
gradient under the old scheme. The soft rank fixed that but the advantage was
originally applied as **one scalar per whole trajectory**, uniformly across
every decode step regardless of which priority-class phase that step is in —
this has since been fixed by causal phase-masking, §7.

`centered_lex_rank` (the CPU/numpy reference version used as unit-test ground
truth for `centered_hier_rank_gpu`) now lives in `src/utils/ops.py`, not this
file — `trainers/grpo.py` never calls it directly (training only uses the GPU
path). It moved there because it's now *also* used by the classical
metaheuristics (`solvers/meta.py::EAHCARP`'s roulette-wheel parent selection
uses it instead of raw objective values, to avoid softmax collapsing to
near-one-hot — same "rank not raw magnitude" principle, different consumer).
Out of scope for this doc; see `solvers/meta.py::BaseHCARP.calc_obj` and
`EAHCARP.__call__` if touching that code.

### Loss

**Not a clipped PPO surrogate — pure REINFORCE at fixed θ.** The mini-batch
replay recomputes log-likelihoods with the *same* weights the rollout used
(no optimizer step happens between rollout and replay), so there's no
old-vs-new policy ratio to clip; `grpo.py`'s own module docstring notes this
explicitly ("opt.step after all chunks keeps theta fixed, so a PPO ratio
would be == 1"). The actual loss is a per-step advantage-weighted
log-likelihood: `surr = -((ll * adv_step).sum(-1) / valid_len).mean()`
(`adv_step` = the phase-gathered advantage, §7.7) `− entropy_lambda *
entropy`. Gradient-clipped by norm, `AdamW`, linear warmup → one-shot
cosine decay to `lr_min` (`configure_optimizers`). (This section previously
described a PPO-style clipped-ratio surrogate that doesn't match the code —
corrected after it was flagged during the oracle-in-training investigation,
§11.)

### Eval step

Greedy rollout, logs `val/T{k}_best` per level and a Horner-encoded
`val/lex_best` scalar (`T1*C² + T2*C + T3`, `C=1e3`) used by
`ModelCheckpoint(monitor="val/lex_best", mode="max")` for checkpoint
selection (`src/train.py:100-103`).

## 5. Training config & entry point

`src/train.py` wires `CARPEnv` + `AttentionModelPolicy` + `GRPO` +
Lightning `Trainer`, launched via `scripts/train.sh` (backgrounded via
`nohup`, logs to `logs/train_<timestamp>.out`). Full-scale defaults:
`max_epoch=1000`, `batch_size=2048` (script override), `embed_dim=128`,
`num_encoder_layers=12`, `num_heads=8`, fleet `M ∈ {2,3,5,7,10}` sampled per
instance, `variant=P`. Data: `data/osm_train` (20 cities) /
`data/osm_val` (chandigarh, nairobi — held out) / `data/osm_test`
(bogota, melbourne — held out), curriculum buckets optionally restrict to
a size subset (`--curriculum small|medium|large`, `utils.data_utils.CURRICULUM`).

## 6. Current status

### 6.1 Baseline run (pre-bậc-2): `outputs/lightning_logs/version_1`

**Run config** (`hparams.yaml`): `batch_size=512`, `group_size=16`,
`mini_batch_size=256`, `adv_gamma=0.5`, `lr=1e-4→1e-5` (cosine, 10k steps,
100 warmup), `entropy_lambda=0.02`, curriculum=`small`
(`train_buckets=['20_30','20_40']`). 2.5M trainable params.

**Convergence**: T1 val 5.81→4.17, T2 10.85→6.95, T3 14.38→12.59 over 200
epochs, but **plateaued after ~epoch 120** — epochs 120-199 oscillate in a
narrow band with no further net improvement. Best checkpoint by
`val/lex_best`: **epoch 183** (`outputs/checkpoints/grpo_osm/epoch=183.ckpt`,
`T1=4.166, T2=6.946, T3=12.586`). `last.ckpt` (epoch 199) is ~0.5% worse.

**OOD benchmark vs ILS** (`data/osm_test_small`, bogota/melbourne — cities
absent from both train and val; full detail + repro commands in
`logs/status_bm.md`):

| Config | T1 | T2 | T3 | gap vs ILS (T2/T3) | speedup vs ILS |
|---|---|---|---|---|---|
| ILS (classical baseline) | 5.012 | 10.056 | 15.804 | — | 1.0x |
| RL, raw K=100 sampling, strict lex select | 4.968-5.064 | 12.5-12.7 | 21.3-21.4 | +25-36% | 14-28x |
| RL + eps-lex select ("bậc 0") | 5.064 | 12.669 | 21.422 | +26.0% / +35.6% | 27.9x |
| RL + eps-lex shortlist + local search ("bậc 1", SL=10) | 4.893 | 11.210 | 18.074 | +11.5% / +14.4% | 19.9x |
| RL + eps-lex shortlist + local search ("bậc 1", SL=25) | 4.843 | 10.934 | 17.532 | +8.7% / +10.9% | 13.5x |

Val-set T2/T3 (6.95/12.59, cities close to training distribution) generalize
much worse to the true OOD test cities (12.5-12.7/21.3-21.4 pre-hybrid) than
T1 does (4.17→~5.0, only ~19% degradation vs T2's ~80%) — consistent with
T2/T3's much smaller gradient share (29%/14%) producing a less robust skill.

**Bậc 0 / bậc 1** are inference-only fixes, implemented in `src/solvers/rl.py`
(`eps_lex_select`, `RLHybHCARP`) and `scripts/bm.sh` (`HYBRID`, `SHORTLIST`,
`EPS` env vars) — no retraining. They roughly halve the T2/T3 gap and flip
the lexicographic win-rate against ILS from ~40% to ~59-67%, while staying
13.5-28x faster than ILS. They do **not** fully close the gap: only 13-16%
of instances Pareto-dominate ILS on all three objectives simultaneously, and
gains from tuning `shortlist`/`eps` further show diminishing returns against
shrinking speed advantage — this is the ceiling of an inference-only fix.
Full detail, all repro commands, and file provenance: `logs/status_bm.md`.

### 6.2 Bậc 2 — IMPLEMENTED, TRAINED, BENCHMARKED — hypothesis confirmed

Causal phase-masked advantage (§7) is implemented, code-reviewed (§7.7),
retrained to completion, and benchmarked against ILS on the same
`osm_test_small` OOD set as `logs/status_bm.md`. Same hyperparameters as the
§6.1 baseline (`MAX_EPOCH=200 BATCH_SIZE=512 MINI_BATCH_SIZE=256
GROUP_SIZE=16 CURRICULUM=small`, seed 6868) for a clean A/B — only the
advantage/loss code differs.

- Log: `logs/train_20260703_090014.out` (`Trainer.fit stopped: max_epochs=200
  reached`, clean finish, no NaN/crash across the full run). Lightning run:
  `outputs/lightning_logs/version_2`.
- Best checkpoint by `val/lex_best`: **`outputs/checkpoints/grpo_osm_phasemask/epoch=189_saved.ckpt`**
  (renamed from `epoch=189.ckpt` after training finished — same pattern as
  the pre-existing `_saved` file in the old `grpo_osm/` checkpoint dir).
  Val-set vs §6.1's `epoch=183.ckpt`: T1 4.166→4.223 (~flat, -1.4%), T2
  6.946→6.829 (**+1.7% better**), T3 12.586→10.222 (**+18.8% better**) —
  exactly the predicted shape: T1 untouched, T3 (the most noise-polluted
  level pre-fix) improves the most.
- **OOD test result (`logs/status_bm.md`, rows v4-v6)** — the real target
  metric:

  | Config | T1 | T2 | T3 | gap vs ILS (T2/T3) | Pareto-dom | speedup |
  |---|---|---|---|---|---|---|
  | ILS | 5.012 | 10.056 | 15.804 | — | — | 1.0x |
  | Bậc 1 SL25 (old arch, for reference) | 4.843 | 10.934 | 17.532 | +8.7%/+10.9% | 16.4% | 13.5x |
  | **Bậc 2 alone**, no local search | 5.101 | 11.935 | 18.098 | +18.7%/+14.5% | 4.0% | 29.9x |
  | **Bậc 2 + Bậc 1 funnel**, SL=10 | 4.901 | 10.720 | 15.820 | +6.6%/+0.1% | 20.6% | 22.8x |
  | **Bậc 2 + Bậc 1 funnel**, SL=25 | **4.850** | **10.473** | **15.496** | **+4.2%/-1.9%** | **25.8%** | 16.9x |

  Bậc 2 alone (no inference-time help at all) already matches bậc-1's T3 gap
  from the old architecture — confirms the fix improved the *policy itself*,
  not just something patchable at inference. Stacked with bậc 1: at SL=10,
  T3 gap closes to +0.1% (effectively tied with ILS); at **SL=25, T3 gap
  goes negative (-1.9%) — RL now beats ILS on T3 on average**, T1 beats ILS
  by -3.2%, T2 gap down to +4.2% (from +26.0% at the starting point),
  Pareto-dominance **25.8%**. This beats the old architecture's SL=25 config
  (v3) on *every* axis simultaneously while also being faster (16.9x vs
  13.5x) — confirms §7.4/§8's "stacking, not just re-selecting from the same
  weak pool" prediction. Unlike the old architecture (where SL 10→25 showed
  diminishing returns, `logs/status_bm.md`), SL 10→25 on the bậc-2
  checkpoint still shows a full-sized jump — SL=50 untried, may have more
  room.
- **Still not literal 100% Pareto-dominance** (25.8%, not 100%) — matches
  the calibrated expectation in §7.6 ("beats ILS on the large majority while
  staying an order of magnitude faster", not 100% dominance on every one of
  2858 instances) — though on *average* across T1/T2/T3 that bar is now met
  or exceeded. §8 ("bậc 3") is the next lever, now implemented and training
  (§6.3).
- **Root-cause analysis of the residual gap** (post-bậc-2): the 25.8%
  Pareto-dominance ceiling is not primarily a compute-asymmetry problem
  (RL too slow/shallow a search vs ILS's 200 iterations) — direct
  instance-level analysis of the benchmark data shows **60.1%** of
  instances are "mixed": RL wins one objective, ILS wins another, neither
  dominates. Both solvers optimize lex-style objectives that permit
  trade-offs across (T1,T2,T3); Pareto-dominance forbids trade-offs
  entirely. This is a metric-geometry mismatch between what's being
  optimized and what's being measured, not evidence either solver is
  weak — it reframes "≥99% Pareto-dominance" as an unrealistic bar for two
  lex-style optimizers being compared under a stricter-than-lex metric, and
  redirects effort toward improving T2 quality at the source (§8) rather
  than searching harder at inference (§6.3.1, §6.3.2 below — both explored
  and set aside for this reason).

### 6.3 Explored alternatives (rejected / refuted) — kept so they aren't re-investigated

**6.3.1 RL-warm-started ILS — tried, rejected as not worth the tradeoff.**
Idea: seed ILS's initial solution from the bậc-2+1 funnel-hybrid winner
(via `vehicles_to_action`) instead of ILS's own constructive heuristic, then
run a much smaller ILS iteration budget on top (`RLWarmILSHCARP` in
`src/solvers/rl.py`, `--warm_ils --ils_iters`). Measured result: "T1 not
worse than ILS" rose from ~81.4% → ~85.9% of instances, Pareto-dominance
25.8% → 27.0%, but speedup dropped 16.9x → 9.2x (needs `ils_iters=50` on top
of the hybrid funnel to get there). Explicit call: **the gain is too small
for the speed given up** — rejected in favor of pushing on §8 (improve the
policy itself) instead of leaning harder on inference-time search.

**6.3.2 "Gamma-misalignment" hypothesis — proposed, then refuted by data.**
Hypothesis: GRPO's soft `adv_gamma=0.5` (T1/T2/T3 gradient share 57/29/14%,
§4) is less T1-strict than ILS's near-exact lexicographic `calc_obj`
comparison, so RL might be systematically under-prioritizing T1 relative to
ILS — fix would be lowering `gamma` to sharpen T1's share further. Checked
against the actual benchmark data: RL **already beats ILS on both T1 and
T3** (v6: T1 -3.2%, T3 -1.9%) and its only remaining deficit is T2 (+4.2%).
Lowering `gamma` shifts gradient share *away* from T2 toward T1 — it would
sharpen an objective that isn't the binding constraint while further
starving the one that is. Hypothesis direction was backwards; discarded
without a retrain.

## 7. Root cause of the T2/T3 gap, and why "bậc 2" closed most of it

**Implemented (§7.7), trained, and confirmed by the OOD benchmark (§6.2)** —
this section's diagnosis (7.1-7.5) describes the *pre-fix* mechanism; it
remains accurate as an explanation of why the old (still-current-in-§4-text)
formula was flawed. §7.6 has been updated with the real result.

### 7.1 Precise mechanism: pre-bậc-2 GRPO was trajectory-level REINFORCE, not per-step

`grpo.py:274-284`:

```python
ll_normed = ll.sum(dim=-1) / valid_len       # sum log-prob over ALL decode steps -> 1 scalar
surr = -(ll_normed.view(-1, 1) * adv).mean() # times ONE advantage for the whole trajectory
```

Every decision in a rollout — whether taken during the class-1, class-2, or
class-3 phase — receives the **exact same** advantage, derived from that
rollout's *final* T-vector (57%/29%/14% T1/T2/T3 weight). A decision made
during the class-3 phase (precedence already guarantees class-1/2 are fully
served — that decision structurally *cannot* influence `T1` or `T2` anymore)
still gets pushed 57% in the direction of "was T1 good in this rollout" —
information it had no causal access to. For a class-3-phase step the
noise/signal ratio is roughly 86% irrelevant / 14% relevant (the one thing
it could have affected, T3, at its full weight).

### 7.2 The fix: causal phase-masked advantage

`env.py:113` already computes `c_min` (lowest still-unserved class) at every
step — the exact phase boundary — but currently only uses it for masking,
then discards it. Fix: a step taken in phase `p` gets advantage restricted
to `Σ_{q≥p} w_q · r_q` (renormalized over `q≥p` only), using the per-level
ranks `centered_hier_rank_gpu` already computes internally before collapsing
them to one scalar (`grpo.py:99`).

- Phase-1 steps: `Σ_{q≥1} w_q r_q` = identical to today's formula — untouched.
- Phase-2 steps: only `T2,T3` terms remain — the T1 noise is gone.
- Phase-3 steps: only `T3` remains — 100% signal, 0% irrelevant noise.

This can't directly worsen T1 (phase-1 steps are mathematically unchanged),
**but** encoder/decoder weights are *shared* across all phases (same network
makes every decision), so cleaning up the phase-2/3 gradient can still have
second-order effects on shared representations that phase-1 decisions also
read from. "Structurally can't hurt T1" is an approximation, not a proof.

### 7.3 Phase-masking alone isn't enough — the decoder is still endgame-blind

A clean gradient signal only fixes *credit assignment*. Separately, §3 notes
`JointMVContext` gives the decoder no per-class remaining-arc-count or
current-phase signal at all — so even with a causally correct gradient, the
network may lack the *inputs* to express "position this vehicle well for the
upcoming class-2 frontier." Credit assignment tells the network *which*
decisions mattered; input features give it something to condition a better
decision *on*. Two independent analyses (training-objective and
architecture-focused) converged on the same decoder-feature fix without
cross-referencing each other — a reasonably strong signal both halves (7.2 +
7.3) are needed together, not just one.

### 7.4 Why bậc 0/1 structurally cannot reach "fully beat ILS", and bậc 2 might

Measured data (`logs/status_bm.md`) already shows the ceiling: shortlist
10→25 keeps closing the T2/T3 gap but with **diminishing returns**
(Pareto-dominance 13.1%→16.4%) while burning the speed advantage
(19.9x→13.5x). Two structural reasons this caps out:

- Bậc 0/1 only *re-rank and locally polish* samples drawn from a policy
  whose phase-2/3 behavior is noisy for the reasons in 7.1 — no amount of
  picking-the-best-of-100 or single-pass local search manufactures skill the
  policy never learned; the pool itself lacks good candidates.
- `intra_route_opt`/`inter_route_opt` in bậc 1 run to a **single** local
  optimum of a deadhead proxy, once. ILS's advantage on T2/T3 comes from
  **200 iterations** of perturb→search→accept, escaping local optima the
  single pass can't. Matching that would mean matching ILS's own iteration
  budget — which erases the entire point of using RL (speed).

Bậc 2 targets the actual source: if the policy itself learns good phase-2/3
behavior, raw rollouts (even greedy, K=1) should already sit closer to ILS
without any local search, and stacking bậc 1's funnel on *top* of a better
policy compounds rather than just re-selecting from the same weak pool.

### 7.5 This is the next link in the same fix chain, not a new direction

| Version | Problem | Effective T2/T3 signal at a phase-3 step |
|---|---|---|
| Hard lex rank (superseded) | T1 ties never happen on continuous floats → T2/T3 gradient is exactly 0 | 0% |
| Hierarchical soft rank (**current**, `gamma=0.5`, commit `0baf47b`) | Nonzero but applied uniformly across the whole trajectory | 14% signal, ~86% causally-irrelevant noise |
| Phase-masked soft rank (**bậc 2, proposed**) | Causally scoped to what each step could actually influence | 100% signal for T3-only-relevant steps |

Same underlying fix (give T2/T3 more/cleaner gradient), one level more
precise each time.

### 7.6 Calibration — confirmed by experiment (updated after §6.2's benchmark)

- **Confirmed, not just hypothesized.** Bậc 0 was predicted to "recover a
  meaningful slice" of the gap and in practice barely moved the needle
  (eps-tied sample pools turned out too narrow) — static reasoning about
  GRPO had already been wrong once this project, so 7.1-7.5 was treated as
  a hypothesis pending a retrain, not a guarantee, until §6.2's OOD
  benchmark came back: bậc 2 alone (no local search) already matches what
  bậc 1 needed local search to achieve on T3, and stacked with bậc 1 the T3
  gap closes to +0.1% (noise-level). The mechanism-level reasoning in
  7.1-7.5 held up this time.
- **"Fully beat ILS" on literally every one of 2858 instances was never the
  bar, and still isn't met** — bậc 2+1 reaches 20.6% Pareto-dominance, not
  100%. ILS spends 200 iterations per instance; some instances have local
  structure its exhaustive search finds that even the improved policy
  doesn't. The realistic target — *beats ILS on the large majority of
  instances while staying an order of magnitude faster* — is what was
  achieved (T1/T3 essentially tied-or-better on average, T2 gap cut from
  +26% to +6.6%, 22.8x faster), not 100% dominance, and that was the
  calibrated expectation going in.
- Cost to validate was a full retrain (~5h), much more than bậc 0/1's
  minutes-scale benchmark-only iteration — paid off this time.

K=16 may still be noisy for the already-low-gradient-share T2/T3 midranks;
K=32 remains untried, a cheap thing to test alongside §8's decoder upgrade
if pursuing further gains.

See conversation history / Fable analysis for the fuller design writeup
(inference-time hybridization vs training-objective fixes vs novel
architecture ideas) if resuming this work.

### 7.7 Implementation (§7.2 — §7.3's decoder features shipped separately in §8)

Code-reviewed line-by-line and independently test-verified (not just taking
the implementing agent's word for it) — 195 tests green, including new
tests specifically proving the phase-1 invariant (`torch.equal` against the
old combined formula) and a `variant='U'` regression guard.

- **`src/env/env.py`**: new `CARPEnv._phase(td)` → `(B,1)` long tensor =
  `c_min` (min still-unserved class, same quantity `get_action_mask` already
  computed internally), clamped to `[1,P]` so done/padding rows stay a valid
  index; constant `1` for `variant='U'` (no precedence, phase undefined).
  Stashed as `td["phase"]` in both `reset()` and `step()`, alongside
  `action_mask`.
- **`src/policy/policy.py`**: decode loop appends `td["phase"]` each
  iteration (the phase in effect *for that decision*, read before `env.step`
  advances state), stacked to `(B, dec_steps)`, returned as `outdict["phase"]`
  — same shape/convention as `valid_mask`.
- **`src/trainers/grpo.py`**:
  - `centered_hier_rank_by_phase(T, gamma)` → `(B, P, K)`: row `p-1` is the
    advantage for phase `p`, i.e. `Σ_{q≥p} w_q r_q` renormalized over `q≥p`
    only. `centered_hier_rank_gpu` is now just `[:, 0, :]` of this (phase-1
    row = old combined formula, verified bit-identical).
  - `_train_step` (variant `P` only): stashes `td["adv_phase"]` = `(B*K, P)`
    at the same point as the old scalar `advantage` (so DAPO group filtering
    slices it consistently); in the mini-batch replay loop, gathers each
    step's own phase-conditional column (`adv_phase.gather(1, phase_idx)`)
    and uses a genuinely **per-step weighted surrogate**
    (`-((ll * adv_step).sum(-1) / valid_len).mean()`) instead of the old
    "sum log-probs first, multiply by one scalar after" formula. `variant='U'`
    path is untouched (verified byte-identical loss under a phase-garbling
    regression test).

§7.3's decoder input features (class-progress, endgame awareness) —
phase-masking only fixes *credit assignment*; the decoder separately needed
an *input signal* to act on it — that's §8, now implemented and training.

## 8. "Bậc 3" — combined decoder upgrade — IMPLEMENTED, TRAINED, BENCHMARKED

Two independent gaps pointed at the same module (`JointMVContext`,
`src/policy/decoder.py`) and were cheapest to fix together in one retrain
rather than sequentially. Both are now implemented and code-reviewed
line-by-line (§8.1 below reflects the actual code, not a proposal); training
completed (stopped manually after plateau, §8.5) and the checkpoint is
benchmarked (§8.6).

- **§7.3's gap**: no per-class remaining-arc-count or current-phase signal.
- **A second gap found by reviewing a SIGKDD 2025 paper** ("Enhancing
  Generalization in Large-Scale HCVRP: A Rank-Augmented Neural Solver",
  Liu/Lian/Liu/Cao, `papers/2025_A Rank-Augmented Neural Solver.pdf`, code at
  `github.com/AAI-ZZU/RANS`): `JointMVContext` has **zero inter-vehicle
  signal** — each vehicle's query is built purely from its own local state;
  vehicle `m` has no way to know what vehicle `m'≠m` is doing. RANS's
  min-max-over-vehicles objective is structurally the same shape as ours
  (`T_k = max over class-k arcs`), and their ablation (their Fig. 8) shows
  this specific fix (they call it **Dual Cross-Attention**, DCA) helps at
  *every* tested scale — unlike their other mechanism (§8.3), it is not
  scale-gated, so it's relevant now, not just later.

### 8.1 Topology (implemented — `src/policy/decoder.py::JointMVContext`)

Encoder is **unchanged**. Only `JointMVContext` changes:

```
hidden (B,N+1,D) + td state (current_node, used_capacity, completion, adj,
                              phase [from §7.7], clss, visited, num_vehicle)
        │
        ▼
┌─────────────────── per-vehicle LOCAL feature (widened) ───────────────────┐
│ feat_m = [ node_embedding[current_node_m]          (D)                    │
│          , remaining_cap_m                         (1)  existing         │
│          , completion_norm_m                        (1)  existing        │
│          , dist_to_depot_m                          (1)  existing        │
│          , phase_feat_m = phase_m / P                (1)  NEW            │
│          , remain_frac_m (per-class remaining-arc     (P) NEW            │
│            fraction, from clss+visited, depot excl.,                     │
│            0 if class absent — div-by-zero clamp)                        │
│          ]                                    → D + (3+1+P) = D+7 for P=3│
└────────────────────────────────────────────────────────────────────────┘
        │  project_vehicle: Linear(D+7 → D)      [_class_progress() computes
        │                                          phase_feat/remain_frac]
        ▼
veh_q_individual (B, M, D)     — "h¹" in RANS's naming, still per-vehicle-only
        │
        ▼  self-attention among the M vehicle tokens, masked by valid_veh
   veh_attn_qkv (Linear D→3D) → masked MHA → veh_attn_out (Linear D→D)
        → veh_q_collab (B, M, D)   — "h²"
        │
        ▼  fuse individual + collaborative (RANS eq. 25 style)
   veh_q = self.fuse( concat(veh_q_individual, veh_q_collab) )   (B, M, D)
        │  Linear(2D → D)
        ▼  (unchanged from here — same as before bậc 3)
veh_q + graph_context (broadcast) → PointerAttention(...) → logits (B,M,N+1)
```

Actual cost: `JointMVContext.__init__(embed_dim, P=3, linear_bias=False)`
adds `veh_attn_qkv`, `veh_attn_out`, `fuse` — measured ~+200K params on top
of the 2.5M baseline. `AttentionModelPolicy.__init__` and `Decoder.__init__`
both gained a `P: int = 3` param threaded down from `src/train.py`
(`P=env.P`). No `env.py` changes beyond what §7.7 already added — `td["phase"]`
is reused, not re-derived. RANS's cross-attention-to-nodes path was **not**
needed — `PointerAttention` already cross-attends `veh_q` against
`glimpse_k/v` under the feasibility mask; the only missing ingredient was the
vehicle-to-vehicle self-attention pre-mixing step, now built.

### 8.2 Dynamic fleet size — handled, proven by test

`valid_veh = arange(M) < td["num_vehicle"]` (B,M) bool is computed once in
`JointMVContext.forward()` and used two ways: (1) masks `comp_norm`'s `amax`
reduction to real vehicles only (so ghost slots can't skew the normalization
denominator), (2) passed as the attention mask to the vehicle self-attention
(`attn_mask=valid_veh.unsqueeze(1)`, True=attend) so ghost "phantom" vehicle
slots — zero-initialized padding when a batch mixes e.g. `M_i=2` and
`M_i=10` instances — can't pollute real vehicles' collaborative
representations. This is proven, not just argued: `tests/test_grpo.py`'s
`TestBac3DecoderUpgrade::test_ghost_slot_masking` perturbs the ghost-slot
values in an otherwise-identical batch and asserts the real vehicles'
outputs are **bit-identical** (`torch.equal`) regardless — attention over
`M=2..10` is permutation-equivariant and count-agnostic given this one mask,
no per-M special-casing needed.

### 8.3 Deferred: Rank-Augmented Module + Disagreement Regularization

RANS's *other* mechanism (raising attention-matrix rank at large N via a
KNN-adjacency term `O=(φ(Q)φ(K)ᵀ+D⁻¹Â)V`, plus a pairwise-cosine
"Disagreement Regularization" loss term) targets `rank(attention) ≤
min(N,d)` becoming a binding constraint when `N >> d`. Our `d=128`; even the
`large` curriculum tops out around `N+1≈134` — this bound is not close to
binding yet, and RANS's own ablation shows it doesn't help (their model
trails the classical SISR heuristic) at the scales comparable to ours
(N=200-1000 in their benchmarks). **Not worth building now.** Cheap
tripwire to add in the same §8.1 change instead: two passive eval-time
metrics (effective rank of the last encoder layer's attention matrix; mean
pairwise cosine similarity of node embeddings — the DR quantity itself,
computed but not optimized against). If either metric trends badly once
training moves to `medium`/`large` curriculum, that's the signal to revisit
this section — until then it's dead weight.

### 8.4 Sequencing

1. ~~Let bậc 2 finish and get benchmarked~~ — **done, §6.2**.
2. ~~Build §8.1+§8.2 together as one change~~ — **done**: implemented,
   code-reviewed line-by-line (not just taking the implementing agent's
   report at face value), and covered by `tests/test_grpo.py`'s
   `TestBac3DecoderUpgrade` (4 tests, including `test_ghost_slot_masking`,
   §8.2) — full suite green before launching the retrain.
3. §8.3's monitoring metrics — **not added**; deferred along with §8.3's
   fix itself (still not worth building, reasoning in §8.3 unchanged).
4. **Retraining now** — see §8.5.

### 8.5 Training status — COMPLETE (stopped manually after plateau)

Launched on top of the bậc-2 checkpoint's identical hyperparameters (clean
A/B — only the decoder architecture differs): `MAX_EPOCH=200 BATCH_SIZE=512
MINI_BATCH_SIZE=256 GROUP_SIZE=16 CURRICULUM=small`, seed 6868.

- Log: `logs/train_20260703_162357.out`. Checkpoint dir:
  `outputs/checkpoints/grpo_osm_bac3/`. Lightning run: `outputs/lightning_logs/version_3`.
- Confirmed healthy at launch (epoch 0, no NaN/crash) and stayed healthy
  throughout — no NaN/crash at any point.
- **Measurably slower per training step** than bậc 2's run (~0.38-0.44 it/s
  vs bậc 2's ~0.56-0.60 it/s, roughly 30-35% slower) — this is the added
  per-decode-step cost of `_class_progress()` + the vehicle self-attention
  block in `JointMVContext`, run once per decode step for every one of the
  `B*K` rollout copies. Confirmed via code inspection that **no local
  search is involved in the training path** — bậc 1's funnel hybrid
  (`RLHybHCARP`) is an inference-only benchmark tool (`src/solvers/rl.py`),
  never invoked from `src/train.py`/`src/trainers/grpo.py`; the slowdown was
  entirely the new decoder compute, not an accidental local-search call.
- **`val/lex_best` plateaued around epoch 159** — no new best checkpoint for
  ~30 further epochs (checked repeatedly at epochs 140, 169, 187, 192) while
  `val/reward` sat flat around -4.22 to -4.28. The user stopped the run
  manually at ~epoch 192/200 once this was clear; the remaining ~8 epochs
  were not expected to move the needle. **Final checkpoint:
  `outputs/checkpoints/grpo_osm_bac3/epoch=159.ckpt`** (best by `val/lex_best`).
- Benchmarked against ILS on `data/osm_test_small`, same `HYBRID=1
  SHORTLIST=25` protocol as bậc 2's v6 row — see §8.6.

### 8.6 Bậc 3 — benchmark result: real but smaller increment than bậc 2

| Config | T1 | T2 | T3 | gap vs ILS (T2/T3) | Pareto-dom | speedup |
|---|---|---|---|---|---|---|
| ILS | 5.012 | 10.056 | 15.804 | — | — | 1.0x |
| Bậc 2 + Bậc 1 funnel, SL=25 (v6) | 4.850 | 10.473 | 15.496 | +4.2%/-1.9% | 25.8% | 16.9x |
| **Bậc 3**, same funnel SL=25 (v7) | 4.846 | 10.271 | 15.492 | **+2.1%/-2.0%** | **27.3%** | 16.9x |

T2 gap roughly halves again (+4.2%→+2.1%), Pareto-dominance nudges up
(25.8%→27.3%), T1/T3 essentially unchanged — consistent with the class-
progress + cross-vehicle features helping the T2-relevant load-balancing
behavior they targeted (§8's motivation), but with real diminishing returns
compared to bậc 2's much larger jump. Full detail: `logs/status_bm.md` (v7).

This alone did not reach a further target the user set mid-session — the
specific metric **"T1 not-worse-than-ILS" ≥ 90-95%** (chosen because
Pareto-dominance was separately established as structurally capped by
metric geometry, §6.2) — which stood at 82.4% on v7. See §9 for how that
gap was closed, without any further training.

## 9. "P1" — closing the T1-not-worse gap via a wider inference-time funnel

### 9.1 Diagnosis: the ~18% T1 shortfall is pool depth, not selection logic

Before building anything, an independent instrumented analysis (per-
candidate T1 values recorded before eps-lex selection, full 2858-instance
benchmark, checkpoint `epoch=159.ckpt`) tested the natural hypothesis that
`eps_lex_select`'s eps-tolerant T1 band (§7.2-era bậc-0 fix) was *knowingly
trading away* some T1 quality in exchange for better T2/T3 tiebreaks. That
hypothesis was **refuted**: of the 486 (of 2858) instances where the final
T1 exceeded ILS's, **0 were selection-logic failures** — the strict-lex
final pick already always recovers the shortlisted pool's best T1; on only
4/2858 instances did the eps-band even drop the pool's raw argmin-T1
sample, and on all 4 the final pick still matched the pool minimum.
**100% of the shortfall (486/486) was "no candidate in the K=100 sample
pool was good enough"** — a pool-depth problem.

Two further measurements contextualized this rather than reading it as a
pure policy-quality gap:
- An exact T1 oracle (variant P's precedence forces every vehicle's
  class-1 arcs to be served as an opening prefix, so T1 is an exact
  min-max-partition-of-Held-Karp-paths problem — tractable at this
  benchmark's ~7-10 class-1-arc instance sizes, verified oracle ≤ both
  ILS's and RL's T1 on 780/780 sampled instances) shows the reference ILS
  run is T1-optimal on only ~18-20% of instances.
- A second ILS run (different seed, same config) is "not worse than" the
  reference ILS run's own T1 only **66.6%** of the time — i.e. ILS vs a
  re-seeded copy of itself already falls well short of 100%, from pure
  construction/perturbation randomness alone. The 82-95% range achieved by
  RL should be read against that ceiling, not against a naively-assumed-
  perfect ILS.

### 9.2 Fix: widen the candidate pool, don't touch selection or training

`RLHybHCARP` (`src/solvers/rl.py`) gained a `topk_t1` parameter (default 0,
no behavior change): an extra top-`topk_t1`-by-raw-T1 candidate lane,
unioned with the existing eps-lex shortlist before local search, so the
strict-lex final pick has more (and more T1-diverse) candidates to choose
from. Combined with widening `num_sample` (K) at the call site — no
architecture or training changes, purely an inference-time budget increase
along a specifically-targeted axis (T1-only rank) rather than raw random
sampling. `scripts/bm.sh` gained a matching `TOPK_T1` env var.

Ruled out (measured, not just reasoned): tightening `eps` toward pure-
argmin-T1 selection scores *worse* (57.5% vs 82.4% baseline — it discards
the eps-band's T2/T3 tiebreak value for no T1 gain, confirming §9.1's
finding that the band was never the bottleneck); a T1-focused fine-tune of
the plateaued checkpoint was not attempted — the diagnosis in §9.1 shows
this metric is a best-of-K tail statistic, not a mean-shift problem that
gradient-based training targets (RL's mean T1 is already better than ILS's,
§8.6), so training-side intervention was correctly identified as the wrong
lever.

### 9.3 Result — measured on the full benchmark, target met

`NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01 TOPK_T1=100`, same `epoch=159.ckpt`
checkpoint, run through `scripts/bm.sh` (not a scratch script):

| Config | T1 | T2 | T3 | gap vs ILS (T2/T3) | lex-win | Pareto-dom | speedup |
|---|---|---|---|---|---|---|---|
| ILS | 5.012 | 10.056 | 15.804 | — | — | — | 1.0x |
| Bậc 3, SL=25 (v7) | 4.846 | 10.271 | 15.492 | +2.1%/-2.0% | 72.9% | 27.3% | 16.9x |
| **Bậc 3 + P1** (v8) | **4.758** | **10.047** | **15.515** | **-0.1%/-1.8%** | **90.2%** | **36.4%** | 5.4x |

**T1-not-worse-than-ILS: 82.4% (v7) → 94.89% (v8)** — meets the 90-95%
target. Every other tracked metric improved simultaneously (T2 gap flips to
essentially tied, lex-win +17pp, Pareto-dominance +9pp, the best of any
config in this document), at the cost of speed dropping from 16.9x to
5.4x ILS — still solidly faster, and no retrain, no architecture change was
needed. Full detail and repro commands: `logs/status_bm.md`.

This is the best result across the full bậc 0→3+P1 progression on every
axis. See §10 for how the remaining ~5% T1 gap was closed to a literal 100%
guarantee.

## 10. "P2" — an exact T1 oracle closes the remaining gap to literal 100%

### 10.1 Why P1 still isn't 100%, and what makes 100% possible at all

P1's 94.89% is fundamentally a sampling result: K=400 rollouts almost always
find ILS's T1 or better, but "almost always" is not "always" — there is no
guarantee against a residual few percent of instances where the pool simply
doesn't contain a good enough sample. Closing that residual with more
sampling alone hits diminishing returns (§9.1's K-scaling data: 20%→30%→
40%→52% not-worse at K=12/25/50/100 — each doubling buys a shrinking
increment). A *guarantee* requires an exact method, not more samples.

One exists here, specific to variant P: under linear precedence, every
vehicle must serve all of its class-1 arcs before any higher class, so a
vehicle's class-1 workload is an **opening route prefix**, and T1 depends
*only* on how the class-1 arcs are partitioned across vehicles and ordered
within each one. With ~7-10 class-1 arcs on this benchmark (never observed
above 10; the implementation guards up to 16), this is a Held-Karp
shortest-path DP over the class-1 arcs combined with a submask partition DP
across ≤M vehicles — solved *exactly*, not sampled, in single-digit
milliseconds per instance (`src/solvers/oracle.py::t1_oracle_argmin`).

### 10.2 Design: an extra candidate arm, not a replacement

The oracle's optimal class-1 partition is forced into the policy's decode
as an opening prefix (`_decode_forced`, bypassing the normal sampling
entrypoint to inject specific actions), then classes 2/3 are completed by
the policy — `oracle_r=64` sampled completions + 1 greedy, shortlisted by a
T2-eps-band (all completions from one forced prefix share T1 exactly, so
plain strict-lex degenerates to argmin-T2 — the same continuous-float
degeneracy `eps_lex_select` fixed for T1 in bậc 0), then local-searched
(`_local_search_one`, the same P1-era "keep post-LS only if lex-better"
convention). This "head" arm's own best candidate is then **unioned** with
the existing P1 pool arm's answer (`oracle_union_pick`): if the head's T1 is
strictly better, take it; if the pool's T1 is strictly better, keep it; if
tied, an eps-band on T2 then min T3 decides — so the oracle arm can only
help or tie, never override an already-at-least-as-good P1 answer. The
union winner is re-evaluated through the float64 `SolutionEvaluator` (the
same precision path local search already uses) before being returned, so
the "100% not-worse" guarantee holds at full precision rather than the
decode-time float32 reward's.

An optional exact secondary DP (`oracle_tiebreak=True`, default on) breaks
ties among multiple T1-optimal class-1 partitions (common — true on ~34% of
instances) by minimizing a "how soon can each vehicle start class-2" proxy
(prefix finish time + distance from final position to the nearest class-2
arc), *within* the T1-optimal tie set — so it cannot regress T1, only choose
better among equally-T1-optimal options.

### 10.3 Result: the feared T2/T3 regression did not happen — it inverted

Before building this, the obvious risk was raised and taken seriously:
forcing a policy into a prefix it never chose itself (out-of-distribution
relative to training, where it always starts from its own decisions) could
plausibly produce worse completions. Measured on the full 2858-instance
benchmark (`epoch=159.ckpt`, `NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01
TOPK_T1=100 ORACLE_HEAD=1`, `scripts/bm.sh`):

| Config | T1 | T2 | T3 | gap vs ILS (T1/T2/T3) | lex-win | Pareto-dom | speedup |
|---|---|---|---|---|---|---|---|
| ILS | 5.012 | 10.056 | 15.804 | — | — | — | 1.0x |
| Bậc 3 + P1 (v8) | 4.758 | 10.047 | 15.515 | -5.1%/-0.1%/-1.8% | 90.2% | 36.4% | 5.4x |
| **Bậc 3 + P1 + P2** | **4.692** | **8.894** | **14.965** | **-6.4%/-11.6%/-5.3%** | **98.0%** | **57.5%** | 4.1x |

**T1-not-worse-than-ILS: 94.89% → 100.00%** — verified at both a `1e-6`
relative tolerance and literal `tol=0` exact comparison (the float64
re-evaluation step delivers the guarantee at full precision, no residual
float32 noise). Every other tracked metric improved substantially rather
than regressing: T2-not-worse 52.9%→85.09%, T3-not-worse 55.9%→63.19%,
lex-win 90.2%→98.0%, Pareto-dominance 36.4%→57.5% — the best result across
the entire bậc 0→3+P1+P2 progression on every axis simultaneously, at the
cost of speed dropping further from 5.4x to 4.1x ILS (still comfortably
faster).

The mechanism behind the inverted risk: the oracle's class-1 partition is
min-max-balanced across vehicles *by construction* (that's literally what
it optimizes), which tends to leave vehicles in a better-positioned state
for class-2 than the policy's own typically-unbalanced prefixes would —
so forcing it doesn't just avoid hurting T2/T3, it actively helps on
average.

### 10.4 Scope and caveats

- **Variant P only.** The oracle's forced-opening-prefix premise depends on
  P's linear precedence; `RLHybHCARP._oracle_head_arm` explicitly declines
  (returns `None`, falls back to the pool arm unchanged) when
  `self.variant != 'P'`. `oracle_head` has no effect for variant U and
  should stay off there.
- **Applicability guard, not an assumption.** `t1_oracle_argmin` requires
  `n1 <= max_n1` (default 16; the DP is `O(2^n1 · n1²)`) and a
  capacity-feasible partition to exist; either failing returns `T1 = inf`
  and the caller falls back to the pool arm with no crash. This benchmark's
  actual max `n1` is 10 — comfortably inside the guard — but the guard is
  enforced, not assumed, for future data with larger class-1 counts.
  Scaling to noticeably larger `n1` would need a different, likely
  approximate, method for the class-1 subproblem — the guard is exactly
  the tripwire for when that becomes necessary.
- **Backward compatible.** `oracle_head` defaults to `False`; every
  existing config, script, and prior benchmark row is unaffected. Proven by
  `tests/test_rl.py::TestOracleHeadRegression.test_disabled_is_byte_identical_to_p1`
  (byte-identical output to the pre-P2 pipeline under a fixed seed) and
  `test_oracle_code_not_reached_when_disabled` (the oracle path isn't even
  invoked unless opted in).

## 11. Investigated and rejected: moving the oracle into training

P2 (§10) only helps at *inference* time and costs real latency (5.4x→4.1x
ILS). The natural next question — can the exact T1 oracle be folded into
*training* instead, so the policy learns to act like it directly (faster
inference, no separate oracle-forced decode arm needed), and does that
lesson transfer to `medium`/`large` curriculum where the oracle itself is
computationally inapplicable (§10.4, n1 ≈ 21-35 there vs the guard's 16)?
Investigated via a smoke-scale experiment (fine-tuning `epoch=159.ckpt` for
a few hundred steps, not a full retrain) before committing to anything —
**result: negative, don't pursue this**, kept here so it isn't re-investigated.

**Mechanism tried**: mix 1-2 of each K=16 training rollouts per group with an
oracle-forced class-1 prefix (same `_decode_forced` mechanism §10 uses at
inference), let the policy complete classes 2/3, and feed the result through
the *existing* GRPO machinery unchanged (phase-masked group-rank advantage,
DAPO filtering, replay loss) — no new loss term. This is mathematically
clean in this trainer specifically because the loss is **pure REINFORCE at
fixed θ, not a clipped PPO ratio** (§4's Loss subsection, corrected during
this investigation) — there's no importance-sampling machinery an off-policy
forced rollout could violate; a forced rollout's gradient is just
rank-advantage-weighted imitation, which should auto-anneal to zero once the
policy already matches the oracle (midrank tie-averaging zeroes a tied
slot's relative advantage).

**Measured result: it makes the policy worse, not better, at both scales.**
At the training curriculum itself ("small"), a symmetry problem bites: the
oracle's `parts` fix one arbitrary vehicle-assignment + linearization out of
a large T1-equivalent class, so naive teacher-forcing punishes the policy
for preferring any other equally-good linearization (probed: base-policy
mean logp on forced steps ≈ -5.3 to -5.8 vs -1.7 to -1.9 on its own
choices). A symmetry-aware "constrained-decode" variant (restrict the
action set to {each vehicle's next oracle-part arc}, let the policy pick
its own interleaving/timing) reduces the damage but still doesn't help:
across every dose/lr tried (teacher 2/16 and 1/16 doses, constrained-decode
2/16), the greedy-decode match-rate against the exact oracle *degraded*
relative to a plain-GRPO control run on the same fixed batches (control:
58-60%, flat; oracle-guided arms: 45-51%, worse), and the best-of-16 gap
only marginally beat the control's own run-to-run noise band.

**Transfer to `medium` — the hoped-for payoff — was clearly negative, not
just absent.** Paired greedy-T1 head-to-head against a matched control on
247 `medium`-curriculum val instances (n1 mean 21.3, oracle verifiably
inapplicable on 100% of them): every oracle-guided arm lost to its control
more often than it won (worst: 74.1% loss rate; best variant tried, the
constrained-decode one: still 61.5% loss rate). Whatever the guidance
taught was small-scale-specific pattern-matching, not a transferable
"balance class-1 completion across vehicles" skill — the opposite of the
hoped-for generalization.

**Caveat, stated plainly**: this fine-tunes an already-converged, plateaued
checkpoint for a few hundred steps — it does not rule out that mixing this
signal in *from scratch* (integrated over a full ~5h training run rather
than bolted onto a plateau) could behave differently. That hypothesis is
untested and not cheaply testable (it requires the full retrain this
investigation was specifically designed to avoid committing to without
evidence).

**Recommendation adopted**: keep the inference-time-only P1+P2 design
(§9-§10) as-is; it already delivers 100% T1-not-worse, 98% lex-win, 57.5%
Pareto-dominance at 4.1x ILS speed, and the smoke evidence argues against
spending a training cycle chasing this specific idea. If inference speed
(4.1x) becomes the binding constraint, the right lever is tuning the
inference budget directly (`NUM_SAMPLE`/`oracle_r`/`oracle_shortlist`) —
orthogonal to training, cheap to re-measure, no retrain required. One cheap
thing worth keeping from this investigation: logging "greedy/best-of-K T1
vs exact-oracle T1" as a passive validation-time diagnostic in any future
training run (near-zero cost, gives a calibrated T1-optimality readout and
would be the pre-registered success metric if this idea is ever revisited
with a from-scratch retrain).

## 12. Investigated and rejected: LS-shaped training advantage (reward shaping)

Sibling investigation to §11, same throughline ("can a classical heuristic
be folded into training instead of staying inference-only") but a
deliberately different mechanism meant to sidestep §11's specific failure
mode. Also negative — kept here for the same reason as §11: so it isn't
re-investigated without new information.

**Mechanism tried**: instead of forcing any action (§11's problem), let the
K=16 rollouts per training group sample entirely on-policy as today, then
run the existing local search (`utils/local_search.py`, deterministic,
inference's `_local_search_one` "keep post-LS only if lex-better" rule) on
**every** rollout's decoded route, and compute the group-relative phase-
masked advantage from the **post-LS** T-vectors instead of the raw decode
ones — while replaying/reinforcing the log-likelihood of the original,
untouched sampled actions (not the LS-modified route). The hope: sharper,
less noisy credit assignment (rank by "how good after a cheap deterministic
polish", not raw sampling luck), with no action forced and therefore none
of §11's symmetry-punishment problem.

**Soundness is genuinely different from §11's**: REINFORCE's unbiasedness
only requires the reward be a deterministic function of the sampled
trajectory (no gradient through it, nothing off-distribution) — local
search satisfies that; this is not an invalid estimator toward the shaped
objective it targets. It has real literature precedent (DeepACO, NeurIPS
2023, trains toward LS-refined solutions via REINFORCE) as a workable
*general* pattern in neural combinatorial optimization.

**But it has its own distinct contamination problem, and the data
confirms it hurts anyway.** How much LS improves a given rollout is driven
by incidental "swap headroom" in that specific sampled route (same-class
swaps only, per `intra_route_opt`'s per-class-`p` structure) — a mediocre
decode with easy nearby fixes gets an artificially boosted rank position, a
decode already near a local optimum doesn't, and because LS runs
sequentially class-by-class on the same route, a phase-1 rearrangement's
knock-on effect also contaminates whatever T2/T3 ranks the phase-2/3
credit. Measured (300-step fine-tune of `epoch=159.ckpt`, same protocol as
§11 — fixed/matched batches and seeds vs a plain-GRPO control):

- **Wall-clock**: 2.67x slower per training step (1.85s vs 0.69s) — local
  search cost dominates, rollout itself is unaffected (~0.04s either way).
- **Convergence on the training curriculum ("small")**: no benefit; several
  deployment-relevant post-LS metrics moved slightly the *wrong* direction
  relative to control over the last 5 evals (`lex16_T2` +0.06, `lex16_T3`
  +0.24, `sls16_T3` +0.12, `bs16_match` -0.01) — consistent in direction
  with the headroom-noise mechanism above, though this is a short fine-tune
  of an already-converged checkpoint so the signal is modest.
- **Transfer to `medium`** (256 held-out instances, same paired protocol as
  §11): decisively negative. Best-of-16 T1 win/tie/loss vs control =
  22.3%/6.1%/71.7%; lex-16 win/tie/loss = 25.9%/0.0%/74.1%. Even on the
  exact metric this mechanism was designed to optimize (deployment-style
  post-LS best-of-17), it **loses to plain-GRPO control more often than it
  wins** (45.7% win / 54.3% loss) — the LS-shaped arm's raw sampling
  diversity/quality measurably degraded on `medium` (best-of-16 T1 mean:
  base 8.13, control 8.23, LS-shaped 8.75), consistent with training having
  rewarded "LS-friendly" raw actions over genuinely good ones.

**Recommendation**: don't pursue further without a redesign that first
fixes the headroom-conflation problem (e.g., shape only the advantage's
*magnitude* while preserving raw-T rank order, or normalize each rollout's
LS gain against its own baseline LS-improvability rather than feeding the
absolute post-LS value into the rank) — the contamination mechanism is
structural, not a "needs more steps" issue, and the medium-transfer test
(this idea's best chance to differentiate itself from §11) confirmed the
concern rather than refuting it. Same bottom line as §11: keep the
inference-time-only P1+P2 design (§9-§10); if inference speed is the
actual constraint, tune the inference budget directly rather than
retraining.

## 13. Investigated and rejected: advantage-filtered ILS-imitation (diverse-target BC)

Third and final round in this line of investigation (sibling to §11-§12),
motivated by wanting a genuine training-time "hybrid" story — this design
was specifically engineered to fix every root cause diagnosed in §11 and
§12, not repeat them. It also failed, and the failure generalizes the
lesson: **any heuristic-derived training signal tried so far degrades
medium-curriculum transfer, no matter how carefully the known failure
modes are patched.**

**Design** (addressing §11/§12's diagnosed causes point-by-point):
- **Teacher = `ILSHCARP`**, not the exact T1 oracle — ILS runs at any
  instance size, so `medium`-curriculum imitation targets could be
  generated directly (no more "hope small-scale training generalizes up";
  §11's untestable-transfer gap is closed by construction here).
- **Diverse targets**: 3 different ILS seeds per instance (measured: full
  diversity, mean 3.0/3.0 distinct T-vectors — never collapsed to one).
- **Policy-preferred linearization + min-CE selection**: rather than one
  arbitrary canonical action sequence (§11's core problem), each ILS
  solution's per-vehicle route content is fixed but *which vehicle acts
  next* is chosen by the policy's own greedy preference
  (`constrained_greedy`), and of the 3 (linearized) targets, only the one
  the policy already assigns the highest log-probability to is used for
  the imitation loss — directly targets the symmetry-punishment cost
  measured in §11 (reduced from -5.3 to -5.8 mean logp there to -2.2 to
  -3.9 here, a real, measured improvement in this dimension, though not
  eliminated).
- **Advantage filter (self-imitation gate)**: a target is only imitated if
  it lexicographically beats the policy's own fresh best-of-8 sample on
  that instance — meant to stop the policy being taught to be *worse*
  where it already wins (recall RL already beats ILS on average, §9-10).
  Verified behaving as intended: at `small`, ILS beats the policy only
  ~50% of the time (little to teach, consistent with RL already winning
  there); at `medium`, ILS beats the policy's best-of-16 essentially 100%
  of the time (real teaching material genuinely exists there).
- **Full decoupling**: BC is a separate optimizer step every 4 GRPO steps
  on a precomputed ILS-solution buffer, never mixed into the GRPO
  rollout/rank/advantage machinery — avoids both §11's forced-rollout
  contamination and §12's shaped-rank contamination by construction.

**Result — still negative, and decisively so at medium** (300-step
fine-tune of `epoch=159.ckpt`, matched seed/batches vs the §12 plain-GRPO
control, evaluated on 247 real held-out `medium` instances with real ILS
references, independently re-verified against the raw held-out-eval data):

| Arm | vs base, medium, lex16 w/t/l | bs16 T1 mean (medium) | T1-not-worse vs ILS (medium) |
|---|---|---|---|
| control (plain GRPO) | 44.1%/0%/55.9% | 8.05 | 20.6% |
| **bc (ILS-imitation)** | **13.4%/0%/86.6%** | **8.86** | **10.9%** (down from base's 23.1%) |
| rlmed (plain GRPO, medium-batch exposure, no imitation) | **53.8%/0%/46.2%** | **7.92** | 19.8% |

The BC arm is worse than the base checkpoint on every medium-curriculum
metric measured, including the T1-not-worse figure it was hoped to
improve. **The `rlmed` control arm — plain GRPO fine-tuning exposed to
medium-curriculum batches, with no heuristic imitation at all — was the
best performer of the three fine-tuned arms at medium**, beating both the
base checkpoint (53.8% win) and the ILS-imitation arm (90.3% win). This
isolates the effect cleanly: whatever value exists in touching
medium-scale data during training comes from ordinary RL exposure to that
distribution, not from imitating a heuristic. Production training
currently uses `curriculum=small` only (§5) — broadening the training
curriculum itself (a completely different, non-hybrid lever) is a
genuine, real signal worth investigating separately, unlike any of the
three heuristic-hybridization attempts.

Why the filter didn't rescue it despite behaving correctly (93.8-100% of
medium instances had a genuinely better ILS teacher, verified): passing
the filter answers "is the teacher better", not "is imitating a whole
foreign route, disconnected from the lexicographic T1/T2/T3 tradeoff GRPO
teaches holistically, actually a good gradient step" — the same
myopic/disconnected-subproblem critique from §11, now at whole-route
granularity instead of just the class-1 prefix. The policy never caught up
to ILS at medium within the 300-step budget, so the filter's intended
self-annealing property (imitation pressure fades once the policy
surpasses the teacher) never had a chance to engage — pass rate stayed
69-97% throughout.

**Cross-round synthesis (§11 + §12 + §13 — read together)**: three
independent designs, each specifically engineered to fix the previous
round's diagnosed root cause (exact-oracle → symmetry problem → diverse
ILS targets + policy-preferred linearization; single-canonical-target →
min-CE-over-targets; contaminated-rank/forced-actions → full decoupling;
"is imitation even warranted" → advantage-filtering), have now all failed
to transfer to `medium` curriculum. The common thread is not any one fixed
bug — it's that a heuristic-derived training signal, however faithfully
isolated from the RL machinery, keeps teaching something that doesn't
generalize past the scale it was computed at.

**Recommendation — final for this line of investigation**: report the
already-shipped, fully-measured inference-time hybrid (P1 wider funnel +
P2 exact-T1-oracle arm, §9-§10 — 100% T1-not-worse, 98% lex-win, 57.5%
Pareto-dominance, 4.1x ILS speed on the full 2858-instance OOD benchmark,
not a smoke test) as the paper's "hybrid RL+heuristic" contribution. It is
not a fallback chosen for lack of a better training-time story — after
three good-faith, well-motivated, progressively-refined attempts, it is
the strongest and most rigorously validated hybrid result available. Do
not attempt a fourth heuristic-in-training variant without genuinely new
information (e.g. evidence from a from-scratch retrain, which none of
these three smoke tests constitute — all were short fine-tunes of an
already-converged checkpoint, a caveat that applies equally to all three
rounds and remains untested).

## 14. Codebase audit — dead code and unearned complexity removed

Requested directly: re-evaluate the current policy/GRPO training code for
anything that can be cut or simplified without reducing accuracy or
generalization, aiming for a minimal, clean implementation. Full audit
(read-only) done first — new smoke-scale ablations run on the trained
checkpoint to test whether the trained network actually relies on each
"never formally isolated" architecture piece before touching anything —
then a scoped set of changes applied based on the findings, all
independently re-verified (reran the ablation script, cross-checked
`outputs/lightning_logs/version_{1,2,3}/metrics.csv` directly, grepped for
every claimed dead-code site) before implementing.

### 14.1 What was NOT touched, and why — everything load-bearing stayed

Weight-level forward-ablation probes (zero a pathway's contribution on the
trained `epoch=159.ckpt`, no retraining — exact, since every relevant
linear is bias-free) on 256 held-out `small` + 247 held-out `medium`
instances showed every previously-"never isolated" architecture piece is
demonstrably relied on by the trained network, **most of all on `medium`
— the exact generalization axis being protected**:

| Ablated | medium paired lex16 loss vs intact | medium T1-not-worse-vs-ILS |
|---|---|---|
| (intact base) | — | 0.231 |
| cross-vehicle self-attention (§8) | **99.6%** | 0.049 |
| both class-progress features (§8) | 87.0% | 0.130 |
| encoder `dist_bias` (§3, pre-dates this project) | 98.8% | 0.053 |

None of these were cut. (Isolating *which* piece of bậc-3's combined
change drove §8.6's benchmark delta, or `dist_bias` with/without from
scratch, would need real retrains — not smoke-testable, not attempted
here; this doesn't block any decision above since both are clearly
load-bearing either way.) `oracle_tiebreak` (§10) was also kept as-is —
its own P2 investigation already measured a small, real, positive
contribution (+1.4pp T2-not-worse, +1.5pp Pareto) at negligible cost; it
was reclassified as "hardcode on" (§14.2) rather than removed.

### 14.2 What was cut (all verified before removal, all behavior-neutral or explicitly signed off)

- **`RLWarmILSHCARP`** (`src/solvers/rl.py`) and everything only it used
  (`vehicles_to_action`, the `_last_best_vehicles`/`_last_best_action`
  bookkeeping in `RLHybHCARP.__call__`, the `--warm_ils`/`--ils_iters` CLI
  flags) — the RL-warm-started-ILS direction the user explicitly rejected
  earlier (§6.3.1: "gain too small for the speed given up," 16.9x→9.2x for
  +1.2pp Pareto). Verified zero test coverage and zero `logs/status_bm.md`
  rows depend on it before deleting.
- **DAPO dynamic group filtering** (`src/trainers/grpo.py::_train_step`) —
  the `active_mask`/`skipped_frac`/early-return/group-dropping machinery.
  Directly measured from all three production runs'
  `train/skipped_frac`: **version_1 (pre-bậc-2) = 0.0 on every logged
  step; version_3 (current production checkpoint's run) = 0.0 on every
  logged step; version_2 (bậc-2) = mean 0.0019%, max 0.195% (~1 group in
  512, once)**. An all-tied K=16 group needs an exact float T-vector tie
  across all 16 samples — the same continuous-tie argument that motivated
  replacing the original hard lex rank (§4) — so this essentially never
  fires, and removing it is behavior-neutral for every training run ever
  done on this codebase. `tests/test_grpo.py`'s
  `TestDynamicGroupFiltering` (which asserted the old early-return/
  `skipped_frac` behavior) was replaced with `TestTiedGroupsDoNotCrash`,
  covering the invariant that actually matters (an all-zero-advantage
  batch stays numerically safe) without asserting removed machinery.
- **Verified-dead code**: unused `numpy` import and a byte-identical
  duplicate `self.log(".../lex_best", ...)` call in `grpo.py`; unused
  `gen_tours` import and a never-called `RLHCARP.import_instance` method
  in `rl.py`; four constructor parameters accepted then silently ignored
  or immediately overwritten — `Decoder.__init__`'s `context_embedding`/
  `dynamic_embedding`/`sdpa_fn`/`pointer` (`src/policy/decoder.py`) and
  `PointerAttention.__init__`'s `sdpa_fn`+`**kwargs`; the `StaticEmbedding`
  no-op class (`forward` always returned `(0,0,0)`) and `Decoder.
  _compute_kvl` (added zero to the cached static keys/values, i.e. an
  identity function) — inlined and deleted; `MultiHeadAttention`'s
  `causal` (never read) and `sdpa_fn` (stored, never called — `forward`'s
  manual matmul path is required for the pre-softmax `dist_bias` add, so
  it never used `self.sdpa_fn`) threaded needlessly through 4 constructor
  signatures (`Encoder`→`GraphAttentionNetwork`→`MultiHeadAttentionLayer`→
  `MultiHeadAttention`) down to `encoder.py`.
- **CLI/config surface collapsed**: `oracle_r`, `oracle_shortlist`,
  `oracle_shortlist_eps`, `oracle_union_eps2`, `oracle_tiebreak` were never
  varied from their defaults in any historical benchmark row (every row in
  `logs/status_bm.md` is reproducible from just `{num_sample, eps,
  shortlist, topk_t1, oracle_head}`) — demoted from CLI flags to
  constructor-only keyword arguments (still directly testable, e.g.
  `tests/test_rl.py`'s oracle tests construct `RLHybHCARP`/
  `t1_oracle_argmin` directly). `--ckpt`/`--cpkt` (two flags for the same
  path, merged via `ckpt_path = args.ckpt or args.cpkt` regardless of
  mode) unified to `--ckpt` everywhere, including `scripts/bm.sh`.

Net: ~215 lines removed across `src/trainers/grpo.py`,
`src/solvers/rl.py`, `src/policy/decoder.py`, `src/policy/encoder.py`,
`scripts/bm.sh`, with `tests/test_grpo.py` updated to match — zero
intended behavior change for the shipped P1+P2 configuration (full test
suite green: 213 tests, 0 failures, 7 pre-existing skips; the production
`HYBRID=1 NUM_SAMPLE=400 SHORTLIST=25 EPS=0.01 TOPK_T1=100 ORACLE_HEAD=1`
config re-run end-to-end after the change to confirm no regression).

## 15. Extending P2 past `max_n1`: a local-search heuristic for `medium`/`large`

P2 (§10) only helps where the exact T1 oracle is computationally applicable
(`n1 <= max_n1`, default 16 — comfortably covers `small`, roughly half of
`medium`, essentially none of `large`, per §10.4/§13's measured n1
distributions: small mean 8.8/max 10, medium mean 15.9/median 15/max 21,
large mean 30/max 35). Above that, the arm was previously just skipped
(safe, but zero benefit). Investigated a fast heuristic to extend the
arm's reach — an ILP/CP-SAT route was explicitly considered and rejected
(user's explicit call, not pursued) — in favor of a construction +
local-search heuristic, scoped to exactly the same subproblem.

### 15.1 Design (`src/solvers/oracle.py::t1_heuristic_argmin`)

Same interface/contract as `t1_oracle_argmin` (drop-in: same `T1`/`parts`
return shape, `inf`/`None` on infeasibility) but `O(restarts * n1^2)`-ish
per local-search sweep bounded by a wall-clock `time_budget`, not
`O(2^n1 * n1^2)` — runtime stays under ~1s across the whole tested n1
range (10-35) regardless of problem size, trading the exact guarantee for
an empirically-measured optimality gap instead.

- **Construction**: farthest-point-dispersion seeding (one arc per
  vehicle, maximizing min distance to already-chosen seeds) + greedy
  best-insertion of the rest, ranked by the resulting *sorted-descending
  route-cost tuple* (min-max first, balance second) so insertion choices
  already anticipate the min-max objective, not just total cost.
- **Local search**: first-improvement relocate/swap between routes, plus
  per-route 2-opt + Or-opt(1,2) — but only below `intra_cutoff` (default
  25) class-1 arcs. Above that, the O(n²)-O(n³) intra-route sweeps were
  measured to crowd out ILS restarts within a fixed time budget; skipping
  them and spending the budget on more relocate/swap/restarts instead
  roughly halved the mean gap at n1=30 (3.88% → 1.98%) — a real,
  measured tradeoff, not a guess.
- **Anytime**: multiple randomized-seed restarts, then ILS (random
  capacity-safe relocation kicks + re-descent) spend the remaining
  budget. A final exact Held-Karp re-ordering polishes any route with
  ≤12 arcs (provably order-optimal for its final assignment, independent
  of budget).
- **Capacity-infeasibility check**: if any single class-1 arc's own
  demand exceeds capacity, no partition can ever place it — checked
  upfront (the construction step otherwise seeds routes unconditionally
  and wouldn't catch this on its own; found and fixed during test-writing,
  `tests/test_rl.py::TestT1HeuristicArgmin::test_capacity_infeasible_returns_inf`).

### 15.2 Measured quality/speed tradeoff

Cross-validated against the exact DP wherever tractable (n1 ≤ 19, 284
real instances from `data/osm_train`'s small/medium buckets):

| Budget | Mean gap | Median | p90 | Exact matches |
|---|---|---|---|---|
| 0.03s | 0.39% | 0.00% | 0.83% | 85.6% |
| 0.15s | 0.10% | 0.00% | 0.00% | 94.4% |
| 0.5s | 0.04% | 0.00% | 0.00% | 97.9% |

At larger n1 (vs a long best-known reference, since the exact DP isn't
tractable there to compare against directly), quality degrades
monotonically but runtime stays cheap (0.5s budget):

| n1 | Mean gap | p90 | Median wall-clock |
|---|---|---|---|
| 19 | 0.18% | 0.21% | 511ms |
| 25 | ~1.5% | ~3% | ~550ms |
| 31 | 2.66% | 6.12% | 587ms |
| 35 | 5.64% | 9.99% | 623ms |

The heuristic never beat the exact optimum in any cross-validated
comparison (a sanity check confirming the DP is a valid lower bound and
the comparison methodology is sound).

### 15.3 Union-based safety net still holds

`oracle_union_pick` (§10.2) only inspects T-vectors, not how they were
produced — the "extra arm can only help or tie" guarantee is structural,
independent of whether the head arm is exact or heuristic. Directly
verified: 12,780 union decisions built from the 284 real exact-vs-heuristic
T1 pairs crossed with a range of pool-arm scenarios and randomized T2/T3
draws → **0 T1-safety violations** (the union result's T1 was always ≤
the better of the two arms, within tie-tolerance, in every case).

### 15.4 Production wiring and the resulting guarantee at scale

`RLHybHCARP` (`src/solvers/rl.py`) gains `oracle_use_heuristic` (default
`True`), `oracle_heuristic_time_budget` (default 0.5s), and
`oracle_heuristic_intra_cutoff` (default 25) constructor parameters — not
exposed on the CLI (§14's "don't add a flag for a knob nothing varies"
philosophy; construct `RLHybHCARP` directly if you need to change them).
`_oracle_head_arm` now computes `n1` once and dispatches: `n1 <=
oracle_max_n1` → exact DP (unchanged code path, so the shipped
small-curriculum benchmark is byte-identical to before this change); `n1
> oracle_max_n1` and `oracle_use_heuristic` → the heuristic; otherwise
(or if `oracle_use_heuristic=False`) → not applicable, pool arm passes
through unchanged, exactly as before.

**What changes about the system's "T1 guarantee" at `medium`/`large`
scale**: at `small` (and the ~half of `medium` with n1 ≤ 16), the
guarantee is unchanged — literal 100% T1-not-worse-than-ILS, exact. Above
that, it degrades gracefully (never regresses below "no oracle-head arm
at all", per §15.3) to **"empirically within roughly 0.2-6% of the best
solution found, in well under a second"** rather than "provably optimal"
— an honest, measured claim, not the same mathematical guarantee, but
still a real improvement over having no head arm at all for that
instance range. A full benchmark run on real `medium`/`large` curriculum
data (analogous to `logs/status_bm.md`'s small-curriculum rows) has not
yet been done — this section documents the component-level validation
that makes such a run possible, not a new end-to-end benchmark result.

## 16. Investigated and rejected (4th and final round): staged imitate-then-RL-cleanup

Fourth and final round in the training-time-hybridization series (§11-13),
testing one specific, previously-untested idea: instead of mixing a
heuristic/imitation signal WITH ongoing RL training every step (§11-13,
all failed), **separate the two into clean sequential stages** — (Phase A)
a dedicated imitation-only fine-tune on class-1 behavior (no RL loss at
all), then (Phase B) a further plain-GRPO-only phase meant to let the
policy "clean up" any imitation artifacts before evaluation. This also
tests a necessary precondition for a further idea (freeze class-1's
result, move to class-2, repeat) that the user proposed and explicitly
deferred pending this narrower result.

**Design**: Phase A = 100 steps of pure cross-entropy on class-1 decode
steps only, target = the exact T1 oracle's solution (fast and exact at
`small`-curriculum n1 sizes), linearized via §13's symmetry-mitigating
policy-preferred interleaving. Phase B = 200 further steps of ordinary
GRPO (no imitation, no forcing at all) on the same data. Total 300 steps,
matching the existing 300-step plain-GRPO control's budget exactly (fair,
matched-budget comparison, reused from prior rounds).

**Result: does not repeat the prior rounds' decisive medium-transfer
failure — but only because Phase B erases what Phase A taught, not
because it found something that survives.** Measured class-1-step
logp-gap trajectory (target vs the policy's own choice, held-out small
instances): pre-Phase-A -1.99 → after Phase A **-0.51** (genuine, strong
learning of the oracle's behavior) → after Phase B **-1.70**, reverting to
within ~0.3 nats of the plain-GRPO control's -2.02. The reversion happens
fast — the first ~25 of Phase B's 200 GRPO steps already undo most of it.

On the medium-curriculum decisive test (247 real held-out instances): the
post-Phase-A-only checkpoint is the **worst performer measured across all
four rounds of this investigation** — 82.6% lex16 loss rate vs base
(worse than §13's BC arm), T1-not-worse-vs-ILS 11.3% (down from base's
23.1%). After Phase B's cleanup, the final checkpoint recovers to
parity-with-noise against the matched-budget control (53.8% lex16 win,
23.5% T1-not-worse — nominally the best figures of any heuristic-involved
arm in this whole series, but within the run-to-run noise band this
project has repeatedly measured for this metric, e.g. §9's finding that
even ILS-vs-itself only reaches 66.6%). So the headline "doesn't lose"
result is real, but it is achieved by the model discarding the imitation
signal almost entirely, not by consolidating it into something useful.

**Verdict on the deferred multi-class staged/freeze idea: not worth
pursuing.** Its premise — that Phase A's class-1 competence could be
locked in (frozen) and built upon for class-2, then class-3 — is directly
contradicted by this data: freezing right after Phase A would lock in the
single worst checkpoint measured in the whole investigation; leaving it
unfrozen (as tested) lets RL erase it within ~25 steps. There is no
retained benefit for a freeze mechanism to protect, so the harder
engineering problem of building one (§13 already noted the architecture
has no separable per-phase module to freeze) isn't worth solving for this.

**Cross-round synthesis (§11+§12+§13+§16 — four independently-designed
attempts, closing this line of investigation)**: oracle-forcing,
LS-reward-shaping, ILS-imitation, and staged imitate-then-cleanup all
converge on the same conclusion — no way of injecting a heuristic-derived
signal into this training loop, however carefully isolated, staged, or
filtered, has produced a policy that is measurably better at `medium`-scale
generalization than plain GRPO alone. The one genuinely positive signal
surfaced anywhere in this series remains §13's `rlmed` control (plain GRPO
fine-tuned with exposure to real medium-curriculum batches, no heuristic
involved at all) — if a training-side improvement is wanted, broadening
the training curriculum itself is the lever with actual evidence behind
it, not heuristic hybridization. Recommendation unchanged from §11-13:
report the inference-time-only P1+P2(+§15) system as the hybrid
contribution; do not attempt a fifth heuristic-in-training variant
without genuinely new information (e.g. a from-scratch retrain, still
untested by any of these four smoke-scale rounds).
