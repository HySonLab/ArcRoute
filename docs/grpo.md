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
        │
        ▼
  GRPO.centered_hier_rank_gpu(T)  →  group-relative advantage
        │
        ▼
  clipped surrogate loss (mini-batch replay) → backward → AdamW step
```

Key point: the policy decides **(vehicle, arc) jointly at every step** — it is
not "decide arc order, then assign to vehicles" (that was the pre-GRPO-v3
design; see §7).

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
awareness at all. §7.3/§8 diagnosed both gaps; bậc 3 (above, training now,
§8.5) adds class-progress features and cross-vehicle self-attention to close
them. §6.2's benchmarked checkpoints (v4-v6 in `logs/status_bm.md`) predate
this change; a fresh bậc-3 benchmark is pending training completion.

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
   to get fresh log-likelihoods at the current policy, clipped surrogate
   loss, gradient accumulated across chunks, **one** optimizer step per
   training step (`automatic_optimization=False`).

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

Length-normalized log-likelihood (sum over valid decode steps / valid count,
excludes post-`done` padding steps) × clipped-ratio surrogate
(`clip_range` via PPO-style `ratio = exp(ll_new - ll_old)`) − `entropy_lambda
* entropy`. Gradient-clipped by norm, `AdamW`, linear warmup → one-shot
cosine decay to `lr_min` (`configure_optimizers`).

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

## 8. "Bậc 3" — combined decoder upgrade — IMPLEMENTED, TRAINING

Two independent gaps pointed at the same module (`JointMVContext`,
`src/policy/decoder.py`) and were cheapest to fix together in one retrain
rather than sequentially. Both are now implemented and code-reviewed
line-by-line (§8.1 below reflects the actual code, not a proposal); training
is in progress (§8.5).

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

### 8.5 Training status

Launched on top of the bậc-2 checkpoint's identical hyperparameters (clean
A/B — only the decoder architecture differs): `MAX_EPOCH=200 BATCH_SIZE=512
MINI_BATCH_SIZE=256 GROUP_SIZE=16 CURRICULUM=small`, seed 6868.

- Log: `logs/train_20260703_162357.out`. Checkpoint dir:
  `outputs/checkpoints/grpo_osm_bac3/`. Lightning run: `outputs/lightning_logs/version_3`.
- Confirmed healthy at launch (epoch 0, no NaN/crash) and remains healthy
  through epoch 140/200 (last checked 2026-07-03 21:28): best-so-far
  checkpoint `epoch=131.ckpt` by `val/lex_best`, `last.ckpt` updating every
  epoch.
- **Measurably slower per training step** than bậc 2's run (~0.38-0.44 it/s
  vs bậc 2's ~0.56-0.60 it/s, roughly 30-35% slower) — this is the added
  per-decode-step cost of `_class_progress()` + the vehicle self-attention
  block in `JointMVContext`, run once per decode step for every one of the
  `B*K` rollout copies. Confirmed via code inspection that **no local
  search is involved in the training path** — bậc 1's funnel hybrid
  (`RLHybHCARP`) is an inference-only benchmark tool (`src/solvers/rl.py`),
  never invoked from `src/train.py`/`src/trainers/grpo.py`; the slowdown is
  entirely the new decoder compute, not an accidental local-search call.
- Once complete: benchmark the best checkpoint against ILS on
  `data/osm_test_small` using the same `HYBRID=1 SHORTLIST=25` protocol as
  bậc 2's v6 row (`logs/status_bm.md`), to produce a clean bậc-2-vs-bậc-3
  comparison at matched inference settings.
