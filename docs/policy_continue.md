# Policy Architecture — Findings & Improvement Plan

## Context

Training: GRPO, K=16, HDCARP-P (P=3 priority classes, lex-minimize T1≺T2≺T3).  
Architecture: 12-layer Transformer encoder + attention decoder (M-agnostic), Scheduler assigns arcs → M vehicles post-hoc.

---

## Measurements (Session 3)

### Phase 0 Baseline — 20 OOD instances, osm_cityA/40, M=2, epoch=164

```
scripts/measure_scheduler_gap.py --n_instances 20 --n_ils_samples 30
                                 --n_policy_samples 64 --n_rand_perms 500
```

#### Per-instance results

| Instance | T1_ILS | T2_ILS | T3_ILS | T1_policy | T2_policy | T3_policy |
|---|---|---|---|---|---|---|
| 38_17_358 | 6.746 | 15.257 | 25.466 | 6.687 | 15.769 | 32.910 |
| 38_18_975 | 5.368 | 14.161 | 15.989 | 6.231 | 15.915 | 18.697 |
| 38_19_855 | 2.089 | 5.060 | 9.037  | 2.907 | 5.315  | 12.104 |
| 38_20_345 | 7.459 | 19.208 | 27.204 | 7.143 | 16.140 | 28.301 |
| 38_20_374 | 4.638 | 9.173  | 13.206 | 4.666 | 9.412  | 14.613 |
| 38_20_986 | 4.888 | 10.375 | 16.022 | 4.865 | 9.235  | 13.805 |
| 39_17_864 | 2.966 | 9.158  | 16.358 | 3.055 | 7.850  | 13.343 |
| 39_18_205 | 6.695 | 12.642 | 16.941 | 6.864 | 12.162 | 15.707 |
| 40_18_125 | 5.173 | 11.462 | 19.395 | 6.217 | 11.145 | 17.491 |
| 40_18_882 | 5.719 | 13.235 | 16.783 | 5.930 | 11.987 | 17.695 |
| 40_20_566 | 7.018 | 18.087 | 27.116 | 7.278 | 16.959 | 27.738 |
| 40_21_334 | 1.908 | 6.578  | 9.322  | 2.207 | 7.237  | 9.241  |
| 40_21_567 | 1.533 | 5.846  | 7.962  | 1.969 | 6.336  | 10.225 |
| 41_17_284 | 7.761 | 17.446 | 27.957 | 10.106| 22.691 | 31.276 |
| 41_20_441 | 3.311 | 9.808  | 17.399 | 3.830 | 8.764  | 15.600 |
| 42_16_031 | 7.427 | 14.342 | 24.979 | 8.015 | 16.402 | 26.508 |
| 42_17_223 | 6.675 | 16.402 | 24.647 | 7.678 | 14.729 | 24.436 |
| 42_17_285 | 9.147 | 16.799 | 26.704 | 8.053 | 15.421 | 27.552 |
| 42_18_636 | 5.559 | 11.884 | 20.033 | 4.805 | 11.317 | 19.679 |
| 42_21_150 | 7.610 | 18.275 | 26.139 | 8.192 | 15.009 | 22.506 |

#### Summary (mean across 20 instances)

| Method | T1 | T2 | T3 |
|---|---|---|---|
| ILS direct (ground truth) | **5.4845** | 12.7600 | 19.4330 |
| Scheduler(ILS order) | 5.4845 | 12.7600 | 19.4330 |
| **Policy v1 epoch=164** | **5.8348** | **12.4897** | **19.9713** |
| Rand-best (500 perms) | 5.9659 | 15.0339 | 23.4359 |

#### Gap decomposition

| Gap | T1 | T2 | T3 | Interpretation |
|---|---|---|---|---|
| Gap_A (Scheduler assignment) | +0.000 | +0.000 | +0.000 | Scheduler perfect — not a bottleneck |
| Gap_order (Policy vs ILS) | **+0.350** | **−0.270** | +0.538 | T1 bottleneck; policy T2 beats ILS |
| Gap_rand (Policy vs rand-best) | −0.131 | −2.544 | −3.465 | Policy already better than random |

**Notable:** Policy v1 outperforms ILS on T2 (−0.270). ILS optimizes lex(T1≺T2≺T3) so once T1 is minimized, T2 may not be fully optimized. T1 gap of +6.4% is the primary target.

**Conclusion:** Gap_A = 0 confirms Scheduler is perfect. Bottleneck is T1 ordering quality. Policy arc ordering is already near-locally-optimal (Gap_rand < 0) — the gap vs ILS comes from structural limitations (encoder adj bug, M-blindness, no local search).

---

## Root Cause Analysis (Opus review)

Three independent sources of the ~6% T1 gap:

### (a) Bug in encoder — adj multiply wrong direction

```python
# encoder.py:197 — CURRENT (WRONG):
attn_weights = attn_weights * adj.unsqueeze(1)
# Comment says "mask out non-neighbors" but adj = Floyd-Warshall distances (continuous).
# Effect:
#   - Diagonal = 0  → self-attention zeroed out at every layer
#   - Farther arcs  → higher attention score (wrong direction)
#   - Large distances → clamp ±1e4 → softmax saturates
#   - 12 encoder layers waste capacity compensating for this distortion
```

Fix:
```python
# Additive negative bias (closer arc → attend more):
attn_weights = attn_weights + self.dist_bias * adj.unsqueeze(1)
# dist_bias: nn.Parameter(shape=(num_heads,1,1)), init = -0.1
```

### (b) Decoder context too sparse — M-blindness

`ARPContext` query at each step = `project([current_node_embed (128), remaining_cap (1)])`.  
Policy sees **1 scalar** of dynamic state. Missing:
- Fraction of arcs served per class → doesn't know priority distribution of remaining work
- M (fleet size) → can't reason about how Scheduler will split into M vehicles
- Dist to depot → can't reason about vehicle-change cost
- Fraction of capacity used in current class → can't balance load across vehicles

### (c) No local search (structural)

ILS wins partly because it has iterated local search. A pure construction policy cannot match a metaheuristic on ordering quality alone. This is partially addressable via POMO multi-start or an improvement operator.

---

## v2 Architecture Changes

### Fix 1 — Encoder adj bias (`encoder.py`)

Replace multiplicative adj with learnable additive negative bias per head:

```python
# In MultiHeadAttentionCompact.forward():
# Remove: attn_weights = attn_weights * adj.unsqueeze(1)
# Add:
attn_weights = attn_weights + self.dist_bias * adj.unsqueeze(1)
# dist_bias: nn.Parameter((num_heads, 1, 1)), init = -0.1
```

Cost: 1 param per head, no speed change. Breaks checkpoint.

### Fix 2 — Extended decoder context (`decoder.py` + `env.py`)

`ARPContext._state_embedding` returns 7 scalars instead of 1:

```python
state = cat([
    vehicle_capacity - used_capacity,   # 1 — remaining capacity (already present)
    frac_served_cls1,                   # 1 — fraction of class-1 arcs served
    frac_served_cls2,                   # 1 — fraction of class-2 arcs served
    frac_served_cls3,                   # 1 — fraction of class-3 arcs served
    dist_to_depot,                      # 1 — adj[current_node, 0]
    M / 10.0,                           # 1 — fleet size normalized
    frac_cap_current_class,             # 1 — load fraction within current class
], dim=-1)
# project_context: Linear(embed_dim + 7, embed_dim)
```

Env changes:
- `reset()`: add `clss_total` (B,P), `clss_served` (B,P)=0 to td
- `step()`: `clss_served[b, cls-1] += 1` for non-depot actions

Cost: ~35 lines (env + decoder). Breaks checkpoint (Linear shape change).

---

## Step-by-Step Plan

### Phase 0 — Baseline benchmark ✅ DONE

Ran on 20 OOD instances (osm_cityA/40, M=2, epoch=164). Results in Measurements section above.

**Key numbers to beat with v2:**
- T1: 5.8348 → target < 5.4845 (ILS)
- T2: 12.4897 (already beats ILS 12.7600 — preserve this)
- T3: 19.9713 → target < 19.4330 (ILS)

To re-run after v2 trains:
```bash
uv run python scripts/measure_scheduler_gap.py \
    --ckpt outputs/checkpoints/<v2_phase>/epoch=NNN.ckpt \
    --ood_dir data/ood/osm_cityA/40 \
    --n_instances 20 --n_ils_samples 30 \
    --n_policy_samples 64 --n_rand_perms 500 --vehicles 2
```

### Phase 1 — Implement v2 architecture (while v1 finishes)

1. Fix `src/policy/encoder.py`: adj multiplicative → additive negative bias
2. Fix `src/policy/decoder.py`: extend `ARPContext` to 7-scalar state
3. Fix `src/env/env.py`: track `clss_served`, `clss_total` in reset/step
4. Update `scripts/train.py`: pass new hyperparameters if needed

### Phase 2 — Ablation (after v1 finishes, before full v2 retrain)

Test only **distance logit bias at decoder** (checkpoint-compatible, strict=False load):
- Adds `dist_scale = nn.Parameter(0.0)` to decoder logits
- Load v1 best checkpoint → fine-tune 30 epochs
- Measures: does T1 drop >2%?

Interpretation:
- Yes → decoder deadhead blindness is significant even with encoder bug
- No → encoder adj bug is the primary root cause

### Phase 3 — Train v2 curriculum from scratch

v1 → v2 warm-start is NOT possible (Linear shapes changed). Start Phase 1 fresh.

```bash
bash scripts/train_curriculum.sh   # 3 phases × 200 epochs with v2 architecture
```

| Phase | Data | Epochs | Start from |
|---|---|---|---|
| Phase 1 (small) | 20:40 | 200 | scratch |
| Phase 2 (medium) | 20:40, 30:60, 40:80 | 200 | v2 phase 1 best |
| Phase 3 (full) | all sizes | 200 | v2 phase 2 best |

### Phase 4 — Benchmark v1 vs v2

After v2 phase 1 completes (~23h), run same benchmark on 20 OOD instances:

| Method | T1 | T2 | T3 |
|---|---|---|---|
| ILS (baseline) | — | — | — |
| v1 epoch=best | baseline | baseline | baseline |
| v2 phase1 epoch=best | ??? | ??? | ??? |

If v2 phase 1 > v1 → continue to phases 2, 3.  
If not → review fixes, tune dist_bias init, check env tracking.

### Phase 5 — POMO multi-start (orthogonal, add to v2 if phase 4 positive)

Change GRPO rollout from K random starts on same instance → K diverse starting arcs.  
Does not change architecture, adds ~10 lines to `grpo.py`. Independent gain expected.

---

## Timeline

```
✅ Phase 0 done   Baseline: T1=5.83, T2=12.49, T3=19.97 (v1 epoch=164)
                  ILS ref:  T1=5.48, T2=12.76, T3=19.43

   Now            v1 epoch 165→200 (~35 epochs remaining, ~3.5h)
                  → Implement v2 architecture in parallel

v1 done           Run Phase 2 ablation (Option A, 30 epochs, ~3h)
                  → Confirm whether deadhead blindness is significant
                  → Start v2 Phase 1 training

v2 Phase 1 done   Benchmark v2 vs v1 on same 20 OOD instances
(~23h later)      → if T1 < 5.83: positive, continue Phase 2/3
                  → if not: investigate

v2 full done      Full benchmark: OOD + bench_small/medium/large val data
```

---

## Files to Modify

| File | Change |
|---|---|
| `src/policy/encoder.py:197` | adj multiply → additive negative bias |
| `src/policy/decoder.py:18,25` | Linear(embed+1) → Linear(embed+7); state_embedding returns 7 scalars |
| `src/env/env.py:60,115` | step: update clss_served; reset: add clss_total, clss_served |
| `scripts/train.py` | expose new hyperparams if needed |
| `scripts/train_curriculum.sh` | no change needed |

---

## What NOT to do

- **Option B/B' (adj row in encoder input)**: redundant (adj already in attention), breaks variable-N batching
- **Option E (reward shaping deadhead)**: biases lex-objective, high risk of instability
- **v1 → v2 warm-start**: Linear shape mismatch, don't attempt
