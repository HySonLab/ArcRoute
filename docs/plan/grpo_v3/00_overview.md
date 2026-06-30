# GRPO v3 — Joint M×N Vehicle-Arc Attention

## Status: PLANNED

## 1. Mục tiêu

Nâng **solution quality** (T1 trước, T2/T3 tiebreak) của RL cho HDCARP-P lên **bằng
hoặc tốt hơn ILS (500 iter)** bằng cách thay thế hoàn toàn pipeline "1 xe ảo +
Scheduler chunking" bằng kiến trúc **Joint M×N**: mỗi decode step policy học chọn
đồng thời xe nào và arc nào.

Trạng thái hiện tại (GRPO v2):

| Instance set | T1_ILS | T1_RL (v2) | Gap |
|---|---|---|---|
| small (n_req=20–30) | 2.46–4.12 | 2.56–5.48 | RL thua |

## 2. Nguyên nhân gốc (GRPO v2)

Policy là **M-agnostic**: sinh thứ tự arc `α` cho 1 xe ảo.
`Scheduler._build_P` (`scheduler.py:150`) cắt mỗi class theo **cumulative-demand
quantile** (`_balanced_chunks`, `scheduler.py:181`). Hệ quả:

- **Credit assignment mù**: gradient không biết arc rơi vào xe nào.
- **Trần T1 thấp**: chunking demand ≠ chunking time → completion time lệch → T1 cao.

## 3. Kiến trúc — Before / After

### BEFORE (v2): single virtual vehicle

```
Encoder(instance) ── H (B, N+1, D)
        │
        ▼  decode loop: while not done
   Decoder: query = f(H[current_node], state_7d)   ← 1 query
   PointerAttention(query (B,1,D), H) → logits (B, N+1)
        │ sample 1 arc j
        ▼
   action sequence α = [j1, j2, ..., 0, ...]       ← M không xuất hiện
        │
        ▼
   Scheduler._build_P: demand-quantile chunking → M routes
        │
        ▼
   T-vector (B, 3)
```

### AFTER (v3): Joint M×N

```
Encoder(instance) ── H (B, N+1, D)                 ← KHÔNG đổi
        │
        ▼  decode loop: while not served[:,1:].all()
   for m in M:  vehicle_query_m = Linear_veh([H[cur_m], rem_cap_m, comp_m, dist_depot_m])
   stack → (B, M, D)
   PointerAttention(query (B,M,D), H) → logits (B, M, N+1)
   mask 3 tầng (served | capacity(m,j) | precedence)
        │ flatten (B, M*(N+1)) → softmax → sample (m*, j*)
        ▼  1 log_p / step  ← credit assignment clean (xe + arc cùng lúc)
   update used_cap[m*], current_node[m*], served[j*]
        │
        ▼
   routes per vehicle TRỰC TIẾP (không cần _build_P)
        │
        ▼
   _completion_times → T-vector (B, 3)
```

## 4. Roadmap

```
Phase 1 (Joint M×N)  ── TARGET CHÍNH
  - action ∈ M×(N+1), 1 log_p/step, vehicle selection được học
  - train từ đầu (hoặc warm-start encoder từ v2)
  - file: 03_phase2_joint_mxn.md

Phase 2 (Scale)
  - curriculum small → medium → large
  - HRDA @ eval
  - file: 04_phase3_scale.md
```

**Quy tắc CI:** Phase N không bắt đầu implement nếu Phase N-1 chưa pass tất cả
blocking test gate (xem `05_test_plan.md`).

## 5. Success metrics

| Metric | Baseline (v2) | v3 Phase 1 (target) |
|---|---|---|
| T1 (mean, small) | RL > ILS | **RL ≤ ILS** |
| Credit assignment | mù (demand-chunk) | **learned (joint M×N)** |
| log_p / step | 1 (arc only) | 1 (m,j joint) |
| Scheduler role | chunking + time | **chỉ _completion_times** |

## 6. Tham chiếu code (đã đọc khi lập plan)

| File | Vai trò | Điểm chạm |
|---|---|---|
| `src/env/env.py` | CARPEnv | `step:36`, `get_action_mask:75`, `reset:98` |
| `src/policy/decoder.py` | ARPContext / PointerAttention / Decoder | `_state_embedding:33`, `_STATE_DIM:22`, `PointerAttention.forward:96` |
| `src/policy/encoder.py` | Encoder | `ARPInitEmbedding:13` |
| `src/policy/policy.py` | decode loop | `forward:39`, loop `:88` |
| `src/solvers/scheduler.py` | Φ map | `_build_P:150`, `_balanced_chunks:181`, `_completion_times:273` |
| `src/utils/local_search.py` | LS operators | `ls:254`, `lsRL:307` |
| `src/trainers/grpo.py` | GRPO trainer | `_train_step:168`, inner loop `:238` |
| `src/utils/ops.py` | helpers | `batchify:76`, `get_log_likelihood:92` |
| `src/solvers/cal_reward.py` | reward bridge | `calc_reward:43` |
