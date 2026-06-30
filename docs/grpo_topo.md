# GRPO — Topology và cơ chế học

## Kiến trúc tổng quan

```
Instance (1 bài toán)
     │
     ▼
 [Encoder]  ──────────────── chạy 1 lần duy nhất (encode-once)
     │
     │ hidden state (chia sẻ cho K bản sao)
     │
  ┌──┴──────────────────────────────┐
  ▼          ▼          ▼           ▼
[Dec k=1] [Dec k=2] [Dec k=3] ... [Dec k=K]   ← K decoder độc lập, sampling ngẫu nhiên
  │          │          │           │
  ▼          ▼          ▼           ▼
route₁    route₂    route₃  ...  routeK
  │          │          │           │
  ▼          ▼          ▼           ▼
(T1,T2,T3)₁  ...                (T1,T2,T3)K   ← K reward vectors (T-vector)
     │
     ▼
 Lex rank → centered advantage ∈ [-1, +1]
 best solution (T-lex nhỏ nhất) → +1
 worst solution                 → -1
 tất cả tied                    → 0  (DAPO: skip, no gradient)
     │
     ▼
 Clipped surrogate loss (mini-batch, grad enabled)
 ratio = exp(log π_new(a) − log π_old(a))
 loss  = −min(ratio·adv, clip(ratio, 1−ε, 1+ε)·adv)
     │
     ▼
 Gradient → update policy (encoder + decoder)
```

Thực thi trong code (`src/trainers/grpo.py`):

| Bước | Code | Ghi chú |
|------|------|---------|
| Encode once | `hidden0, _ = self.policy.encoder(td0)` | B instances, no_grad |
| Batchify K | `td = batchify(td0, K)` | layout k\*B+b |
| Rollout | `out = self.policy(td, env, phase="train", hidden=hidden)` | sampling, no_grad |
| Advantage | `_centered_lex_rank_batch(T)` | (B,K) → (B,K) |
| DAPO filter | `active_mask = adv_bk.abs().amax(dim=1) > 1e-6` | bỏ group tied |
| Mini-batch loop | encode actual\_mb instances with grad, teacher-force actions | clipped surrogate |

---

## Tại sao học được: 3 cơ chế cốt lõi

### 1. Contrastive signal thay cho critic

Policy không cần biết reward tuyệt đối tốt hay xấu. Chỉ cần so sánh K solutions của *cùng một instance* với nhau. Advantage là relative rank — tự động scale về [-1, +1] bất kể đơn vị của T.

PPO cần critic (value function) để ước lượng baseline `V(s)`. GRPO dùng chính group K solutions làm baseline → loại bỏ hoàn toàn critic và variance do critic estimate gây ra.

### 2. Lexicographic rank giữ đúng priority structure

Rank theo T₁ trước, tiebreak bằng T₂, rồi T₃:

```python
# np.lexsort: last key = most significant
order = np.lexsort(T[:, ::-1].T)   # T[:,0] = T₁ là primary key
```

Gradient tự nhiên push policy ưu tiên giảm T₁ → khi T₁ bằng nhau mới care T₂ → etc.

PPO với scalar reward (weighted sum `w₁T₁ + w₂T₂ + w₃T₃`) mất cấu trúc lexicographic vì gradient trộn lẫn cả ba objectives. GRPO giữ được hierachy này mà không cần tuning weight.

### 3. Clipped ratio giữ ổn định

`clip(ratio, 1−ε, 1+ε)` ngăn policy thay đổi quá mạnh trong 1 step. Nếu không clip, một batch xấu (advantage estimate noisy) có thể phá policy hoàn toàn.

---

## Trực giác gradient

Với action sequence `a` có `adv = +1` (best trong group):

```
loss = −min(ratio·(+1), clip(ratio)·(+1))
     = −min(ratio, clip(ratio))
```

Muốn minimize loss → tăng ratio → tăng `π_new(a)` → policy học làm route đó thường xuyên hơn.

Với `adv = −1` (worst trong group):

```
loss = −min(ratio·(−1), clip(ratio)·(−1))
     = +max(ratio, clip(ratio))
```

Muốn minimize loss → giảm ratio → giảm `π_new(a)` → policy học tránh route đó.

Sau nhiều epoch: phân phối của policy lệch dần về các action dẫn đến solutions top-rank trong group → giải tốt hơn.

---

## Tie-handling (DAPO)

Khi tất cả K solutions trong một group cho cùng T-vector (tied), advantage = 0 cho tất cả → gradient = 0 → bước update vô nghĩa. DAPO filter bỏ những group này:

```python
active_mask = adv_bk.abs().amax(dim=1) > 1e-6   # (B,) — drop all-tied groups
```

Điều này xảy ra thường hơn ở giai đoạn đầu training khi policy chưa đủ diversity, hoặc với instances nhỏ có ít feasible solutions. `train/skipped_frac` log metric này.

---

## Tie-handling trong rank

Các solutions có T-vector giống nhau chia đều rank (average rank), không phá symmetry:

```
K=4, T-vectors: [A, A, B, C]  (A < B < C)
rank positions: [3, 2, 1, 0]  (best → K-1)
A tied ở pos 3,2 → average = 2.5
advantage: A→+0.67, A→+0.67, B→-0.33, C→-1.0
```

Hai solutions A nhận cùng gradient → không có spurious signal giữa các solutions bằng nhau.

---

## Tham số chính

| Param | Default | Vai trò |
|-------|---------|---------|
| `group_size` K | 8 | Số rollouts per instance. K lớn hơn → rank signal tốt hơn nhưng tốn VRAM |
| `clip_range` ε | 0.2 | PPO clip. Lớn hơn = update mạnh hơn, kém ổn định hơn |
| `mini_batch_size` | 256 | Số instances per gradient step trong inner loop |
| `lr` | 1e-4 | AdamW learning rate, chỉ update policy (không có critic) |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
