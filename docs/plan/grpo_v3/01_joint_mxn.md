# Phase 2: Joint M×N Vehicle-Arc Attention (TARGET CHÍNH)

## Status: PLANNED

Mục tiêu: vehicle selection **được học** thay vì heuristic argmin. Mỗi decode step,
policy chọn **một cặp `(vehicle m, arc j)`** từ không gian `M×(N+1)`. Một `log_p`
duy nhất / step → GRPO credit assignment sạch tới cả xe lẫn arc.

```
Score(m, j) = ⟨vehicle_query_m, arc_key_j⟩ / √D
logits (B, M, N+1) → flatten (B, M*(N+1)) → softmax → sample (m*, j*)
```

---

## 1. Core architecture (decode loop step t)

```python
# Tiền điều kiện: encoder đã chạy 1 lần → H = cached.node_embeddings (B, N+1, D)
#   glimpse_key/val/logit_key đã precompute (decoder._precompute_cache, decoder.py:241)

# ----- 1. VEHICLE STATE EMBEDDINGS  (B, M, D) -----
ar = torch.arange(B)
veh_feats = []                                        # build (B, M, state_dim)
for m in range(M):
    cn_m   = current_node[:, m]                       # (B,)
    h_cn_m = H[ar, cn_m]                              # (B, D) node embedding xe m
    rem_m  = (1.0 - used_cap[:, m])[:, None]          # (B, 1)
    comp_m = (completion[:, m] / comp_norm)[:, None]  # (B, 1)
    ddep_m = adj[ar, cn_m, 0][:, None]                # (B, 1) dist tới depot
    veh_feats.append(torch.cat([h_cn_m, rem_m, comp_m, ddep_m], -1))  # (B, D+3)
vehicle_state = torch.stack(veh_feats, dim=1)         # (B, M, D+3)
vehicle_embeds = Linear_veh(vehicle_state)            # (B, M, D)  ← head MỚI

# ----- 2. M×N LOGITS  (B, M, N+1) -----
glimpse_q = vehicle_embeds + graph_context            # (B, M, D)  M queries
logits = PointerAttention(
    query     = glimpse_q,                            # (B, M, D)
    key       = glimpse_key,                          # (B, N+1, D)
    value     = glimpse_val,                          # (B, N+1, D)
    logit_key = logit_key,                            # (B, N+1, D)
    attn_mask = attn_mask_BMN,                        # (B, M, N+1) bool
)                                                     # → (B, M, N+1)

# ----- 3. MASK  (B, M, N+1) -----  (xem mục 2 — guard infeasibility)
# ----- 4. SAMPLE 1 ACTION từ M×(N+1) -----
logits_flat = (logits + mask_add).reshape(B, M * (N + 1))   # mask_add: 0 hoặc -inf
log_p_flat  = log_softmax(logits_flat, dim=-1)              # (B, M*(N+1))
action_flat = sample(log_p_flat)                            # (B,)
m_star = action_flat // (N + 1)                             # (B,)
j_star = action_flat %  (N + 1)                             # (B,)
log_p_step = log_p_flat.gather(1, action_flat[:, None]).squeeze(1)   # (B,)

# ----- 5. UPDATE STATE -----
is_depot = (j_star == 0)
dem_j    = gather_by_index(demand, j_star[:, None], squeeze=True)
used_cap[ar, m_star] = (used_cap[ar, m_star] + dem_j) * (~is_depot).float()
step_cost = adj[ar, current_node[ar, m_star], j_star] + service[j_star]
completion[ar, m_star] += step_cost
current_node[ar, m_star] = j_star
visited.scatter_(-1, j_star[:, None, None], 1)        # served global (depot bỏ qua tự nhiên)

# ----- 6. DONE -----
done = visited[..., 1:].sum(-1) == (N)                # served hết arc không-depot
```

`comp_norm`: `completion.amax(-1, keepdim=True).clamp(min=1e-6)` (tương đối giữa xe).

---

## 2. Infeasibility guard — chứng minh không bao giờ all-inf

Mask 3 tầng trên `(B, M, N+1)`:

```
# Layer 1 — served arcs (broadcast over M):
mask[:, :, 1:][served]            = -inf        # cột arc đã serve
# Layer 2 — capacity per (m, j):
mask[b, m, j] = -inf  if used_cap[b,m] + demand[j] > 1.0   (j ≥ 1)
# Layer 3 — precedence (variant P, broadcast over M):
mask[:, :, 1:][clss_j > c_min]    = -inf        # arc thuộc class cao hơn c_min
# Bất biến depot (cột j=0):
mask[b, m, 0] = -inf  if used_cap[b,m] == 0     # cấm "đứng yên" ở depot khi chưa rời
mask[b, m, 0] = 0     if used_cap[b,m] > 0      # cho phép reload
```

**Định lý (luôn tồn tại ≥1 ô feasible khi chưa done).** Với mỗi instance còn arc
chưa serve:

1. **Tồn tại xe đã rời depot** (`used_cap_m > 0`): cột depot `j=0` của xe đó luôn
   feasible — không bị Layer 1 (depot không nằm trong `served`), không bị Layer 2
   (`demand[0]=0`), không bị Layer 3 (depot không có class). ⇒ ≥1 ô feasible.

2. **Mọi xe đều ở depot** (`used_cap_m = 0 ∀m`, đầu rollout hoặc sau khi tất cả vừa
   reload): còn ≥1 arc chưa serve của class `c_min`. Arc đó: không bị Layer 1
   (chưa serve), không bị Layer 3 (`clss_j == c_min`), và Layer 2 cần `demand[j] ≤
   1.0` — **đúng** vì dữ liệu đảm bảo mọi arc demand ≤ Q (capacity normalized 1.0;
   `import_instance` chia demand cho C, và bài toán khả thi yêu cầu mỗi arc vừa 1 xe).
   ⇒ ô `(bất kỳ m, j)` feasible.

Hai trường hợp phủ hết ⇒ **không bao giờ all-inf** trên `M×(N+1)`. (So với v2 một
query, guard này mạnh hơn vì có M cột depot dự phòng.)

**Edge case** capacity-infeasible instance (Σdemand > M·1.0): bị loại từ data
generation (`ops.check_feasibility`, `ops.py:363`) → không xảy ra trong train/eval.

---

## 3. Thay đổi `decoder.py`

### `PointerAttention.forward` (`decoder.py:96`) — hỗ trợ M queries

Hiện tại `query` shape `(B, 1, D)` (single step), `attn_mask` `(B, N+1)` hoặc
`(B,1,N+1)`. Output sau `squeeze(-2)` → `(B, N+1)`.

Với Joint M×N, `query` shape `(B, M, D)`:

| | Trước | Sau |
|---|---|---|
| `query` | `(B, 1, D)` | `(B, M, D)` |
| `attn_mask` | `(B, 1, N+1)` | `(B, M, N+1)` |
| `_inner_mha` heads | `(B, 1, D)` | `(B, M, D)` |
| `logits` (sau bmm) | `(B, N+1)` | `(B, M, N+1)` |

`_inner_mha` (`decoder.py:112`) dùng `rearrange(v, "... g (h s) -> ... h g s")` —
**đã generic theo `g`** (số query), nên M queries chạy được không cần sửa. `attn_mask`
xử lý `unsqueeze(1)` khi `ndim==3` (`decoder.py:118-122`) → `(B,1,M,N+1)` broadcast
qua heads — OK.

`forward` (`decoder.py:103`): `logits = bmm(glimpse, logit_key...).squeeze(-2)`.
Với glimpse `(B, M, D)` và logit_key `(B, N+1, D)`:
`bmm(glimpse, logit_key.transpose(-2,-1))` → `(B, M, N+1)`. **Bỏ `squeeze(-2)`**
(nó chỉ đúng khi M=1). Plan: thay `.squeeze(-2)` bằng giữ nguyên `(B, M, N+1)`;
khi M=1 vẫn tương thích bằng cách squeeze có điều kiện. → tạo path riêng hoặc
parametrize.

### Head mới: `Linear_veh`

`nn.Linear(D + 3, D)` (3 = remaining_cap, completion_norm, dist_depot). Đặt trong
`Decoder.__init__` (`decoder.py:144`) hoặc một `JointMNContext` module thay
`ARPContext`. `ARPContext` cũ (single-query, state 7/8d) **không dùng** ở Phase 2;
giữ lại cho Phase ≤1.

### `Decoder.forward` (`decoder.py:217`)

Xây `glimpse_q` (B,M,D) từ M vehicle states + graph_context, gọi `pointer` rồi trả
`(logits (B,M,N+1), mask (B,M,N+1))`. `decode_strategy.step` cần nhận logits 3D →
xem mục 5.

---

## 4. Thay đổi `get_log_likelihood` (`ops.py:92`)

Hiện tại (`ops.py:103-104`):
```python
if actions is not None and logprobs.dim() == 3:
    logprobs = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
```
Giả định `logprobs (B, seq, N+1)` và `actions (B, seq)` index theo arc.

Với Joint M×N, mỗi step lưu `log_p_flat (B, M*(N+1))` và `action_flat (B,)` trong
`[0, M*(N+1))`. Stack qua step → `logprobs (B, seq, M*(N+1))`, `actions (B, seq)`.

| | Trước | Sau |
|---|---|---|
| logprobs dim cuối | `N+1` | `M*(N+1)` |
| actions giá trị | arc index `[0,N]` | flat index `[0, M*(N+1))` |
| gather | `gather(-1, actions)` | **giống hệt** — gather theo flat index |

→ `get_log_likelihood` **không cần sửa logic gather** (vẫn `gather(-1, action)` trên
dim cuối). Chỉ cần đảm bảo decode_strategy lưu `log_p_flat` (đã flatten) làm
`logprobs`, và `action_flat` làm action. `return_sum` (`ops.py:111`) vẫn sum qua
step → `ll (B,)`. Inner-loop `ll.sum(dim=-1)` khi `return_sum=False` → cũng OK.

**Quan trọng — log_p của trajectory = Σ log_p_step.** Vì mỗi step softmax chuẩn hóa
trên toàn bộ `M*(N+1)` (joint), `log_p_step` là log-prob của **cặp (m*,j*)**. Tổng
qua step cho log-prob trajectory đúng (chain rule). Đây là điểm test
`TestLogProbCorrectness` kiểm.

---

## 5. Thay đổi `decode_strategy.py`

`process_logits` (`decode_strategy.py:34`) và `_step` giả định logits 2D `(B, A)`.
Với joint, ta **flatten trước khi vào decode_strategy**: Decoder trả logits đã
reshape `(B, M*(N+1))` và mask `(B, M*(N+1))`. Khi đó decode_strategy không cần biết
cấu trúc M×N — nó chỉ thấy action space phẳng kích thước `M*(N+1)`.

- `greedy` / `sampling` (`decode_strategy.py:201-226`): chạy nguyên trên dim phẳng.
- Sau khi có `selected_action` (flat), env.step tự tách `m_star, j_star`.
- `store_all_logp=True` (train, return_entropy) → logprobs lưu full `(B, M*(N+1))`
  để tính entropy; `store_all_logp=False` (rollout) → gather về `(B,)`.

→ **decode_strategy gần như không đổi**; thay đổi nằm ở chỗ Decoder flatten output
và env.step un-flatten action.

---

## 6. Thay đổi `grpo.py`

- `logprobs` stash (`grpo.py:198`): rollout với `store_all_logp=False` → vẫn `(B*K,)`
  sum log-prob (không đổi shape). ✅
- Inner loop (`grpo.py:257-268`): policy chạy với `return_entropy=True` →
  `store_all_logp=True` → `out_i["log_likelihood"]` shape `(B*K, seq)` (per-step,
  `return_sum_log_likelihood=False`). `ll.sum(dim=-1)` (`grpo.py:268`) sum qua step
  → `(B*K,)`. **Vẫn work** vì gather trong `get_log_likelihood` đã chọn đúng flat
  action mỗi step.
- **Entropy** (`decode_strategy.calculate_entropy`, `decode_strategy.py:86`): tính
  trên dim cuối `M*(N+1)` per step → entropy của phân phối joint. Mean qua step
  (`decode_strategy.py:92`). Lớn hơn entropy v2 (action space M× rộng hơn) → có thể
  cần giảm `entropy_lambda` (hiện 0.01, `grpo.py:151`). Plan: A/B `entropy_lambda ∈
  {0.003, 0.01}`.
- `_eval_step` (`grpo.py:294`): greedy joint decode → routes trực tiếp → reward.
  Thêm HRDA (Phase 3) ở đây.
- **Reward path:** action giờ là flat `(B*K, seq)` joint index. `calc_reward` cần
  routes per-vehicle. Env phải decode `m_star,j_star` từ flat action và ghi
  `action_vehicle` (như Phase 1) → Scheduler `routes_from_actions` (xem
  `02_phase1_sequential.md` §4) → `_completion_times`. Không dùng `_build_P` nữa.

---

## 7. Warm-start strategy (từ Phase 1)

1. Load checkpoint Phase 1 (encoder + arc decoder: `project_node_embeddings`,
   `project_fixed_context`, `pointer.project_out`, encoder layers).
2. **Freeze** encoder + arc-side decoder weights.
3. Khởi tạo `Linear_veh` (head vehicle-query mới) ngẫu nhiên; **chỉ train head này**
   5–10 epoch (để query M-vehicle học căn chỉnh với key arc đã pretrain).
4. **Unfreeze all**, train tới hội tụ với lr nhỏ hơn (vd lr/3).

Mapping weight: Phase 1 `ARPContext.project_context (D+8 → D)` không tái dùng trực
tiếp cho `Linear_veh (D+3 → D)` (input khác). Nhưng `glimpse_key/val/logit_key`
(`project_node_embeddings`, `decoder.py:183`) và `pointer` tái dùng được hoàn toàn —
đây là phần "hiểu arc" quan trọng nhất.

---

### Test Gate

| Test name | File | Pass criterion | Blocking? |
|---|---|---|---|
| `TestJointMxNInfeasibility` | `tests/test_phase2.py` | 100 random instance, decode tới done: KHÔNG step nào all-inf logits | Yes |
| `TestLogProbCorrectness` | `tests/test_phase2.py` | manual trace 1 rollout: `sum(log_p_steps) == log_p(trajectory)` (atol 1e-5) | Yes |
| `TestMaskCorrectness::test_served` | `tests/test_phase2.py` | sau chọn (m*,j*): `served[j*]=True`, không step sau chọn lại j* | Yes |
| `TestMaskCorrectness::test_capacity` | `tests/test_phase2.py` | không step nào chọn arc vượt capacity của xe m* | Yes |
| `TestJointShapes` | `tests/test_phase2.py` | logits `(B,M,N+1)`, flat `(B,M*(N+1))`, action split đúng m*,j* | Yes |
| `TestGRPOIntegrationP2` | `tests/test_grpo.py` | train 2 step không crash, grads finite, entropy finite | Yes |
| `TestBenchmarkP2` | `tests/test_phase2.py` | epoch 50: mean T1 Phase 2 ≤ mean T1 Phase 1 (5 instance) | No (perf) |
</content>
