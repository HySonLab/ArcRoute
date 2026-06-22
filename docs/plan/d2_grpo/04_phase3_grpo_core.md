# Phase 3 — **GRPO core** (K-sample group + centered lexicographic rank + adv=rank, bỏ critic)

> Mục tiêu: thêm path **GRPO** vào `rl/ppo.py` sau cờ `--algo grpo`: roll out **K sample/instance** (nhóm),
> reward = **centered lexicographic rank** trong nhóm K, **advantage = rank** (group mean = baseline → bỏ
> critic, `vf_lambda→0`). Giữ surrogate clip + ratio + entropy. **Path PPO cũ giữ nguyên** (A/B). Phụ thuộc
> Phase 2 (vector reward). Code: `rl/ppo.py`.

## 3.1 — Hiện trạng PPO (đã xác nhận, `rl/ppo.py`)

- `shared_step` (`:157`): rollout no-grad (`:161-163`) → set `logprobs/reward/action` (`:180-182`) →
  vòng mini-batch PPO (`:185-247`).
- Advantage **bandit**: `adv = previous_reward - value_pred.detach()` (`:207-208`); critic 1 scalar head
  (`rl/critic.py:18-19`).
- Loss = surrogate + `vf_lambda*value_loss - entropy_lambda*entropy` (`:229-233`); `vf_lambda=0.5` default
  (`:26`), `normalize_adv=False` (`:28`).
- `configure_optimizers` (`:148-155`) gộp `policy + critic` params.

## 3.2 — Thiết kế GRPO (sau cờ `--algo`)

Thêm `algo='ppo'` vào `PPO.__init__` (lưu `self.algo`). Trong `shared_step`, **rẽ nhánh** path train.

### (a) Roll out nhóm K — POMO-style qua `batchify`
```python
from common.ops import batchify, unbatchify
K = self.ppo_cfg["group_size"]
td0 = self.env.reset(batch)              # (B, ...)
td  = batchify(td0, K)                   # (B*K, ...) — mỗi instance lặp K lần
with torch.no_grad():
    out = self.policy(td.clone(), self.env, phase="train")   # decode_type=train=sampling
# env.reward_mode='vector' (Phase 2) -> out["reward"] là (B*K, 3) T-vector
```
⚠️ **Verify layout `batchify`** ([`01_phase0 §0.4`](01_phase0_scope.md)): `_batchify_single`
(`common/ops.py:71-74`) cho layout cụ thể; dùng **`unbatchify(x, K)`** (`common/ops.py:122`, nghịch đảo
chuẩn của batchify) để về `(B, K, ...)` — **không tự reshape tay**. Test layout (gate #2) khoá đúng.

### (b) Centered lexicographic rank trong nhóm K
```python
T = out["reward"].view(B, K, 3).cpu().numpy()        # (B,K,3) T-vector (qua unbatchify đúng trục)
adv = np.empty((B, K), dtype=np.float32)
for b in range(B):
    order = np.lexsort((T[b,:,2], T[b,:,1], T[b,:,0]))   # index tốt->xấu (T_1 primary)
    rank = np.empty(K); rank[order] = np.arange(K-1, -1, -1)   # tốt nhất -> K-1
    adv[b] = (rank - (K-1)/2) / ((K-1)/2)               # [-1,1], mean=0, tốt nhất = +1
adv = torch.as_tensor(adv).view(B*K, 1).to(device)
```
> **Dấu** ([`01_phase0 §0.3`](01_phase0_scope.md)): nghiệm T thấp ⇒ rank cao ⇒ **adv dương** ⇒ surrogate
> tăng xác suất (PPO maximize). `K=1` ⇒ adv=0 (đặt guard, GRPO yêu cầu `K≥2`).

### (c) Advantage = rank, **bỏ critic**
- Set `sub_td["advantage"] = adv` thay vì tính từ critic. Trong vòng mini-batch (`rl/ppo.py:187-247`),
  với `algo=='grpo'`:
  - **không gọi** `self.critic(sub_td)`; `adv = sub_td["advantage"]` (đã centered).
  - **value_loss = 0** (hoặc skip), `vf_lambda` ép 0 ⇒ loss = `surrogate - entropy_lambda*entropy`.
  - giữ `ratio = exp(ll.sum(-1) - logprobs)` (`:202`), surrogate clip (`:215-223`) **NGUYÊN**.
- `configure_optimizers`: giữ `policy + critic` (khỏi vỡ load_from_checkpoint); critic **không nhận
  gradient** ở GRPO (không vào loss) → an toàn.
- ⚠️ `normalize_adv`: GRPO rank **đã** zero-mean/unit-scale → để `normalize_adv=False` (default) cho path
  này (tránh chuẩn hoá kép).

### (d) td plumbing
- `td.set("reward", out["reward"])` với (B*K,3) — chỉ dùng để **rank**, **không** đưa thẳng làm scalar.
- `td.set("logprobs"/"action", ...)` từ `out` (B*K). Mini-batch loop chạy trên B*K rows.
- ⚠️ `B*K ≥ 10` (run_parallel, `env/env.py:147,153`) — đảm bảo trong cả train (B lớn) lẫn smoke.

## 3.3 — Cờ + config
- `PPO.__init__`: `algo='ppo'`, `group_size=8`; vào `self.ppo_cfg`.
- `--algo grpo` ⇒ env phải `reward_mode='vector'` (Phase 2). `train.py` set khi `algo=='grpo'` (Phase 4).
- `--algo ppo` (default) ⇒ path cũ **không đổi 1 byte** (rẽ nhánh sớm).

---

## ✅ Cổng test Phase 3

**File test mới:** `tests/test_grpo.py`.

1. **⭐ Rank đúng (ví dụ tay):** T `(1,K,3)` = `[[5,9,9],[5,2,9],[5,2,1],[7,0,0]]` (K=4) → hàm rank trả
   nghiệm `[5,2,1]` **adv lớn nhất** (=+1, tốt nhất), `[7,0,0]` **adv nhỏ nhất** (=-1). adv.mean()≈0.
2. **⭐ Layout batchify/unbatchify:** `batchify(td0,K)` rồi `unbatchify(.,K)` về (B,K,...) **khớp** từng
   instance (so `td0` vs `unbatchify` slice) → reshape (B,K,3) đúng trục, không trộn instance.
3. **⭐ Advantage finite + zero-mean per group:** với T ngẫu nhiên (B,K,3), `adv.view(B,K).mean(1)≈0`,
   `adv.abs()≤1`, finite.
4. **⭐ Backward + gradient flows:** GRPO `shared_step` (B=2,K=8 ⇒ 16≥10) chạy → loss finite, `loss.backward`
   (hoặc manual_backward) → **policy params có grad không None/không NaN**; **critic params KHÔNG có grad**
   (không vào loss).
5. **⭐ Path PPO cũ vẫn chạy:** `--algo ppo` (default) `shared_step` train → loss finite, **critic CÓ grad**
   (chứng minh không đụng path cũ).
6. **⭐ Rollout smoke GRPO:** `env(reward_mode='vector') → reset → batchify(K) → policy → rank → backward`
   finite, ở **variant `P`** rồi lặp **`U`**; `B*K≥10`.
7. **Suite cũ xanh:** 90 test (default algo=ppo, reward scalar).

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_grpo -v
```

### Checklist
- [ ] `rl/ppo.py`: `algo`/`group_size`; rẽ nhánh GRPO (batchify K, lex rank, adv=rank, bỏ value term).
- [ ] `--algo ppo` default ⇒ path cũ byte-identical; critic giữ trong optimizer, không vào loss GRPO.
- [ ] `tests/test_grpo.py`: ⭐ rank tay, ⭐ layout, ⭐ adv finite/zero-mean, ⭐ backward+grad, ⭐ PPO cũ, ⭐ smoke {P,U}.
- [ ] `unittest discover` xanh — 90 cũ + mới.
- [ ] Commit.

### Commit message
```
D2 Phase 3: GRPO core (K-sample group, centered lexicographic rank advantage)

shared_step gains --algo grpo: batchify each instance to K, lexsort the K by
(T_1,T_2,T_3), advantage = centered rank (group mean baseline -> no critic,
vf_lambda=0). Keeps clipped surrogate+ratio+entropy. Old PPO path unchanged.
```
