# Phase 2 — **Vector reward** (env lộ T-vector ra train path, sau cờ)

> Mục tiêu: cho path train **thấy được T-vector (T_1,T_2,T_3) per-rollout** thay vì chỉ scalar
> `-(rs*w).sum`. Giữ `get_objective` nguyên; giữ scalar cũ **sau cờ** cho A/B. Code: `env/env.py`. Phụ thuộc
> Phase 0 (chốt `reward_mode`). **KHÔNG đụng** Scheduler/`calc_reward` (đã trả full vector).

## 2.1 — Hiện trạng (đã xác nhận)

- `common/cal_reward.py:33-39`: `calc_reward` trả **full T-vector** `torch.tensor(rs)` `[T_1,T_2,T_3]`. ✓ sẵn.
- `env/env.py:150-159 get_reward`: scalarize `rs = -(rs*w).sum(-1, keepdim=True)`, `w=obj_weights`
  (`env/env.py:37`). → đây là chỗ T-vector **bị nén** trước khi tới `policy.forward` (`policy/policy.py:103`).
- `env/env.py:144-148 get_objective`: đã trả full vector (eval) — **GIỮ NGUYÊN**.

## 2.2 — Thay đổi cụ thể

**File:** `env/env.py`

1. `__init__` thêm cờ `reward_mode='scalar'` (mặc định = hành vi cũ). Lưu `self.reward_mode`.

2. `get_reward(td, actions)` tách 2 nhánh, **dùng chung** `run_parallel(...)` (giữ `local_search=False,
   return_torch=True`, `env/env.py:153-154`):
   ```python
   rs = run_parallel(calc_reward, actions, td, num_workers=24, num_epochs=10,
                     local_search=False, return_torch=True, variant=self.variant)  # (B,3) T-vector
   if self.reward_mode == 'vector':
       return rs.to(td.device)                       # (B,3): GRPO path xếp rank ngoài
   w = torch.tensor(self.obj_weights, dtype=rs.dtype, device=rs.device)
   return -(rs * w).sum(-1, keepdim=True).to(td.device)   # (B,1): path cũ, byte-identical
   ```
   ⚠️ **Dấu**: vector mode trả **T dương** (nhỏ = tốt); việc đổi dấu/center để cho PPO làm ở **Phase 3
   GRPO** (rank). Scalar mode giữ `-(...)` y như cũ.

3. `policy/policy.py:103` `td.set("reward", env.get_reward(td, actions))`: với `reward_mode='vector'`,
   `td["reward"]` thành `(B,3)`. ⚠️ Path PPO cũ (`rl/ppo.py:181,190`) đọc `out["reward"]`/`sub_td["reward"]`
   **giả định scalar** → **chỉ bật `reward_mode='vector'` cùng `--algo grpo`** (Phase 3 xử lý shape (B,3)).
   Phase 2 chỉ **thêm khả năng**; default scalar ⇒ PPO cũ không vỡ.

> **A/B/rollback:** `reward_mode='scalar'` (default) ⇒ `get_reward` trả **đúng (B,1)** như trước → path cũ
> không đổi 1 byte. `reward_mode='vector'` là opt-in, ráp với GRPO ở Phase 3.

---

## ✅ Cổng test Phase 2

**File test mới:** `tests/test_vector_reward.py`.

1. **⭐ Shape theo mode:** env `reward_mode='scalar'` → `get_reward` trả `(B,1)`; `reward_mode='vector'`
   → `(B,3)`. (Dựng td nhỏ qua `env.reset`; `B×1 ≥ 10`? — `run_parallel num_epochs=10` ⇒ **dùng B≥10**
   hoặc patch `num_epochs`; test ghi B=12 cho an toàn, xem [`01_phase0_scope §0.4`](01_phase0_scope.md).)
2. **⭐ Tương đương scalar:** với cùng actions, `scalar_reward == -(vector_reward * obj_weights).sum(-1)`
   (≤1e-5) → khẳng định **không đổi nghĩa**, chỉ **lộ thêm**.
3. **⭐ Vector finite & dương:** mọi phần tử `(B,3)` finite, ≥0 (completion time).
4. **Default = scalar:** env không truyền `reward_mode` → `get_reward` trả `(B,1)` (rollback an toàn).
5. **⭐ Rollout smoke (scalar path nguyên):** `env.reset → policy(calc_reward=True) → reward (B,1) →
   backward` finite — **PPO cũ không vỡ** (variant `P`, lặp `U`).
6. **Suite cũ xanh:** 90 test không đổi (vì default scalar).

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_vector_reward -v
```

### Checklist
- [ ] `env/env.py`: `reward_mode` (default `'scalar'`); `get_reward` 2 nhánh; `get_objective` GIỮ NGUYÊN.
- [ ] `tests/test_vector_reward.py`: ⭐ shape, ⭐ tương đương scalar, ⭐ finite, default scalar, rollout smoke.
- [ ] Xác nhận `calc_reward`/Scheduler **không đụng**.
- [ ] `unittest discover` xanh — 90 cũ + mới.
- [ ] Commit.

### Commit message
```
D2 Phase 2: vector reward mode (env exposes T-vector, flagged)

get_reward gains reward_mode={scalar,vector}; vector returns (B,3) T-vector
for the GRPO path, scalar keeps -(T.w) byte-identical (default). A/B safe.
```
