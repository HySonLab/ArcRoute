# Phase 3 — **GRPO core** trong **`rl/grpo.py` RIÊNG** (K-group + centered lexicographic rank, bỏ critic)

> Mục tiêu: tạo **file mới `rl/grpo.py`** với `class GRPO(PPO)` **kế thừa** và override đúng phần khác:
> roll out **K sample/instance** (nhóm), reward = **centered lexicographic rank** trong nhóm, **advantage =
> rank** (group mean = baseline → **bỏ critic**, không value loss). **`rl/ppo.py` KHÔNG sửa 1 byte** ⇒ path
> PPO cũ an toàn tuyệt đối (A/B + rollback). Phụ thuộc Phase 2 (vector reward). Code: **`rl/grpo.py` (mới)**,
> `train.py` (chọn lớp).

> **Vì sao file riêng (không nhồi cờ vào `ppo.py`):** GRPO là thuật toán KHÁC (không critic, advantage =
> rank nhóm). Tách `GRPO(PPO)` → `ppo.py` bất biến, không rẽ nhánh rối, dễ test/mở rộng, A/B = chọn lớp.

## 3.1 — Hiện trạng PPO (đã xác nhận, `rl/ppo.py` — sẽ TÁI DÙNG, không sửa)

- `shared_step` (`:157`): rollout no-grad → set `logprobs/reward/action` → vòng mini-batch PPO (`:185-247`).
- Advantage **bandit**: `adv = previous_reward - value_pred.detach()` (`:207-208`); critic 1 scalar (`rl/critic.py:18`).
- Loss = surrogate + `vf_lambda*value_loss - entropy_lambda*entropy` (`:229-233`).
- `configure_optimizers` (`:148-155`) gộp `policy + critic`.
- Phần **dùng chung** (dataloaders, `setup`, `log_metrics`, mini-batch loop, surrogate clip) → GRPO **kế thừa**.

## 3.2 — `rl/grpo.py`: `class GRPO(PPO)` override 3 chỗ

```python
# rl/grpo.py  (MỚI)
import torch, numpy as np
from rl.ppo import PPO
from common.ops import batchify, unbatchify

class GRPO(PPO):
    def __init__(self, *args, group_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size            # K
        # critic vẫn tồn tại (khỏi vỡ super), nhưng KHÔNG vào loss & KHÔNG vào optimizer

    def configure_optimizers(self):             # override: chỉ policy params
        return torch.optim.AdamW(self.policy.parameters(), lr=1e-4)

    def shared_step(self, batch, batch_idx, phase, dataloader_idx=None):
        ...                                     # override: K-group + lex rank (dưới)
```

### (a) Roll out nhóm K — POMO-style qua `batchify`
```python
K = self.group_size
td0 = self.env.reset(batch)                     # (B, ...)
td  = batchify(td0, K)                           # (B*K, ...) — mỗi instance lặp K lần
with torch.no_grad():
    out = self.policy(td.clone(), self.env, phase="train")   # sampling; reward_mode='vector' (Phase 2)
# out["reward"]: (B*K, 3) T-vector
```
⚠️ **Verify layout** ([`01_phase0 §0.4`](01_phase0_scope.md)): dùng **`unbatchify(x, K)`** (`common/ops.py:122`)
để về `(B, K, ...)` đúng trục — **KHÔNG reshape tay**. Test layout (gate #2) khoá đúng.

### (b) Centered lexicographic rank trong nhóm K
```python
T = unbatchify(out["reward"], K)                 # (B, K, 3)  (đúng trục, không trộn instance)
T = T.cpu().numpy()
adv = np.empty((B, K), np.float32)
for b in range(B):
    order = np.lexsort((T[b,:,2], T[b,:,1], T[b,:,0]))   # tốt->xấu, T_1 primary
    rank = np.empty(K); rank[order] = np.arange(K-1, -1, -1)
    adv[b] = (rank - (K-1)/2) / ((K-1)/2)         # [-1,1], mean=0, tốt nhất=+1
adv = batchify_like(torch.as_tensor(adv).view(B, K, 1))  # về (B*K,1) cùng trục batchify
```
> **Dấu:** T thấp ⇒ rank cao ⇒ **adv dương** ⇒ surrogate tăng xác suất. `K=1` ⇒ adv=0 (guard, yêu cầu `K≥2`).

### (c) Mini-batch update: **advantage = rank, bỏ critic**
- Tái dùng vòng mini-batch của PPO (gọi `self.policy(..., actions=...)` lấy `ll, entropy`, `ratio`,
  surrogate clip **NGUYÊN**), nhưng:
  - `adv = sub_td["advantage"]` (rank đã centered) — **không gọi `self.critic`**.
  - **không value_loss**: `loss = surrogate - entropy_lambda*entropy`.
- `normalize_adv` để **False** (rank đã chuẩn hoá, tránh chuẩn hoá kép).
- Để tái dùng tối đa: có thể tách phần "mini-batch surrogate" của `PPO.shared_step` thành 1 helper được cả
  PPO lẫn GRPO gọi (nice-to-have); tối thiểu thì GRPO copy vòng đó với 2 dòng khác (no-critic, no-vf).

## 3.3 — `train.py` chọn lớp (KHÔNG cờ trong ppo.py)
```python
from rl.ppo import PPO
from rl.grpo import GRPO
Model = GRPO if args.algo == "grpo" else PPO
model = Model(env, policy, ..., **({"group_size": args.group_size} if args.algo=="grpo" else {}))
```
- `--algo ppo` (default) ⇒ **PPO y nguyên** (Phase 4 thêm arg). GRPO yêu cầu env `reward_mode='vector'`.

---

## ✅ Cổng test Phase 3

**File test mới:** `tests/test_grpo.py`.

1. **⭐ Rank đúng (ví dụ tay):** T `(1,K,3)` = `[[5,9,9],[5,2,9],[5,2,1],[7,0,0]]` (K=4) → `[5,2,1]` **adv lớn
   nhất** (=+1), `[7,0,0]` **adv nhỏ nhất** (=−1), `adv.mean()≈0`. (Hàm rank tách riêng để test thuần.)
2. **⭐ Layout batchify/unbatchify:** `unbatchify(batchify(td0,K),K)` khớp `td0` từng instance → (B,K,3) đúng trục.
3. **⭐ Advantage finite + zero-mean/nhóm:** T ngẫu nhiên (B,K,3) → `adv.view(B,K).mean(1)≈0`, `|adv|≤1`, finite.
4. **⭐ Backward + grad:** `GRPO.shared_step` (B=2,K=8 ⇒ 16≥10) → loss finite, **policy có grad** (không NaN);
   **critic KHÔNG có grad** (ngoài loss + ngoài optimizer).
5. **⭐ `rl/ppo.py` BẤT BIẾN:** `git diff rl/ppo.py` rỗng; `PPO` train step vẫn chạy, **critic CÓ grad**.
6. **⭐ Rollout smoke GRPO:** `env(reward_mode='vector') → reset → batchify(K) → policy → rank → backward`
   finite, ở variant `P` rồi lặp `U`; `B*K≥10`.
7. **Suite cũ xanh:** 90 test.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_grpo -v
git diff --stat rl/ppo.py   # phải rỗng
```

### Checklist
- [ ] **`rl/grpo.py` MỚI**: `GRPO(PPO)` override `__init__`(group_size), `configure_optimizers`(policy-only),
      `shared_step`(K-group + lex rank + no-critic). **`rl/ppo.py` không sửa.**
- [ ] `train.py` chọn `GRPO`/`PPO` theo `--algo` (Phase 4 wire arg).
- [ ] `tests/test_grpo.py`: ⭐ rank tay, ⭐ layout, ⭐ adv finite/zero-mean, ⭐ backward+grad, ⭐ ppo.py bất biến, ⭐ smoke {P,U}.
- [ ] `unittest discover` xanh — 90 cũ + mới.
- [ ] Commit.

### Commit message
```
D2 Phase 3: GRPO in a separate rl/grpo.py (GRPO(PPO) subclass)

New rl/grpo.py: class GRPO(PPO) overrides shared_step (batchify each instance to
K, lexsort the K by (T1,T2,T3), advantage = centered rank — group mean baseline,
no critic) and configure_optimizers (policy only). rl/ppo.py is UNCHANGED, so the
old PPO/weighted path is byte-identical for A/B and rollback.
```
