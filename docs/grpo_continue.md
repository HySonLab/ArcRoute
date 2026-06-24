# GRPO / D2 — Handoff để tiếp tục ở session khác

> Cập nhật: 2026-06-24 (session 2). GRPO_old đã cleanup hoàn toàn, 4 training speedups đã
> implement, 146/146 tests pass. **Sẵn sàng chạy full-scale.**

---

## 1. TL;DR — trạng thái hiện tại

- **Kiến trúc bài toán (không đổi):** policy **M-agnostic** (chỉ sinh thứ tự arc) → **Scheduler `Φ`**
  (`src/solvers/scheduler.py`, map order+M → routes + (T₁,T₂,T₃)) → reward. Mục tiêu **lexicographic**
  (T₁ ≺ T₂ ≺ T₃). M chỉ vào Scheduler ("train once, M là tham số eval-time").
- **GRPO hiện tại: GRPO_old (Option A) — centered lex-rank advantage:**
  - `centered_lex_rank(T_bk3)` → advantage ∈ [-1,+1], mean=0. Best→+1, worst→-1, ties=average rank.
  - Rollout toàn bộ B×K trong `torch.no_grad()`, stash `(reward, logprobs, advantage, action)` vào `td`.
  - Inner loop mini-batch (chunk `mini_batch_size=256`) re-run policy với PPO-style clipped surrogate.
  - DAPO dynamic filtering: bỏ nhóm all-tied (zero advantage) → không update gradient.
  - **Không dùng critic**, không value loss. Kế thừa `BaseRL` trực tiếp.
  - `_centered_lex_rank_batch` (batched B×K×P numpy) dùng trong `_group_advantage` để vectorize.
- **A/B benchmark (40 epoch, validate params) đã chạy:**
  GRPO thắng PPO trên 3 objectives: T1 -9.2%, T2 -4.4%, T3 -2.9%.
- **Chất lượng ĐÃ XÁC NHẬN** ở validate scale (40 epoch). Full-scale train chưa chạy.

---

## 2. Kiến trúc code (sau refactor) — installable `src/` package

```
src/
├── env/        env.py (CARPEnv, reward_num_workers), generator.py
├── policy/     encoder/decoder/policy/context/init/decode_strategy   (mạng neural, M-agnostic)
├── trainers/   base.py (BaseRL)  ppo.py (PPO+critic)  grpo.py (GRPO mini-batch)
├── solvers/    scheduler.py, cal_reward.py + aco/ea/ils/lp/meta/rl_hyb
├── eval/       run_grid.py, stats.py
└── utils/      ops.py (batchify/unbatchify/run_parallel/import_instance), consts, local_search
scripts/        train.py, train.sh, bm_trainer.sh, eval_bm.py, gen_data.py
tests/          test_grpo.py, test_scheduler.py (+ 9 others), 146 tests total
outputs/        checkpoints + lightning_logs (gitignored)
data/           artifact (gitignored: cache, ood, osm, *.data)
```

- **Cài đặt:** `uv sync` build editable → import `from trainers.grpo import GRPO`, etc.
- **`BaseRL`** (`trainers/base.py`): scaffolding Lightning chung. PPO và GRPO là anh em kế thừa `BaseRL`.
- Critic nằm trong `trainers/ppo.py` (`CriticNetwork` + `create_critic_from_actor`).

---

## 3. GRPO — chi tiết (`src/trainers/grpo.py`)

### 3.1 Cấu trúc _train_step

```python
# (1) No-grad rollout — không giữ activation graph
with torch.no_grad():
    td = batchify(td0, K)
    out = policy(td.clone(), env, phase="train")

# (2) Tính rank advantage từ T-vector (batch vectorized)
T = unbatchify(reward, K)              # (B, K, P)
advantage = _group_advantage(T_bk3)   # calls _centered_lex_rank_batch → (B*K,)

# Stash lên td (reward echo cho inner loop calc_reward=False)
td.set("reward", reward)
td.set("logprobs", out["log_likelihood"])
td.set("advantage", advantage)
td.set("action", out["actions"])

# (3) Mini-batch inner loop với gradient
for mini_batch in shuffled_chunks(td, mini_batch_size):
    out_i = policy(sub_td, actions=..., calc_reward=False, return_entropy=True)
    ratio = exp(ll_new - ll_old)
    loss = -min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv).mean() - entropy_lambda*entropy
    backward(loss); clip_grad; step()
```

### 3.2 Params GRPO

| Param | Default | Ghi chú |
|---|---|---|
| `group_size` | 8 | K samples/instance |
| `clip_range` | 0.2 | PPO epsilon |
| `ppo_epochs` | 1 | inner passes/update |
| `mini_batch_size` | 256 | chunk size inner loop |
| `entropy_lambda` | 0.0 | entropy bonus |

### 3.3 `centered_lex_rank` + `_centered_lex_rank_batch`

`centered_lex_rank(T)` — dùng trong tests, single-instance (K,P) → (K,) advantage ∈ [-1,+1].  
`_centered_lex_rank_batch(T_np)` — module-level, batched (B,K,P) numpy. Successive stable argsorts,
vectorized tie detection `np.all(Ts[:,1:,:] == Ts[:,:-1,:], axis=2)`, per-tied-group averaging.
Normalize ÷ (K-1) để đưa về [-1,+1]. Dùng trong `_group_advantage`.

DAPO: nhóm all-tied → advantage=0 → bị lọc ra khỏi gradient update.

### 3.4 Checkpoint monitor

`val/lex_best` (Horner scalar, `_LEX_C=1e3`). Val cũng log `val/T1_best`, `T2_best`, `T3_best`.

---

## 4. Training speedups đã implement (commit `7cc8cde`)

| Fix | File | Chi tiết |
|---|---|---|
| **Fix 1** float32_matmul_precision | `scripts/train.py` | `torch.set_float32_matmul_precision('medium')` — Tensor Cores RTX 4090, ~20-40% GPU speedup |
| **Fix 2** data reload bug | `src/trainers/base.py` | Chỉ xóa cache nếu `path.startswith("data/")` — tránh xóa external data như `outputs/bm_*/data/` |
| **Fix 3** DataLoader workers | `src/env/env.py` | `effective_workers = min(num_workers, max(0, data_size // 2000))`, `pin_memory=True`, `persistent_workers=True` |
| **Fix 4** vectorize advantage | `src/trainers/grpo.py` | `_centered_lex_rank_batch` (B,K,P) numpy thay vì loop over B |

---

## 5. Lịch sử thay đổi GRPO

### Tại sao đổi từ REINFORCE sang mini-batch?

Bản REINFORCE (commit `941d8fd`) dùng "encode-once, share K" — encoder forward có grad, batchify output
cho B×K=1024 sequences decode. Với size ladder `40:120` (120 required arcs), activation graph cho
1024×120 steps chiếm ~22-23 GB VRAM trên RTX 4090 → OOM.

Bản mini-batch fix bằng `no_grad` rollout: memory peak từ 23.4 GB → **5.7 GB** (giảm 4×).

### Tại sao revert về GRPO_old (Option A)?

BC warmup + asymmetric clip (Option F) thử nghiệm không có lợi rõ rệt → cleanup. GRPO_old
(centered lex-rank, symmetric clip) đơn giản hơn và đã A/B xác nhận.

### Scheduler vectorization thử rồi revert

Numpy ops (`_balanced_chunks` cumsum, `_completion_times` mask) thử vectorize → **chậm hơn 5-20%**
ở mọi training size. Lý do: per-class block chỉ 10-30 arcs, per-trip chỉ 10-20 arcs — Python loop
thắng numpy vì numpy có fixed overhead (ravel, astype, mask creation) lớn hơn 10-20 Python iterations.
**Lesson:** numpy chỉ thắng khi array ≥ ~50 elements/call. (revert tại commit `b678b97`)

### Bug fixes

1. `KeyError: 'reward'` — `policy.forward(calc_reward=False)` vẫn echo `td["reward"]`; fix: stash trước inner loop.
2. `on_train_epoch_end` xóa external data — fix: chỉ xóa nếu `path.startswith("data/")`.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — thêm vào `bm_trainer.sh` để giảm fragmentation.

---

## 6. A/B Benchmark (đã chạy xong)

### Params

Cùng tham số cho cả hai: `batch_size=128`, `max_epoch=40`, `train_data=10000`, `val_data=1000`,
`embed_dim=128`, `num_encoder_layers=6`, `num_heads=8`, `sizes=20:40,30:60,40:80,50:100,40:120`,
`fleet=2,3,5,7,10`, `seed=6868`.

### Kết quả (512 test instances, 40×80, M=3)

|  | T1 (↓) | T2 (↓) | T3 (↓) |
|---|---|---|---|
| **GRPO** | **21.15** | **43.05** | **65.32** |
| PPO | 23.29 | 45.03 | 67.31 |
| **Delta** | **-9.2%** | **-4.4%** | **-2.9%** |

GRPO thắng PPO trên cả 3 objectives. PPO early-stop ở epoch 5 (val/reward không cải thiện),
GRPO tiếp tục đến epoch 35 → GRPO học ổn định hơn trên multi-objective lexicographic.

### Artifacts giữ lại

```
outputs/bm_20260624_010623/
  ckpt/grpo/epoch=035-*.ckpt    # GRPO validate checkpoint
  ckpt/ppo/epoch=005-*.ckpt     # PPO validate checkpoint (early-stop)
  data/train.data, val.data, test.data
logs/bm_runner_20260624_010623.out
```

### Scripts

```bash
bash scripts/bm_trainer.sh          # chạy GRPO→PPO→eval, log ở logs/bm_runner_<ts>.out
uv run python scripts/eval_bm.py \  # eval thủ công nếu cần
    --grpo_ckpt <ckpt> --ppo_ckpt <ckpt> \
    --num_loc 40 --num_arc 80 --fleet 3 \
    --embed_dim 128 --num_encoder_layers 6 --num_heads 8
```

---

## 7. VIỆC CẦN LÀM TIẾP (theo ưu tiên)

### 7.1 — Train full-scale GRPO

A/B validate đã PASS. Tiếp theo:

```bash
MODE=full ALGO=grpo bash scripts/train.sh
```

- ⚠️ **Dọn cache trước:** `rm -f data/*.data`
- Theo dõi `outputs/lightning_logs/version_*/metrics.csv` cột `val/T1_best,T2_best,T3_best`
- Full-scale: `max_epoch=1000`, `batch_size=4096`, `train_data=100000`, `num_encoder_layers=12`

### 7.2 — Eval trên `data/ood/`

```bash
uv run python -m eval.run_grid --ckpt <grpo_ckpt> --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 --out outputs/grpo_P.csv
uv run python -m eval.run_grid --ckpt <ppo_ckpt>  --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 --out outputs/ppo_P.csv
# paired_win_rate(grpo_rows, ppo_rows) trong eval/stats.py → win-rate lex + T₁-regress
```

### 7.3 — Cân nhắc tăng K hoặc ppo_epochs

Hiện `group_size=8`, `ppo_epochs=1`. Với memory còn ~18 GB free (không có GPU contention), có thể thử
K=16 hoặc ppo_epochs=2.

### 7.4 — Dọn doc

`README.md` còn tham chiếu đường dẫn cũ. `scripts/train.sh` MODE=full 1000 epoch — chỉnh nếu cần.

---

## 8. Vận hành / cạm bẫy (QUAN TRỌNG cho session sau)

- **GPU SHARE:** kiểm `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` trước khi
  chạy. User `iec` thỉnh thoảng chạy `wgp.py --i2v-14B` chiếm ~20 GB. Đợi hoặc hỏi trước.
- **PYTORCH_CUDA_ALLOC_CONF:** `bm_trainer.sh` đã set `expandable_segments:True`. Nếu train thủ công
  qua `train.sh` bị OOM fragmentation, export trước: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Cache trap:** generator LOAD `data/*.data` nếu tồn tại. Luôn `rm -f data/*.data` trước run mới
  với size/M khác. `data/eval_bm_tmp.data` là file tạm của `eval_bm.py` — có thể xóa.
- **Load checkpoint:** `torch.load(ckpt, weights_only=False)`.
- **Import qua package cài editable:** nếu `ModuleNotFoundError` → chạy `uv sync`.
- **Test suite:** `uv run python -m unittest discover -s tests -p "test_*.py"` (146 tests, ~10s).
- **Job sống sót session-restart:** `setsid bash -c '...' < /dev/null > log 2>&1 &`.
- **bm_trainer.sh** dùng nohup nội bộ — follow: `tail -f logs/bm_runner_<ts>.out`.

---

## 9. Trạng thái git

- Branch `dev`. Working tree sạch sau commit `b678b97` ("revert: restore original Scheduler").
- Commits trong 2 session này:
  - `b678b97` — revert Scheduler vectorization (backfired)
  - `3d7b4e7` — thêm `tests/test_scheduler.py` (11 pinning tests)
  - `7cc8cde` — 4 training speedups + GRPO_old cleanup
  - `10458be` — DAPO dynamic filtering (deprecated, đã cleanup)
  - `941d8fd` — refactor src layout + standalone REINFORCE-GRPO
- Lịch sử GRPO: subclass PPO → REINFORCE standalone (`941d8fd`) → mini-batch standalone (`7cc8cde`) → GRPO_old cleanup.
