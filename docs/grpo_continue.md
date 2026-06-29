# GRPO / D2 — Handoff để tiếp tục ở session khác

> Cập nhật: 2026-06-29 (session 4). FD exhaustion bug trong `curriculum_medium` đã fix xong.
> Toàn bộ curriculum pipeline sẵn sàng chạy. Xem `docs/continue_ils.md` cho chi tiết ILS.

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
- **`curriculum_small` (20:40, 200 epoch) đã chạy ổn.** `curriculum_medium` (20:40,30:60,40:80) bị
  crash EMFILE ~epoch 55 — đã fix xong trong session 4.

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

## 4. Speedups & fixes đã implement

### Session 2 (commit `7cc8cde`)

| Fix | File | Chi tiết |
|---|---|---|
| **Fix 1** float32_matmul_precision | `scripts/train.py` | `torch.set_float32_matmul_precision('medium')` — Tensor Cores RTX 4090, ~20-40% GPU speedup |
| **Fix 2** data reload bug | `src/trainers/base.py` | Chỉ xóa cache nếu `path.startswith("data/")` — tránh xóa external data như `outputs/bm_*/data/` |
| **Fix 3** DataLoader workers | `src/env/env.py` | `effective_workers = min(num_workers, max(0, data_size // 2000))`, `pin_memory=True`, `persistent_workers=True` |
| **Fix 4** vectorize advantage | `src/trainers/grpo.py` | `_centered_lex_rank_batch` (B,K,P) numpy thay vì loop over B |

### Session 3 (commits `91b51df`, `e62b26c`, `f1334ec`, `91016f8`, `ee61e82`)

| Fix | File | Chi tiết |
|---|---|---|
| **Perf** Scheduler hot-path | `src/solvers/scheduler.py` | numpy-ify `_trip_profile` — 2.8× speedup |
| **Perf** encoder sharing GRPO | `src/trainers/grpo.py` | Share encoder output across K rollouts + fix `store_all_logp` |
| **Feat** inter/intra LS | `src/utils/local_search.py` | Swap-based intra-route + inter-route operators, numpy broadcasting — 15× so với Python loops |
| **Fix** capacity bug `get_once` | `src/solvers/meta.py` | `chosen = idxs[idx]` thay `routes[idx]` — sai vehicle khi có vehicle infeasible |
| **Feat** `get_Ts` + `run_parallel2` | `src/solvers/cal_reward.py`, `src/utils/ops.py` | Batch T1/T2/T3 qua Scheduler cho ILS; sequential map helper |

### Session 4 (commit `748be1d` + working tree, 2026-06-29)

`curriculum_medium` crash `OSError: [Errno 24] Too many open files` ~epoch 55. Ba nguồn FD:

| Nguồn | Fix |
|---|---|
| PyTorch `file_descriptor` sharing: 1 FD/tensor qua IPC | `file_system` strategy scoped trong `generate_dataset` (save/restore) |
| Worker pipe FDs từ DataLoader không được giải phóng ngay | `del it, dataloader; gc.collect()` trước `return` trong `generate_dataset` |
| **Root cause**: `MultiSizeCARPGenerator` gọi `generate_dataset` với `num_workers=24` × 3 buckets trong khi Lightning's 29 persistent workers đang sống | `num_workers=0` hardcode khi gọi từ `MultiSizeCARPGenerator` (generation là pure CPU math, không cần parallel workers) |

**Critical Lightning 2.x API fix** (phát hiện bởi Opus review):  
`Trainer.reset_train_dataloader()` không tồn tại trong Lightning 2.x (bị xóa từ 2.0). Thay bằng:
```python
self.trainer.fit_loop._combined_loader = None
```
Lightning's `setup_data()` rebuild từ `train_dataloader()` hook khi `_combined_loader is None` ở đầu epoch tiếp theo. Không có race condition vì `on_train_epoch_end` fire sau khi epoch đã hoàn toàn xong.

**Đồng thời**: xóa `reload_dataloaders_every_n_epochs` khỏi `Trainer` (trước đó dùng để trigger reload, nhưng nó reload cả val/test mỗi 4 epoch — sai). `BaseRL` tự quản lý reload train-only qua `_make_train_dataloader()`.

**Dead code removed** (Opus review):
- `shuffle_train_dataloader` param (`BaseRL` + `GRPO`) — stored nhưng không dùng
- `MultiSizeCARPGenerator.num_workers` param — bị hardcode `0` ngay bên trong
- `self.log_on_step = metrics.get("log_on_step", True)` trong `instantiate_metrics` — bị overwrite ngay sau bởi constructor arg
- `print(">>>>>>>>>>>>>>>>>>>>")` trong `setup()`

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
thắng numpy vì numpy có fixed overhead lớn hơn 10-20 Python iterations.
**Lesson:** numpy chỉ thắng khi array ≥ ~50 elements/call. (revert tại commit `b678b97`)

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

---

## 7. ILS baseline (session 3)

Scripts sẵn sàng, giải và validate được trên bất kỳ `.npz` instance nào:

```bash
uv run python scripts/ils_log.py \
    --file data/ood/osm_cityB/40/34_13_632.npz \
    --variant P --vehicles 3 --num_sample 20 \
    --log outputs/my_solution.txt

uv run python scripts/validate_solution.py \
    --instance data/ood/osm_cityB/40/34_13_632.npz \
    --solution outputs/my_solution.sol
```

**Lưu ý:** tất cả 800 instances đều cần **≥ 3 xe** (công thức `Q = Σq/3 + 0.5` hardcode `/3`).
Xem chi tiết: **`docs/continue_ils.md`**.

---

## 8. VIỆC CẦN LÀM TIẾP (theo ưu tiên)

### 8.1 — Chạy curriculum pipeline

FD bug đã fix. Commit working tree trước, rồi chạy theo thứ tự:

```bash
# Phase 1 (nếu chưa có checkpoint)
MODE=curriculum_small bash scripts/train.sh

# Phase 2 — warm-start từ phase 1
RESUME_FROM=outputs/checkpoints/curriculum_small/last.ckpt \
MODE=curriculum_medium bash scripts/train.sh

# Phase 3 — warm-start từ phase 2
RESUME_FROM=outputs/checkpoints/curriculum_medium/last.ckpt \
MODE=curriculum_large bash scripts/train.sh
```

- ⚠️ **Dọn cache trước mỗi phase:** `rm -f data/*.data`
- Theo dõi: `tail -f logs/train_curriculum_<mode>_<ts>.out`
- Monitor: `outputs/lightning_logs/version_*/metrics.csv` cột `val/T1_best,T2_best,T3_best`

### 8.2 — So sánh GRPO vs ILS trên `data/ood/`

ILS baseline đã sẵn sàng. Bước tiếp là A/B GRPO vs ILS:

```bash
# GRPO eval
uv run python -m eval.run_grid --ckpt <grpo_ckpt> --path data/ood --M 3,5,7,10 --variant P --num_sample 100 --out outputs/grpo_P.csv

# ILS eval (cần script batch — chưa có, phải viết)
# paired_win_rate(grpo_rows, ils_rows) trong eval/stats.py
```

### 8.3 — Cân nhắc tăng K hoặc ppo_epochs

Hiện `group_size=8` (train.sh: 16), `ppo_epochs=1`. Với memory còn ~18 GB free, có thể thử K=16/32
hoặc ppo_epochs=2.

---

## 9. Vận hành / cạm bẫy (QUAN TRỌNG cho session sau)

- **GPU SHARE:** kiểm `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` trước khi
  chạy. User `iec` thỉnh thoảng chạy `wgp.py --i2v-14B` chiếm ~20 GB. Đợi hoặc hỏi trước.
- **PYTORCH_CUDA_ALLOC_CONF:** `train.sh` đã set `expandable_segments:True` qua `nohup env ...`.
- **Cache trap:** generator LOAD `data/*.data` nếu tồn tại. Luôn `rm -f data/*.data` trước run mới
  với size/M khác. `data/eval_bm_tmp.data` là file tạm của `eval_bm.py` — có thể xóa.
- **FD exhaustion fixed:** `MultiSizeCARPGenerator` dùng `num_workers=0` khi generate (pure CPU math).
  `generate_dataset` với `num_workers>0` vẫn đúng khi gọi standalone (e.g. `save_cache`, `gen_data.py`).
- **Lightning 2.x reload:** không dùng `reload_dataloaders_every_n_epochs`. `BaseRL.on_train_epoch_end`
  tự set `trainer.fit_loop._combined_loader = None` để trigger reload train-only.
- **Load checkpoint:** `torch.load(ckpt, weights_only=False)`.
- **Import qua package cài editable:** nếu `ModuleNotFoundError` → chạy `uv sync`.
- **Test suite:** `uv run python -m unittest discover -s tests -p "test_*.py"` (146 tests, ~10s).
- **Job sống sót session-restart:** `setsid bash -c '...' < /dev/null > log 2>&1 &`.

---

## 10. Trạng thái git

- Branch `dev`. HEAD: `4d07cca` (chưa commit session 4 fixes — working tree có thay đổi).
- Session 4 changes (working tree, chưa commit):
  - `src/trainers/base.py` — `_make_train_dataloader()` split, `fit_loop._combined_loader = None`, dead code xóa
  - `src/trainers/grpo.py` — xóa `shuffle_train_dataloader` param
  - `src/env/generator.py` — xóa `num_workers` param từ `MultiSizeCARPGenerator`
  - `src/env/env.py` — xóa kwarg `num_workers` tại call site
  - `scripts/train.py` — xóa `reload_dataloaders_every_n_epochs` khỏi Trainer
- Session 4 committed:
  - `4d07cca` — feat: add MILP solution verifier for HDCARP-P and HDCARP-U
  - `748be1d` — fix: LP inter-level constraints, FD leak in generator, and dataloader reload wiring
- Session 3: `ee61e82`, `91016f8`, `f1334ec`, `91b51df`, `e62b26c`
- Session 2: `56e105d`, `b678b97`, `3d7b4e7`, `7cc8cde`, `941d8fd`
- Lịch sử GRPO: subclass PPO → REINFORCE standalone → mini-batch standalone (`7cc8cde`) → GRPO_old cleanup.
