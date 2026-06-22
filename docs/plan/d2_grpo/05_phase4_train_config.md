# Phase 4 — **Train config** (`--group_size`, `--algo`, MODE) + smoke train

> Mục tiêu: wire GRPO ra CLI/`train.sh`, mở rộng MODE, và chạy **smoke train ngắn** chứng minh đường cong
> **per-objective** (T_1,T_2,T_3 log riêng) — T_2/T_3 **giảm** trong khi T_1 đứng ở sàn. Phụ thuộc Phase 3.
> Code: `train.py` (chọn lớp), `train.sh`, `rl/grpo.py` (metrics log). **`rl/ppo.py` không đổi.**

## 4.1 — `train.py`: thêm args + wire

**File:** `train.py` (sau `parse_args`, `train.py:11-42`)
- Thêm:
  ```python
  parser.add_argument('--algo', type=str, default='ppo', choices=['ppo','grpo'])
  parser.add_argument('--group_size', type=int, default=8, help='K samples/instance for GRPO')
  parser.add_argument('--reward_mode', type=str, default=None,
                      help='scalar|vector; auto=vector if algo==grpo else scalar')
  ```
- Wire (quanh `train.py:62-83`):
  ```python
  from rl.grpo import GRPO
  reward_mode = args.reward_mode or ('vector' if args.algo == 'grpo' else 'scalar')
  env = CARPEnv(..., variant=args.variant, sizes=sizes, reward_mode=reward_mode)
  Model = GRPO if args.algo == 'grpo' else PPO       # chọn LỚP, không cờ trong ppo.py
  extra = {'group_size': args.group_size} if args.algo == 'grpo' else {}
  model = Model(env, policy, ..., **extra)
  ```
- ⚠️ **`val/reward` monitor** (`train.py:90`): GRPO `td["reward"]` là (B,3); metric log phải scalarize cho
  monitor (dùng `-T_1` hoặc `-(T.w)` chỉ để **log/checkpoint chọn**, KHÔNG cho gradient). Ghi rõ ở 4.3.

## 4.2 — `train.sh`: MODE toggle + biến

**File:** `train.sh` (`train.sh:7-49`)
- Thêm biến shared: `ALGO="${ALGO:-grpo}"`, `GROUP_SIZE="${GROUP_SIZE:-8}"`.
- Truyền `--algo "$ALGO" --group_size "$GROUP_SIZE"` vào lệnh `train.py` (`train.sh:59-77`).
- ⚠️ **Batch hiệu dụng = BATCH_SIZE × K**; smoke (`validate`) hiện `BATCH_SIZE=512` ⇒ với K=8 là 4096
  rollouts/step. Cân nhắc hạ `BATCH_SIZE` ở validate (vd 128) để epoch nhanh; **đảm bảo `BATCH_SIZE×K ≥ 10`**
  (luôn thoả). Giữ `nohup → logs/train_*.out` (CLAUDE.md: **không** redirect ad-hoc).
- Giữ `--algo ppo` chạy được (set `ALGO=ppo ./train.sh`) cho A/B.

## 4.3 — Log per-objective (sanity đường cong)

**File:** `rl/grpo.py` (trong `GRPO.shared_step`; tái dùng `log_metrics` của PPO `rl/ppo.py:88-113`)
- Sau khi có (B,K,3) T, log thêm **`T1_mean,T2_mean,T3_mean`** (trung bình batch) vào `out` + `metrics["train"]`.
- Mục tiêu sanity: **T_1 phẳng ở sàn, T_2/T_3 GIẢM** (dưới weighted reward chúng phẳng — đây là bằng chứng
  GRPO hoạt động). Đây là **đo**, không assert cứng trong unittest (chạy thật ở Phase 6).

## 4.4 — Smoke train (ngoài unittest)
```bash
ALGO=grpo GROUP_SIZE=8 MODE=validate ./train.sh
tail -f logs/train_validate_*.out      # xem train/T1_mean,T2_mean,T3_mean
```
Kỳ vọng: loss finite, không NaN; `T2_mean/T3_mean` **xu hướng giảm** vài epoch; `T1_mean` ổn định.

---

## ✅ Cổng test Phase 4

**File test mới:** `tests/test_train_config.py` (parse args + wire, KHÔNG chạy full train).

1. **⭐ Parse `--algo`/`--group_size`:** `parse_args(["--algo","grpo","--group_size","16"])` → `algo='grpo'`,
   `group_size=16`; default → `algo='ppo'`, `group_size=8`.
2. **⭐ Auto reward_mode:** `algo=='grpo'` ⇒ `reward_mode=='vector'`; `algo=='ppo'` ⇒ `'scalar'`; override
   `--reward_mode` được tôn trọng.
3. **⭐ PPO build với algo=grpo (shape):** dựng `CARPEnv(reward_mode='vector')` + `PPO(algo='grpo',
   group_size=8)`; chạy **1 `shared_step` train** trên batch nhỏ (B=2,K=8 ⇒ 16≥10) → loss finite (tích hợp
   Phase 3 qua đường CLI).
4. **⭐ Metrics có per-objective key:** sau `shared_step` GRPO, `out` chứa `T1_mean/T2_mean/T3_mean` finite.
5. **⭐ algo=ppo build vẫn chạy:** default path → `shared_step` train loss finite (A/B).
6. **Rollout smoke** ở cả `--algo {ppo,grpo}` × variant `{P,U}`, reward finite.
7. **Suite cũ xanh** (default ppo/scalar).

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_train_config -v
# smoke train (ngoài unittest):
ALGO=grpo MODE=validate ./train.sh && tail -n 50 logs/train_validate_*.out
```

### Checklist
- [ ] `train.py`: `--algo`,`--group_size`,`--reward_mode`(auto); **chọn lớp `GRPO`/`PPO`**; wire env.
- [ ] `train.sh`: `ALGO`/`GROUP_SIZE` biến + MODE; giữ `nohup → logs/`; A/B `ALGO=ppo`.
- [ ] `rl/grpo.py`: log `T1_mean/T2_mean/T3_mean`; metric monitor scalarize chỉ để chọn ckpt. **`rl/ppo.py` không đổi.**
- [ ] `tests/test_train_config.py`: ⭐ parse, ⭐ auto-mode, ⭐ build+step grpo, ⭐ per-obj metrics, ⭐ ppo build, smoke {P,U}.
- [ ] Smoke train: T_2/T_3 giảm, T_1 phẳng, không NaN.
- [ ] `unittest discover` xanh — 90 cũ + mới.
- [ ] Commit.

### Commit message
```
D2 Phase 4: train config for GRPO (--algo, --group_size) + per-objective logs

train.py/train.sh expose --algo {ppo,grpo} and --group_size K; reward_mode
auto-selects vector for grpo. Logs T1/T2/T3 means so smoke train shows T2/T3
descend while T1 holds. PPO path A/B-preserved via ALGO=ppo.
```
