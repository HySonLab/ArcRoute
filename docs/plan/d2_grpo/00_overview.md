# Plan tổng quan — D2: **vector reward + GRPO lexicographic rank**

> Mục tiêu: thay **scalar reward weighted-sum + 1 critic** bằng **tín hiệu train GRPO theo rank
> lexicographic**, để policy **thật sự học T_2/T_3 trong khối lớn các nghiệm hoà T_1**. Stack:
> **policy (GRPO, reward = rank tuple lexicographic) → Scheduler tất định (GIỮ NGUYÊN) → eval-time
> lexicographic best-of-K → within-class local search**.
>
> ⚠️ **Đây là plan về TÍN HIỆU HỌC + SELECTOR + ĐO** — KHÔNG đụng Scheduler, KHÔNG đụng global-class P mask,
> policy vẫn **M-agnostic**. Trực giao với `dynamic_plan/` (cái đó đổi *Scheduler/M*; D2 đổi *learning signal*).

## Bối cảnh: vì sao weighted-sum thất bại với T_2/T_3

`env/env.py:get_reward` hiện scalarize: `rs = -(rs*w).sum(-1)` với `w = obj_weights = [1, 1e-2, 1e-4]`
(`env/env.py:37,157-158`). Vì T_1 là ưu tiên cao **và** chiếm khối nghiệm hoà rất lớn, gradient gần như
chỉ thấy T_1; T_2/T_3 (hệ số 1e-2, 1e-4) bị nhấn chìm → **đường học T_2/T_3 phẳng**. Critic 1 scalar head
(`rl/critic.py:18-19`) chỉ fit được scalar đó. PPO ở đây là **bandit 1 bước** (`adv = previous_reward -
value_pred.detach()`, `rl/ppo.py:208`, không GAE).

→ Cần một tín hiệu **so sánh nghiệm theo đúng thứ tự lexicographic** (T_1, rồi T_2, rồi T_3) thay vì cộng
trọng số. **GRPO** làm đúng việc đó.

## Vì sao **GRPO** (group-relative, rank-based)

| Tính chất | Lý do hợp D2 |
|---|---|
| **Scale-free** | rank không phụ thuộc đơn vị/độ lớn T_k → khỏi tune `obj_weights`, khỏi lệch scale critic |
| **Critic-free** | baseline = trung bình nhóm K (group mean) → **bỏ critic**, bỏ value loss (`vf_lambda→0`) |
| **Đúng lexicographic** | xếp hạng K nghiệm bằng `np.lexsort((T_3,T_2,T_1))` → rank phản ánh **chính xác** thứ tự phân cấp |
| **Hợp multi-sample** | decoder đã sample (`store_all_logp=True`, `policy/policy.py:73`); `batchify` (`common/ops.py:76`) nhân td K lần (POMO) → có sẵn nhóm K để rank |

## Cái gì ĐỔI vs cái gì GIỮ

| Thành phần | Hiện trạng | D2 |
|---|---|---|
| **Selector best-of-K** | chọn theo **T_1 đơn** (`obj[:,0].argmin()`, `eval/run_grid.py:111`, `baseline/rl_hyb.py:37`) | **lexicographic** `np.lexsort((obj[:,2],obj[:,1],obj[:,0]))[0]` — **FREE WIN, không retrain** (Phase 1) |
| **Reward env** | scalar `-(rs*w).sum` (`env/env.py:157-158`) | **lộ T-vector** ra train path (cờ A/B); scalar cũ giữ sau cờ (Phase 2) |
| **PPO advantage** | bandit `prev_reward - value_pred` + critic (`rl/ppo.py:207-208`) | **GRPO**: K sample/instance → reward = **centered lexicographic rank** trong nhóm K → `adv = rank`, **bỏ critic** (`vf_lambda→0`); sau cờ `--algo grpo` (Phase 3) |
| **Train config** | 1 path PPO | thêm `--group_size K`, `--algo {ppo,grpo}`; log **T_1,T_2,T_3 riêng** (Phase 4) |
| **Local search** | chỉ `lsRL` intra 2-opt (`common/local_search.py:58`); `ls` inter bị comment | mạnh hoá within-class LS, chỉ chạy trên **winner best-of-K** ở eval (Phase 5, **tùy chọn**) |
| **Scheduler `Φ`** | `common/scheduler.py` | **GIỮ NGUYÊN** |
| **Global-class P mask** | `env/env.py:82-93` | **GIỮ NGUYÊN** |
| **Policy M-agnostic** | encoder/decoder không thấy M | **GIỮ NGUYÊN** |

## Các phase (critical path: `0 → 1 → 2 → 3 → 4 → 6`; Phase 5 tùy chọn)

| Phase | File | Nội dung | Bắt buộc? |
|---|---|---|---|
| **0** | [`01_phase0_scope.md`](01_phase0_scope.md) | Chốt: K, công thức rank, bỏ-critic-vs-cờ, biểu diễn vector reward, A/B + rollback | ✅ quyết định |
| **1** | [`02_phase1_lex_selector.md`](02_phase1_lex_selector.md) | **Lexicographic best-of-K** ở selector (free win, không retrain) + test | ✅ |
| **2** | [`03_phase2_vector_reward.md`](03_phase2_vector_reward.md) | `get_reward` lộ T-vector (sau cờ); test shape/giá trị + suite xanh + rollout smoke | ✅ |
| **3** | [`04_phase3_grpo_core.md`](04_phase3_grpo_core.md) | GRPO core: nhóm K + centered lex rank + `adv=rank` + bỏ critic, sau cờ `--algo grpo` | ✅ nền tảng |
| **4** | [`05_phase4_train_config.md`](05_phase4_train_config.md) | `train.py/sh` (`group_size`,`algo`) + MODE; smoke train + đường cong per-objective | ✅ |
| **5** | [`06_phase5_localsearch.md`](06_phase5_localsearch.md) | *(tùy chọn)* mạnh hoá within-class LS trên winner best-of-K | ⏹ tùy chọn |
| **6** | [`07_phase6_eval_measure.md`](07_phase6_eval_measure.md) | đường cong per-objective; win-rate lexicographic vs weighted; gap-to-LP; best-of-K yield | ✅ |

Bảng cổng test gộp: [`08_test_matrix.md`](08_test_matrix.md).

## Bất biến phải bake vào plan (xem cả `08_test_matrix.md`)

- **KHÔNG phá 90 unit test hiện có**: `uv run python -m unittest discover -s tests -p "test_*.py"` phải xanh.
- **Scheduler, global-class P mask, policy M-agnostic = GIỮ NGUYÊN.** D2 chỉ đổi *tín hiệu học + selector + đo*.
- **Không regress T_1**: ưu tiên T_1 không bao giờ bị vi phạm; win-rate vs weighted baseline phải có **zero T_1 regression**.
- Mỗi phase ship **rollout-smoke** (`env.reset → policy → reward → backward`, finite) và giữ suite xanh.
- **Giữ cờ** cho **path PPO/weighted cũ vẫn chạy** (A/B + rollback an toàn).
- Mỗi phase = **1 commit độc lập, PASS hết test mới qua phase sau**.

## Quy tắc "qua phase"

1. Mỗi phase có cổng test riêng — **PASS hết mới qua phase sau**.
2. `uv run python -m unittest discover -s tests -p "test_*.py"` phải xanh (90 cũ + test mới).
3. Mỗi phase = 1 commit độc lập.
4. Mỗi phase có **smoke test rollout** (encoder+decoder+reward+backward, finite).
