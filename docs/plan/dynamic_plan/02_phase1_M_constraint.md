# Phase 1 — Tách **`Scheduler`** thành module first-class (M chỉ vào đây)

> Mục tiêu: đưa khâu "chuỗi action → routes của M xe → mục tiêu phân cấp `(T_1,…,T_p)`" ra thành **một module
> độc lập `common/scheduler.py`** (hiện bị nhét trong `calc_reward`). Policy **M-agnostic** — không sửa
> encoder/decoder/mask. Phụ thuộc Phase 0 (Q=Σq/3 khít, Scheduler multi-trip). Code: `common/scheduler.py`
> (mới), `common/cal_reward.py` (mỏng lại). **`env/generator.py` KHÔNG đổi `Q`.**

## 1.1 — Hợp đồng (interface) của `Scheduler`

**File mới:** `common/scheduler.py`

```python
class Scheduler:
    """Toán tử Φ: (chuỗi action, M) → solution + mục tiêu phân cấp.
       M chỉ tham gia ở ĐÂY — policy hoàn toàn M-agnostic."""
    def __init__(self, variant='P'):   # multi-trip; tự thành single-trip khi K≤M
        ...
    def __call__(self, action_seq, M, clss, service_times, adj) -> (routes, T_vec):
        # routes: phân hoạch chuỗi thành các route của M xe
        # T_vec : (T_1,…,T_p) — max completion time mỗi lớp (paper: §"The problem")
        ...
```

- **Đầu vào** lấy từ `td` per-instance (qua `run_parallel` cắt `td[i]` như hiện tại). `M = td["num_vehicle"]`.
- **Đầu ra** `T_vec` đúng định nghĩa paper: `T_k` = thời điểm mọi xe phục vụ xong arc lớp `k`.
- **Tôn trọng P/U:** P enforce precedence; U dùng **hierarchy-level** (paper §problem) khi tính `T_k`.

## 1.2 — Thuật toán `Φ` (RẼ NHÁNH theo variant — "B+")

Hai variant cho **route khác nhau** (precedence khác), `T_k` chung công thức (max completion/lớp).

### Variant **P** (mặc định) — global-class mask + per-class spread
- **Mask (env):** `get_action_mask` cho P ép **GLOBAL precedence** — chỉ phục vụ lớp **nhỏ nhất chưa xong**,
  KHÔNG reset ở depot → policy phát **một tour mạch lạc cho mỗi lớp**.
- **`_build_P`:** chia **mỗi lớp** thành **M chunk cân-demand kề nhau** (quantile) → **xoay offset chunk→xe
  per-lớp** (cân tải) → nối theo lớp tăng dần mỗi xe (precedence) → chèn **capacity reload** (multi-trip khi
  `total/M > cap`, vd M=2). ⇒ **mỗi lớp trải khắp M xe → T₁ giảm theo M** (thắng phương án "class-cut" vốn
  kẹt lớp-1 ở ~k_min xe).
- Thực đo (1 instance): precedence ✓, cap ✓, lớp-1 trải **2/2,3/3,5/5,7/7**, `T₁: 33.6→23.2→11.2→11.0`.

### Variant **U** — capacity re-split (không precedence)
- `_segment` chia order thành `max(M, k_min)` segment capacity-feasible → `_assign` LPT multi-trip khi
  `k_min>M`. `_order_trips` đẩy trip ưu tiên cao trước.

> Vì sao B+ (không phải "class-cut" đơn giản): order policy chỉ monotone **trong từng depot-segment**; re-split
> capacity vắt ngang ranh giới → P-infeasible. Global-class mask + per-class spread vừa **đúng precedence** vừa
> **trải lớp-1 ra M xe** (T₁ tốt) vừa **học khớp** (policy lo routing trong-lớp, Scheduler lo M-spread).
> `variant` được wire qua `get_reward/get_objective → calc_reward → Scheduler` (trước đây mặc định 'P').

## 1.3 — `calc_reward` mỏng lại + env/generator KHÔNG đổi

**File:** `common/cal_reward.py`
- `calc_reward` chỉ còn: gọi `Scheduler(...)` → `T_vec` → **scalarize** (RL reward = `−T_1`, giữ như
  `env.get_reward` hiện tại; vector `T` đầy đủ cho eval `get_objective`).

**File:** `env/env.py`
- **KHÔNG đổi** `get_action_mask`/rollout (policy M-agnostic; capacity+depot+P mask giữ nguyên).
- `reset` đưa `num_vehicle` (per-instance, tensor `(B,1)`) vào `td` để Scheduler đọc.

**File:** `env/generator.py` — **KHÔNG đổi `Q`** (`Σq/3 + 0.5` đã đúng paper gốc). Chỉ Phase 3 sửa pick M
per-instance (cho reward quét M), không liên quan capacity.

---

## ✅ Cổng test Phase 1

**File test:** `tests/test_scheduler.py` (test Scheduler **độc lập**, không cần rollout).

1. **⭐ T_k đúng định nghĩa:** ví dụ tay (như Figure paper) → `Scheduler` trả `(T_1,T_2,T_3)` khớp.
2. **⭐ Hierarchy-level P vs U:** cùng routes, variant P và U cho `T_k` khác đúng như paper mô tả.
3. **⭐ Đơn điệu theo M:** cùng chuỗi α, `T_1(M)` **không tăng** khi M tăng.
4. **⭐ Continuity (K≤M):** M≥3 → mỗi segment 1 xe → `T_vec` **bằng** makespan song song cũ (≤1e-6).
5. **⭐ M=2 multi-trip:** Q khít → 3 trip → 2 xe; ví dụ tay khớp scheduler; **feasible, không deadlock**.
6. **⭐ Rollout smoke (tích hợp):** `env.reset → policy(M-agnostic) → calc_reward(Scheduler) → backward`,
   reward finite — ở **`M∈{2,3,5}` × variant `{P,U}`**.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist — ✅ ĐÃ XONG
- [x] `common/scheduler.py`: class `Scheduler` multi-trip; **`_segment` resplit** (chia order→max(M,k_min) segment capacity-feasible, **bỏ qua `0` của policy**), `_assign` LPT, `_completion_times` nối tiếp.
- [x] `calc_reward` mỏng: delegate `Scheduler` → `−T_1`; `env/env.py:reset` đưa `num_vehicle (B,1)`. **Q code KHÔNG đổi.**
- [x] `tests/test_scheduler.py` (11 test): accounting==parallel-ref, resplit feasible, **⭐ M matters** (nhiều xe→makespan giảm), M=2 multitrip, đơn điệu M, calc_reward delegate, **⭐ rollout-smoke `M∈{2,3,7}×{P,U}`** — xanh.
- [x] `unittest discover` xanh — **76/76** (65 cũ không vỡ + 11 mới).
- [ ] Commit "Dynamic Phase 1: Scheduler module (M-agnostic policy + Φ)".

> **Cập nhật sau review:**
> - `[hierarchy]` — **KHÔNG phải lỗ hổng.** Theo định nghĩa chính thức (Ha 2024): `T_k` = *"min time mọi xe
>   phục vụ xong mọi arc lớp k"* = **max-over-class-k-arcs**, **giống nhau cho P và U** (ví dụ paper: U có
>   `T_3=4 < T_2=5`). Khác biệt P/U nằm ở **mask** (env). Đã khoá bằng test `Tk_variant_independent`.
> - `[assign]` — **ĐÃ LÀM:** `_order_trips` xếp trip trong xe **ưu tiên lớp cao trước** (multi-trip M<k_min);
>   đo thực tế M=2 `T_1` giảm ~10%. Còn lại: assignment trip→xe vẫn là greedy LPT (heuristic).
> - Policy vẫn M-agnostic — cho thấy M = Phase 2 (hoãn).
