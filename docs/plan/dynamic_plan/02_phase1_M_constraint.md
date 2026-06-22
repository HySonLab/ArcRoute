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

## 1.2 — Thuật toán `Φ`: multi-trip (tự suy biến single-trip khi K≤M)

`Q = Σq/3` khít (Phase 0) ⇒ Scheduler **một mode duy nhất: multi-trip**, nhưng đường nhanh single-trip khi đủ xe.

1. **Chia order** (chuỗi α của policy, bỏ qua các `0` policy tự chèn) thành **`K = max(M, ⌈Σdemand/Q⌉≈3)`
   segment capacity-feasible** (mỗi segment ≤ Q). Vì Q khít, tối thiểu ~3 segment; với M≥3 chia đúng M segment.
2. **Gán K segment → M xe:**
   - **K ≤ M** (M≥3): mỗi segment 1 xe, **single-trip** (xe dư idle). ← đường nhanh "evaluator".
   - **K > M** (vd M=2, K=3): **multi-trip** — gán bằng LPT + ưu tiên lớp, xe ≥2 segment chạy nối tiếp.
3. **`T_k`** tính trên lịch (single-trip: song song; multi-trip: cộng nối tiếp), tôn trọng P (precedence) /
   U (hierarchy-level) theo paper.

> **Continuity:** K≤M → công thức = makespan song song cũ (≤1e-6). M=2 → 1 xe ôm 2 trip → makespan cao hơn,
> không infeasible. **Không có vách deadlock.**

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

> Còn TODO (đánh dấu trong `scheduler.py`): `[hierarchy]` HDCARP-U dùng hierarchy-level cho `T_k`; `[assign]`
> xếp thứ tự trip trong xe theo ưu tiên. Policy vẫn M-agnostic — cho thấy M = Phase 2 (hoãn).
