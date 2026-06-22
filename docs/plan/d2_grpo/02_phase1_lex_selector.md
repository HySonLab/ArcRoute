# Phase 1 — **Lexicographic best-of-K selector** (FREE WIN, không retrain)

> Mục tiêu: sửa **2 selector best-of-K** hiện chọn theo **T_1 đơn** → chọn **lexicographic** trên
> (T_1,T_2,T_3). Đây là cải thiện chất lượng **ngay lập tức, KHÔNG cần train lại** + tạo **baseline đo**
> cho các phase sau. Độc lập, rủi ro thấp. Code: `eval/run_grid.py`, `baseline/rl_hyb.py`.

## 1.1 — Lỗi hiện tại (đã xác nhận)

- `eval/run_grid.py:111`: `best = obj[obj[:, 0].argmin()]  # best by top-priority T_1`
  → bỏ qua T_2/T_3; trong khối hoà T_1 (rất lớn) chọn **ngẫu nhiên** theo thứ tự sample.
- `baseline/rl_hyb.py:37`: `idx = obj[:, 0].argmin()` → cùng lỗi.

`obj` là `(num_sample, 3)` = `(T_1,T_2,T_3)` từ `env.get_objective` (`env/env.py:144-148` →
`run_parallel(calc_reward,...)` trả full vector).

## 1.2 — Thay đổi cụ thể

**File:** `eval/run_grid.py` (~dòng 111, trong `RLSolver.solve`)
```python
# CŨ:  best = obj[obj[:, 0].argmin()]
# MỚI: lexicographic argmin trên (T_1, T_2, T_3) — T_1 ưu tiên, T_2 phá hoà, T_3 phá hoà tiếp.
best = obj[np.lexsort((obj[:, 2], obj[:, 1], obj[:, 0]))[0]]
```
> `np.lexsort` lấy **key cuối cùng làm primary** → đặt `obj[:,0]` (T_1) cuối tuple ⇒ T_1 primary. `[0]` =
> index nghiệm tốt nhất.

**File:** `baseline/rl_hyb.py` (~dòng 37, trong `RLHCARP.__call__`)
```python
# CŨ:  idx = obj[:, 0].argmin()
# MỚI:
idx = np.lexsort((obj[:, 2], obj[:, 1], obj[:, 0]))[0]
```
⚠️ `obj` ở rl_hyb là tensor (từ `get_objective`); ép `np.asarray(obj)` trước `lexsort`, hoặc dùng torch
tương đương. Giữ `obj = obj[idx]` phía sau.

**Không đổi** gì khác (Scheduler, mask, policy nguyên).

## 1.3 — Vì sao đây là free win

Cùng K nghiệm sample, weighted-train policy đã sinh **nhiều nghiệm hoà T_1 khác nhau ở T_2/T_3**. Selector
cũ vứt thông tin đó; lexicographic argmin **nhặt đúng nghiệm tốt nhất** trong khối hoà → giảm T_2/T_3 mà
**T_1 không đổi** (cùng tập, cùng min T_1). Không train lại.

---

## ✅ Cổng test Phase 1

**File test mới:** `tests/test_lex_selector.py` (test thuần hàm chọn, không cần model).

1. **⭐ Lexicographic đúng:** dựng `obj` tay, vd
   `[[5,9,9],[5,2,9],[5,2,1],[7,0,0]]` → `lexsort((obj[:,2],obj[:,1],obj[:,0]))[0]` phải là **index 2**
   (`[5,2,1]`): T_1 nhỏ nhất (5) ∧ trong khối T_1=5 thì T_2 nhỏ nhất (2) ∧ T_3 nhỏ nhất (1).
2. **⭐ Không regress T_1:** với mọi `obj` ngẫu nhiên, `selected[0] == obj[:,0].min()` (T_1 của nghiệm chọn
   = T_1 nhỏ nhất) — lexicographic **không bao giờ** chọn T_1 tệ hơn argmin cũ.
3. **Tie-break thật xảy ra:** trên `obj` có nhiều hàng cùng T_1 khác T_2/T_3, nghiệm chọn của lex **khác**
   (hoặc ≤ về T_2) so với `obj[:,0].argmin()` (chọn hàng đầu). Khẳng định lex ≤ argmin-cũ theo (T_2,T_3).
4. **Smoke selector tích hợp** *(skip nếu thiếu ckpt/`data/ood`)*: gọi `RLSolver.solve`/`RLHCARP.__call__`
   với `num_sample≥10` → trả `(3,)` finite, `T_1 == min` trên batch.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_lex_selector -v
```

### Checklist
- [ ] `eval/run_grid.py:111` → lexicographic argmin.
- [ ] `baseline/rl_hyb.py:37` → lexicographic argmin (`np.asarray` trước `lexsort`).
- [ ] `tests/test_lex_selector.py`: ⭐ lex đúng, ⭐ no-T_1-regress, tie-break, smoke (skippable).
- [ ] `unittest discover` xanh — **90 cũ + mới**.
- [ ] Commit.

### Commit message
```
D2 Phase 1: lexicographic best-of-K selector (free win, no retrain)

run_grid.py + rl_hyb.py picked best sample by T_1 only; now lexsort over
(T_1,T_2,T_3) so ties on T_1 are broken by T_2 then T_3. No T_1 regression.
```
