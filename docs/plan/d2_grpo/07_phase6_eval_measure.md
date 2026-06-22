# Phase 6 — **Đo & eval**: per-objective curve, lexicographic win-rate, gap-to-LP, best-of-K yield

> Mục tiêu: chứng minh D2 thật sự thắng: **(1)** đường cong học per-objective (T_2/T_3 GIẢM khi T_1 ở sàn —
> dưới weighted thì phẳng); **(2)** **win-rate lexicographic vs weighted baseline** trên `data/ood/`,
> M∈{2,3,5,7,10}, **zero T_1 regression**; **(3)** gap-to-LP/baseline; **(4)** best-of-K yield vs K. Tái dùng
> `eval/run_grid.py` + `eval/stats.py`. Phụ thuộc Phase 1–4 (5 nếu làm). Không sinh data mới.

## 6.1 — Tài nguyên tái dùng (đã có)

- **Test grid:** `data/ood/<topology>/<|A|>/*.npz` (data_plan); `import_instance(f, M=)` (`common/ops.py:249`).
- **`eval/run_grid.py`:** `RLSolver` (lex selector sau Phase 1) + `run_grid` loop `(file×M×variant)` +
  `summarize`/`write_csv`; `num_sample≥10` (`eval/run_grid.py:217`, do `get_objective num_epochs=10`).
- **`eval/stats.py`:** gap-to-BKS, Wilcoxon, Friedman (đã có `tests/test_stats.py`).

## 6.2 — Bốn phép đo (headline)

| # | Đo | Cách | Kỳ vọng |
|---|---|---|---|
| **1** | **Per-objective learning curve** | log `T1/T2/T3_mean` (Phase 4) qua epoch; so **GRPO vs weighted-PPO** (cùng data/seed) | GRPO: **T_2,T_3 ↓**, T_1 ổn ở sàn. Weighted: T_2/T_3 **phẳng** |
| **2** | **Lexicographic win-rate vs weighted baseline** | mỗi instance × M∈{2,3,5,7,10}: so winner lex của **GRPO ckpt** vs **weighted ckpt**; "win" = lex (T_1,T_2,T_3) tốt hơn | win-rate > 50% **với ZERO T_1 regression** (T_1_grpo ≤ T_1_weighted mọi instance) |
| **3** | **Gap-to-LP/baseline** | RL vs EA/ACO/ILS/LP cùng `(M,variant)`; Wilcoxon/Friedman (`eval/stats.py`) | gap không tệ hơn baseline weighted; **ép `--variant P`** mọi solver |
| **4** | **Best-of-K yield** | với 1 ckpt, quét `num_sample/K ∈ {1,2,4,8,16,32}`, vẽ lex-best T vs K | T_2/T_3 giảm theo K (đường cong bão hoà) |

## 6.3 — Thay đổi cụ thể (mở rộng script, không sửa solver core)

**File:** `eval/run_grid.py`
- Thêm cột `algo`/`ckpt_tag` vào row để **A/B GRPO vs weighted** trên cùng grid.
- `summarize`: break-down `T_1,T_2,T_3` theo **M / topology / n_req / d** (đã có khung `summarize`); thêm
  **paired win-rate** GRPO-vs-weighted + **đếm T_1 regression** (phải = 0).
- **Best-of-K yield:** loop `num_sample` ∈ list → ghi lex-best T per K (cờ `--yield_curve`).

**File:** `eval/stats.py` — dùng như cũ (gap, Wilcoxon, Friedman); thêm helper `win_rate(a,b)` lexicographic
nếu chưa có (so (T_1,T_2,T_3)).

> **Bất biến đo:** mọi so sánh dùng **cùng selector lexicographic** (Phase 1) cho cả GRPO lẫn weighted ⇒
> cô lập đóng góp của **tín hiệu học** (không lẫn với free-win selector). T_1 regression count **phải 0**.

---

## ✅ Cổng test Phase 6

**File test mới/ mở rộng:** `tests/test_eval_grid.py` (đã có) + thêm assert; `tests/test_stats.py` (đã có).

1. **⭐ Win-rate lexicographic đúng:** dựng 2 mảng T (grpo,weighted) tay → `win_rate` đếm đúng số lex-thắng;
   **T_1 regression count** đúng (ví dụ có/không regression).
2. **⭐ Zero T_1 regression (synthetic):** với cặp ckpt giả mà T_1 bằng nhau, regression count == 0.
3. **Grid rows + monotone theo M:** (đã có ở `test_eval_grid.py`) — giữ xanh; thêm cột `algo` không phá schema.
4. **Best-of-K yield shape:** `--yield_curve` trả 1 row/K, T finite, **không tăng** theo K (lex-best đơn điệu
   không xấu đi khi thêm sample) — report, assert mềm.
5. **Stats hợp lệ:** `eval/stats.py` p-value∈[0,1], gap dấu đúng (`tests/test_stats.py` xanh).
6. **Suite cũ xanh.**

> Phép đo **1/3/4 chạy thật** cần ckpt train đầy đủ (ngoài unittest); unittest chỉ khoá **logic
> aggregation/win-rate/schema** (skippable khi thiếu `data/ood`/ckpt) — giống `test_eval_grid.py` hiện tại.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python eval/run_grid.py --ckpt <grpo.ckpt>  --path data/ood --variant P   # GRPO
uv run python eval/run_grid.py --ckpt <weighted.ckpt> --path data/ood --variant P # baseline A/B
uv run python eval/run_grid.py --ckpt <grpo.ckpt> --yield_curve --path data/ood/unit_square
```

### Checklist
- [ ] `eval/run_grid.py`: cột `algo`; paired win-rate + T_1-regression count; `--yield_curve`.
- [ ] `eval/stats.py`: `win_rate` lexicographic (nếu thiếu); gap/Wilcoxon/Friedman như cũ.
- [ ] Per-objective curve GRPO vs weighted (từ logs Phase 4) — report.
- [ ] **Báo cáo: win-rate > 50%, T_1 regression = 0**; gap-to-LP; best-of-K yield.
- [ ] `tests/`: ⭐ win-rate, ⭐ zero-regress, yield shape, monotone-M, stats hợp lệ.
- [ ] `unittest discover` xanh.
- [ ] Commit.

### Commit message
```
D2 Phase 6: eval & measure (per-objective curves, lex win-rate, yield)

run_grid/stats gain algo column, paired lexicographic win-rate + T_1-regression
count, best-of-K yield curve. Reports GRPO vs weighted on data/ood across
M in {2,3,5,7,10}: T_2/T_3 descend, win-rate>50%, zero T_1 regression.
```
