# Phase 5 — Eval & report theo `M` và `size`

> Mục tiêu: đánh giá model động (M-conditioned, đa size) trên **test grid sẵn có** + `eval/stats.py`, báo
> cáo generalization theo M và size. Phụ thuộc Phase 1–3 (4 nếu làm). Không sinh data mới.

## 5.1 — Tài nguyên đã có (tái dùng)

- **Test grid:** `data/ood/<topology>/<|A|>/*.npz` (unit_square + cluster + osm_cityA/B), 800 npz.
- **Fleet là trục eval:** `import_instance(f, M=...)` + `baseline --M` (data_plan Phase 3 revised) → giải
  **cùng instance dưới mỗi M** (paired).
- **Thống kê:** `eval/stats.py` (gap-to-BKS, Wilcoxon, Friedman, tightness τ theo M).

## 5.2 — Ma trận báo cáo

| Trục | Cách | Nguồn |
|---|---|---|
| **Fleet M** | mỗi instance giải dưới `M∈{3,5,7,10}` (override; **thêm M=2** theo calib đã chốt) | `--M` / `import_instance(M=)` |
| **Size** | break-down theo `\|A_r\|` (size ladder) | metadata `n_req` |
| **Density d** | break-down theo `d` | metadata `d` |
| **Topology** | in-dist (unit_square) vs OOD (cluster/osm) | metadata `topology` |
| **Variant** | **`P` (mặc định, báo cáo chính)**; `U` để so sánh | `--variant` |

- **RL vs baselines** (EA/ACO/ILS/LP) trên cùng grid + cùng M **+ cùng variant** → Wilcoxon/Friedman.
  ⚠️ Baseline default lệch (`aco/ea`→`U`, `ils/lp/rl_hyb`→`P`); **ép tất cả về `--variant P`** khi so sánh.
- **Ablation headline:** 1 model **M-agnostic** (train once) vs **model-mỗi-M** vs **M-conditioned** (Phase 2,
  nếu đã implement) trên cùng grid → chứng minh M-agnostic + Scheduler là đủ tốt.
- **Ablation Scheduler** (nếu muốn): greedy vs LPT vs optimal-assignment `Φ` → tách đóng góp của Scheduler.
- **τ theo M** (report-time, data_plan Phase 5) để mô tả độ khó từng cấu hình.

## 5.3 — Script — ✅ ĐÃ SCAFFOLD (`eval/run_grid.py`)

**File:** `eval/run_grid.py` (đã dựng, đã validate):
- `load_instance_td(file, M)`: dựng td từ `.npz` (qua `import_instance(M=)`) đúng key của env hiện tại
  (rl_hyb.py cũ **stale** — dùng `service_time`/`env.reset(batch_size=)` sai; KHÔNG dùng).
- `RLSolver(ckpt)`: load PPO ckpt → sample `num_sample` rollout (decode `sampling`) → `env.get_objective`
  → giữ best theo `T_1` (kiểu HRDA). ⚠️ `get_objective` dùng `run_parallel(num_epochs=10)` → `num_sample ≥ 10`.
- `run_grid` loop `(file × M × variant)` → thu `(T_1,T_2,T_3)` + metadata; `write_csv`.
- `summarize`: dùng `eval/stats.py:describe` — break-down `T_1` theo **M / topology / n_req / density** +
  sanity **monotone theo M**.
- `--dry-run` (không cần model) để kiểm scaffold; đã chạy thật trên ckpt throwaway (M=3→7 giảm đúng).
- **Test:** `tests/test_eval_grid.py` (metadata, grid rows + monotone, CSV) — xanh (skip nếu thiếu `data/ood`).

**Còn lại (cần checkpoint train đầy đủ):** chạy thật trên toàn `data/ood`; thêm **baseline (EA/ACO/ILS/LP)**
cùng `(M, variant)` rồi đẩy vào `gap_summary`/`pairwise_wilcoxon`/`friedman` (TODO đánh dấu trong script).

---

## ✅ Cổng test Phase 5

1. **Eval chạy cùng instance đa M:** 1 file, `M∈{3,5,7,10}` → 4 kết quả; **makespan giảm (hoặc không tăng)
   khi M tăng** (nhiều xe → song song/ít chuyến nối tiếp hơn). Scheduler khiến tính đơn điệu này gần như
   **cấu trúc** (cùng chuỗi α, M lớn hơn chỉ nới phân bổ) — vẫn report, không assert cứng.
2. **Break-down đúng trục:** group theo `n_req/d/topology` từ metadata khớp số file.
3. **Stats hợp lệ:** `eval/stats.py` trên kết quả thật → p-value∈[0,1], gap-to-BKS dấu đúng (đã có test ở
   `tests/test_stats.py`).
4. **In-dist ≤ OOD (kỳ vọng):** model thường tốt hơn trên unit_square (train-dist) so với osm/cluster —
   **report**, không assert cứng (chỉ log để quan sát generalization gap).

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python baseline/rl_hyb.py --variant P --cpkt <ckpt> --path data/ood/unit_square --M 5
uv run python eval/stats.py --tightness data/ood/unit_square
```

### Checklist
- [ ] Eval loop `(file × M)` dùng `--M` override; thu makespan per class. **Variant mặc định `P`.**
- [ ] Break-down theo size/d/topology **(× variant)**; bảng RL vs baselines **ép cùng `--variant P`**.
- [ ] `eval/stats.py`: Wilcoxon/Friedman/gap-to-BKS theo trục.
- [ ] Sanity: makespan đơn điệu theo M; report generalization gap in-dist vs OOD.
- [ ] `unittest discover` xanh.
- [ ] Commit "Dynamic Phase 5: eval & report theo M/size trên test grid".
