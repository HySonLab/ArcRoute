# Phase 5 — Chặt chẽ thống kê + metadata báo cáo

> Mục tiêu: nâng độ tin cậy lên chuẩn Q1/A* — **≥20–30 seed/cell**, kiểm định thống kê, **gap-to-BKS**,
> và report **tightness `τ`**. Thuần quy trình + metadata, **không đụng công thức data**. Phụ thuộc P0–P4.
> Tham chiếu `docs/data.md §5 P5`.

## 5.1 — Seeds ≥ 20–30 / cell

- Mọi cell test (A/B/C/D ở `docs/data.md §8`) sinh **≥20 seed** (gốc chỉ 5). `data/gen.py --per_bucket`
  đã có (default 20) — nâng default lên 20–30 và đảm bảo seed độc lập (seed = base + index).
- Report **mean ± std** mỗi cell, không chỉ 1 con số.

## 5.2 — Metadata vào `.npz` (gom các phase trước)

Thêm/đảm bảo các khóa trong mọi `.npz`: `d` (mật độ), `M` (fleet), `topology`, `|A_r|`, và **`τ`**:

```
τ = Σ_{a∈A_r} q_a / (M · Q)     # tightness — độ chặt capacity (Smith-Miles 2023)
```
> `τ` mô tả phân bố độ khó; tính sẵn lúc sinh và lưu để eval/report khỏi tính lại.
> ⚠️ **Phụ thuộc `Q` đã sửa đúng ở Phase 0 §0.5** (`Σq/3+0.5`). Nếu còn dùng `Q` cũ `Σ(q/3+0.5)` (quá lớn)
> thì `τ` sẽ nhỏ giả tạo → báo cáo độ khó sai lệch. Kiểm tra `Q` đúng trước khi tính `τ`.

**File:** `data/gen.py:gen_graph` → `np.savez(fpath, req=..., nonreq=..., P=3, M=M, C=C, d=d, tau=tau, topology=topo, n_req=len(req))`.

## 5.3 — Script đánh giá thống kê (mới)

Tạo `eval/stats.py` (hoặc `tests/`-độc lập script) tính:
- **Wilcoxon signed-rank** (so cặp 2 thuật toán, α=0.05).
- **Friedman + Critical-Difference diagram** (≥3 thuật toán).
- **gap-to-BKS** = `(obj − BKS) / BKS` (best & mean), runtime chuẩn hóa (Accorsi-Lodi-Vigo 2022).
- Break-down theo trục: size, `d`, `M`, topology.

> Dùng `scipy.stats` (`wilcoxon`, `friedmanchisquare`). CD diagram: tự vẽ hoặc lib `Orange`/`scikit-posthocs`.
> Thêm dependency qua `uv add scipy scikit-posthocs` nếu chưa có.

## 5.4 — Báo cáo tightness

- Histogram/phân bố `τ` mỗi cell để mô tả độ khó (Smith-Miles 2023). Không lọc bỏ instance theo `τ`
  (giữ phân bố tự nhiên), chỉ **report**.

---

## ✅ Cổng test Phase 5

### Test cần THÊM
1. **Công thức `τ`:** unit test cho hàm tính tightness: `τ == (q_req.sum()) / (M * C)`; với cùng instance,
   `M` lớn → `τ` nhỏ (đơn điệu giảm theo M).
2. **Metadata đầy đủ:** mọi `.npz` mới có đủ khóa `d, M, topology, tau, n_req`; `import_instance` vẫn load OK
   (các khóa thừa không phá `np.load`).
3. **Seed độc lập:** 2 instance cùng cell khác seed → khác nhau (không trùng do quên reseed).
4. **Stats script chạy:** trên dữ liệu giả (2–3 "thuật toán" × N seed), `wilcoxon`/`friedman` trả p-value
   hợp lệ ∈ [0,1]; gap-to-BKS tính đúng dấu (obj=BKS → gap=0).

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python eval/stats.py --selftest      # nếu thêm cờ self-test trên dữ liệu giả
```

### Checklist
- [ ] `--per_bucket` ≥ 20; seed độc lập theo index.
- [ ] `.npz` có `d, M, topology, tau, n_req`; `import_instance` không vỡ.
- [ ] `eval/stats.py`: Wilcoxon + Friedman/CD + gap-to-BKS + break-down theo trục.
- [ ] Report `τ` (mean±std / histogram) mỗi cell.
- [ ] Test τ, metadata, seed, stats — xanh.
- [ ] `unittest discover` xanh.
- [ ] Commit "Phase 5: statistical rigor (seeds, Wilcoxon/Friedman, τ, gap-to-BKS)".
