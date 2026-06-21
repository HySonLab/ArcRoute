# Phase 2 — Khai thác mật độ `d ∈ {1.5, 2, 2.5, 3}`

> Mục tiêu: sinh đủ **cả 4 mức mật độ `d`** (đã nằm trong F1, HRDA hiện bỏ trục này) và **report theo `d`**.
> Đa dạng cấu trúc "miễn phí" — không đổi công thức. Phụ thuộc Phase 1. Tham chiếu `docs/data.md §5 P2`, §8.B.

## 2.1 — Hiểu `d` trong code

`|A| = n·d` (n = số đỉnh, d = bậc trung bình / mật độ). Hiện `sample_arcs(num_loc, num_arc)` nhận
`num_arc` trực tiếp ⇒ `d` ẩn = `num_arc/num_loc`. Để **quét d**, sinh `num_arc = round(n·d)` với
`d ∈ {1.5, 2.0, 2.5, 3.0}`.

> Vẫn phải tôn trọng trần Phase 1: `3*(round(n·d)//4) ≤ 100`. Ví dụ `n=40, d=3 → |A|=120 → |A_r|=90` ✓;
> `n=50, d=3 → |A|=150 → |A_r|=111` ✗ (vượt trần — loại). Lọc `n·d` để `|A_r| ≤ 100` (xem `docs/data.md §8.E`).

## 2.2 — Thay đổi code

**File:** `env/generator.py`
- Thêm tham số `density` (scalar hoặc list) cho `generate`. Tính `num_arc = round(num_loc * d)` nếu
  `density` được truyền (ưu tiên hơn `num_arc` trực tiếp). Random `d` từ list mỗi instance.
- Lọc: nếu `3*(num_arc//4) > 100`, resample `d`/`n` (hoặc skip).

**File:** `data/gen.py`
- Thêm `--density` nhận nhiều giá trị (mặc định `[1.5,2.0,2.5,3.0]`). Khi sinh, chọn target `|A|`
  theo `d` và bucket riêng theo `(d, |A|)`.
- Lưu `d` vào metadata `.npz` (chuẩn bị cho Phase 5): `np.savez(..., d=d_level)`.

## 2.3 — Report theo `d`

- Chuẩn bị sub-study mật độ (`docs/data.md §8.B`): fix `n=40`, biến `d ∈ {1.5,2,2.5,3}`,
  cell D1–D4 mỗi cell ≥20 seed. Đây là **dữ liệu test**, không train.
- Ghi rõ trong README/eval script rằng kết quả phải break-down theo `d`.

---

## ✅ Cổng test Phase 2

### Test cần THÊM
1. **Quan hệ `|A| ≈ n·d`:** với `n=40, d=2.5`, `num_arc == round(40*2.5) == 100`, `|A_r| = 3*(100//4)=75`.
2. **Trần vẫn giữ:** với mọi `(n, d)` được chấp nhận, `|A_r| ≤ 100`; cấu hình vượt trần bị loại/resample.
3. **Mọi mức d sinh được:** loop `d ∈ {1.5,2,2.5,3}` với `n=40` đều tạo instance hợp lệ (không NaN,
   adjacency finite, ¼-split giữ nguyên).
4. **Metadata d:** `.npz` của `data/gen.py` chứa khóa `d` đúng giá trị mức đã sinh.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `generate` nhận `density`; tính `num_arc = round(n·d)`; lọc theo trần.
- [ ] `data/gen.py --density` + lưu `d` vào `.npz`.
- [ ] Test quan hệ `|A|≈n·d`, trần, 4 mức d, metadata — xanh.
- [ ] ¼-split & cân bằng lớp (Phase 0) **không** bị phá khi đổi d.
- [ ] `unittest discover` xanh.
- [ ] Commit "Phase 2: sweep + report density d".
