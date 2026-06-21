# Phase 3 — Quét số xe `M ∈ {1, 2, 3, 5, 7, 10}`

> Mục tiêu: coi số xe `M` là **tham số input hợp lệ** (vì mục tiêu là **makespan** — thêm xe → makespan
> giảm), quét dải rộng, train/test nhất quán. **Giữ nguyên F5** (`Q = Σ q/3 + 0.5`; chỉ số xe khả dụng
> đổi). `M=1` nối lý thuyết HCPP. Phụ thuộc Phase 1. Tham chiếu `docs/data.md §3.1, §5 P3, §8.C`.

## 3.1 — Điểm mấu chốt: M KHÔNG đổi capacity

`Q` = `(Σ_{a∈A_r} q_a)/3 + 0.5` — **hằng `/3` không đổi theo M** (đổi nó = out-of-scope, đổi F5).
M chỉ là số route khả dụng. Kiểm tra:
- `env/generator.py:sample_vehicle_capacity` — KHÔNG phụ thuộc `num_vehicle` (đúng). **Lưu ý:** công thức
  này đã được **Phase 0 §0.5 sửa** từ `Σ(q/3+0.5)` → `Σq/3+0.5`; Phase 3 dùng bản đã sửa, không đụng lại.
- `data/gen.py:build_instance` — `C` không nhận `M` vào công thức (đúng; `M` chỉ ghi vào `.npz`).

> ⚠️ Đừng "sửa nhầm" capacity theo M, và đừng vô tình quay lại công thức `Σ(q/3+0.5)` cũ. Test 3 chốt điều này.

## 3.2 — Thay đổi code

**File:** `env/generator.py`
- `num_vehicle` đã hỗ trợ dải sau Phase 1 (mục 1.2). Bổ sung: nhận **danh sách rời rạc**
  `M ∈ {1,2,3,5,7,10}` (không phải khoảng liên tục) — random.choice từ list.

**File:** `data/gen.py`
- `--vehicles`: bỏ `choices=[2,5]`, cho nhận **nhiều giá trị** (`nargs='+'`, default `[1,2,3,5,7,10]`).
- Vòng sinh: với mỗi `M` trong list, ghi vào thư mục `data/<M>m/...` như cũ (cấu trúc thư mục theo M
  đã có sẵn — chỉ mở rộng tập M).
- `refine_routes` trong `common/ops.py` có `max_vehicles=3` default — kiểm tra nơi gọi để không hard-code 3.

## 3.3 — Sub-study fleet (test)

Chuẩn bị cell F1–F6 (`docs/data.md §8.C`): fix `|A|=80, |A_r|=60`, biến `M ∈ {1,2,3,5,7,10}`,
mỗi cell ≥20 seed, cả variant P & U. F1 (M=1) = HCPP single-vehicle, F3 (M=3) = gốc Hà.

---

## ✅ Cổng test Phase 3

### Test cần THÊM
1. **Dải M rời rạc:** `generate(..., num_vehicle=[1,2,3,5,7,10])` nhiều lần → `td['num_vehicle']` luôn
   thuộc tập đó.
2. **M=1 hợp lệ:** sinh được instance với `M=1` (không chia 0, không lỗi).
3. **Capacity bất biến theo M (QUAN TRỌNG):** cùng seed/instance, đổi `M` từ 1→10, `C` (và `Q`) **không đổi**.
   ```python
   # build_instance với cùng edges/coords/rng-seed nhưng M khác → C bằng nhau
   assert C_M1 == C_M10
   ```
4. **`data/gen.py --vehicles 1 2 3 5 7 10`:** parse OK, tạo đúng các thư mục `1m,2m,3m,5m,7m,10m`.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `num_vehicle` nhận danh sách rời rạc {1,2,3,5,7,10}.
- [ ] `data/gen.py --vehicles nargs='+'`, bỏ `choices=[2,5]`.
- [ ] Kiểm tra `refine_routes`/nơi gọi không hard-code `max_vehicles=3`.
- [ ] Test: dải M, M=1, **capacity bất biến theo M**, parse CLI — xanh.
- [ ] `unittest discover` xanh.
- [ ] Commit "Phase 3: sweep fleet M (keep F5 capacity)".
