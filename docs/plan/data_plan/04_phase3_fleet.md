# Phase 3 — Số xe `M`: tham số lúc GIẢI, KHÔNG phải trục sinh data

> ⚠️ **BẢN SỬA (supersedes lần implement đầu).** Lần đầu Phase 3 sinh **riêng từng M** vào `data/<M>m/`.
> Đó là **dư thừa và sai về khoa học**: instance (arcs, demand, **C**) **độc lập với M**, nên phải **sinh
> arcs MỘT LẦN** rồi **chỉ định M lúc giải/eval**. Tham chiếu `docs/data.md §3.1, §8.C`.
>
> 🔒 **Ngoài phạm vi (bạn xử lý sau):** đưa `M` thành **input của policy** (M-conditioning) + enforce M
> trong masking. Hiện policy KHÔNG condition trên M (features chỉ `[demand,clss,service,traversal]`). Phase
> này chỉ chỉnh **data + eval**; training giữ `M` là một scalar cố định.

## 3.1 — Vì sao M không phải thuộc tính của instance

`C = Σ_{a∈A_r} q_a / 3 + 0.5` — số **`3` là HẰNG trong công thức paper** ("3 homogeneous vehicles"),
**KHÔNG phải fleet thật**. Khi quét M ta **giữ `C` cố định** (docs §3.1), chỉ số route khả dụng đổi.
⇒ Toàn bộ `.npz` (arcs, demand, class, service, traversal, **C**) **không đổi theo M**.

- Đã chứng minh: `tests/test_gen.py::test_capacity_invariant_to_M` (cùng instance, M=1 vs M=10 → `C` y hệt).
- Thứ duy nhất phụ thuộc M: `tau = Σq/(M·C)` — **suy ra được** từ `req`+`C`+`M`, tính lúc report.

**Hệ quả:** sinh per-M = nhân bản cùng loại instance; tệ hơn, mỗi M là **bộ ngẫu nhiên KHÁC nhau** →
không làm được fleet study đúng (vốn cần **cùng instance, đổi M** để so sánh ghép cặp/Wilcoxon).

## 3.2 — Thiết kế đúng

```
Sinh arcs MỘT LẦN cho mỗi (topology, size)   →  data/ood/<topology>/<|A|>/*.npz   (KHÔNG có tầng <M>m)
Chỉ định M lúc GIẢI/EVAL                       →  import_instance(..., M=<override>) + baseline --M
Fleet study (Bảng C)                           =  giải CÙNG bộ instance dưới M ∈ {1,2,3,5,7,10}
tau                                            =  tính lúc report theo M đã chọn
```

## 3.3 — Thay đổi code

**File:** `data/gen.py`
- **Bỏ vòng lặp per-M** khi sinh. Mỗi cell sinh 1 lần; **bỏ tầng thư mục `<M>m/`** →
  `data/ood/<topology>/<|A|>/*.npz` (và `data/<topology>/<|A|>/` cho OSM in-dist).
- `--vehicles` đổi nghĩa: chỉ là **M nominal** ghi vào `.npz` (mặc định 3) để loader cũ chạy được; KHÔNG
  còn nhân thư mục theo M. (Hoặc bỏ hẳn, lưu `M` rỗng và bắt eval truyền `--M`.)
- `_save_instance`: vẫn lưu `tau` cho **M nominal** + ghi rõ "reference"; report tự tính lại theo M eval.

**File:** `common/ops.py`
- `import_instance(es, M=None)`: nếu `M` được truyền → dùng nó thay cho `es['M']` (override lúc eval).
  Không truyền → giữ `es['M']` (backward-compatible).

**File:** `baseline/*.py` (aco, ea, ils, lp, rl_hyb)
- Thêm `--M` (nargs hoặc đơn) để **quét/đặt fleet lúc eval**; loop qua M cho fleet study. `--path` trỏ
  `data/ood/<topology>` rồi glob `/*/*.npz` (giờ tầng dưới là `<|A|>`, hợp lệ).

**File:** `env/generator.py`
- Giữ nguyên: `num_vehicle` vẫn nhận scalar/list (cho training on-the-fly). KHÔNG đổi (M-conditioning để sau).

## 3.4 — Ảnh hưởng layout (cập nhật Phase 4/5)

- `data/ood/<topology>/<|A|>/*.npz` (bỏ `<M>m`). Số instance giảm mạnh (unit_square ×6 → ×1).
- Phase 4 (OOD) & Phase 5 (tau) tham chiếu layout mới — xem các file tương ứng.

---

## ✅ Cổng test Phase 3 (bản sửa)

### Test cần THÊM/SỬA
1. **Capacity bất biến theo M (giữ):** `test_capacity_invariant_to_M` — cùng instance, M khác → `C` bằng.
2. **`import_instance` override M:** load 1 `.npz`, gọi `import_instance(f, M=7)` → `M==[0..6]`; không
   truyền → dùng `es['M']`. arcs/C/demands **không đổi** giữa hai lần (chỉ list M khác).
3. **Sinh 1 lần, không tầng `<M>m`:** `gen_synth`/`gen_graph` ghi vào `data/ood/<topology>/<|A|>/`; không
   tạo thư mục theo M.
4. **tau theo M (report-time):** với cùng instance, `tau(M)=Σq/(M·C)` đơn điệu giảm theo M (tính từ `req`+`C`).
5. **Baseline `--M`:** parse OK; eval cùng bộ file dưới nhiều M.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `data/gen.py`: bỏ per-M loop + tầng `<M>m`; sinh 1 lần/cell.
- [ ] `import_instance(es, M=None)` override M; backward-compatible.
- [ ] `baseline/*`: thêm `--M` (fleet là trục **eval**, không phải sinh).
- [ ] `tau` = report-time theo M; lưu reference cho M nominal.
- [ ] Test: capacity-invariant, import override M, sinh-không-`<M>m`, tau theo M, baseline `--M` — xanh.
- [ ] Regenerate grid theo layout mới (`data/ood/<topology>/<|A|>/`).
- [ ] Commit "Phase 3 (revised): M là tham số eval, sinh arcs 1 lần".

> Lưu ý: policy M-conditioning (đưa M vào input mạng) **KHÔNG thuộc phase này** — bạn xử lý sau.
