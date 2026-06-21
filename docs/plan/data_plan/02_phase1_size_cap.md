# Phase 1 — Trần cứng `|A_r| ≤ 100`, dải size, bucket theo size

> Mục tiêu: chốt **trần cứng `|A_r| ≤ 100`** (⇒ `|A| ≤ ~133`), hỗ trợ sinh theo **dải size**, và
> đảm bảo **mỗi batch chỉ 1 size `n`** (ràng buộc encoder). Phụ thuộc Phase 0 (¼-split).
> Tham chiếu: `docs/data.md §5 P1`, §5.7.

## 1.1 — Size ladder và quan hệ size

Với ¼-split (F2): `|A_r| = 3·⌊|A|/4⌋ ≈ 0.75·|A|`. Ladder mục tiêu (xem `docs/data.md §8`):

| `|A_r|` | `|A|` ≈ | `n = |A_r|+1` | batch khả thi (4090) |
|---|---|---|---|
| 20 | 27 | 21 | ~4096 |
| 40 | 53 | 41 | ~4096 |
| 60 | 80 | 61 | ~4096 |
| 80 | 107 | 81 | ~3000 |
| **100 (trần)** | 133 | 101 | ~2000 |

> Quy đổi: muốn `|A_r| = R` thì đặt `num_arc = |A| ≈ round(R/0.75)` (chính xác: chọn `|A|` sao cho
> `3·(|A|//4) = R`, vd `R=60 → |A|=80`; `R=100 → |A|=133` cho `3·33=99`, hoặc `|A|=134→3·33=99`...
> chốt `|A_r|` thực tế = `3·(|A|//4)`, **không ép tròn 100** — báo cáo giá trị thực).

## 1.2 — Hỗ trợ dải trong `env/generator.py`

**File:** `env/generator.py` → `generate`, `save_cache`, `CARPGenerator`, `WrapDataset`.

- Cho `num_loc`, `num_arc`, `num_vehicle` nhận **scalar HOẶC (low, high)**. Mỗi instance random 1 giá trị
  trong dải (giữ on-the-fly). Thêm helper:
  ```python
  def _pick(v):   # v là int hoặc (lo, hi)
      if isinstance(v, (tuple, list)):
          return int(torch.randint(v[0], v[1] + 1, (1,)))
      return v
  ```
- **CHỐT TRẦN:** sau khi pick, đảm bảo `|A_r| = 3*(num_arc//4) ≤ 100` (clamp `num_arc ≤ 133`).
  Thêm `assert 3*(num_arc//4) <= 100` trong `generate` để fail-fast nếu cấu hình vượt trần.

## 1.3 — Bucket theo size (BẮT BUỘC vì encoder chưa hỗ trợ mask)

> `policy/encoder.py:257 → assert mask is None, "Mask not yet supported!"` và
> `collate_fn = torch.cat(adj (1,n,n))` ⇒ **không trộn nhiều `n` trong 1 batch**.

- Khi sinh dải size, **bucket**: gom instance theo `n` rồi mỗi batch lấy từ 1 bucket. Cách an toàn nhất:
  giữ `save_cache` sinh **từng size riêng** (một file cache / size), rồi train chọn bucket theo epoch.
- Nếu train "mixed ≤100": cần sampler trả về batch đồng nhất `n`. Ghi rõ trong code rằng **không** được
  `torch.cat` hai `n` khác nhau (sẽ lỗi shape ở `collate_fn`).
- Cập nhật `train.sh` / `train.py`: cho phép truyền dải hoặc danh sách size; mặc định train 1 mức
  (vd `|A_r|=60`) rồi test cả thang.

## 1.4 — Sinh size ladder cho benchmark `data/gen.py`

- `main`: `buckets` hiện `range(first, 201, 10)` → **giới hạn theo trần**: chỉ giữ bucket có
  `3*(B//4) ≤ 100` (tức `B ≤ 133`). Hoặc đặt buckets theo `|A|` tương ứng ladder ở 1.1.
- Cho phép cờ `--max_req 100` để fail-fast nếu bucket vượt trần.

---

## ✅ Cổng test Phase 1 (PHẢI PASS mới qua Phase 2/3)

### Test cần THÊM
1. **Trần cứng:** với mọi `num_arc` trong dải sinh, `3*(num_arc//4) <= 100`. Test bằng cách gọi
   `generate` / `build_instance` ở các size biên (133, 134) và assert vượt 134 thì raise/clamp.
2. **Dải size:** gọi `generate(num_loc=(20,50), num_arc=(27,133), num_vehicle=(2,7))` nhiều lần,
   assert `n` thay đổi và luôn ≤ 101, `num_vehicle` trong [2,7].
3. **Đồng nhất batch:** mô phỏng `collate_fn` (`torch.cat`) trên 2 td **cùng `n`** → OK; trên 2 td
   **khác `n`** → assert raise (chứng minh phải bucket). Đây là test "chống lỗi trộn size".
4. **Ladder benchmark:** `required_count` tại các `|A|` của ladder cho ra `|A_r| ∈ {≈20,40,60,80,100}`.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
# Smoke test sinh thật 1 size nhỏ (không cần GPU):
uv run python -c "from env.generator import generate; td=generate((20,40),(27,80),(2,5)); print(td['adj'].shape)"
```

### Checklist
- [ ] `generate`/`save_cache` nhận dải; clamp/assert `|A_r| ≤ 100`.
- [ ] Cơ chế bucket theo size, có comment chỉ rõ ràng buộc `assert mask is None`.
- [ ] `data/gen.py` buckets bị giới hạn theo trần; cờ `--max_req`.
- [ ] Test trần + dải + đồng nhất batch + ladder đều xanh.
- [ ] `unittest discover` xanh toàn bộ (gồm cả test Phase 0).
- [ ] Commit "Phase 1: hard cap |A_r|≤100 + size ladder + bucketing".
