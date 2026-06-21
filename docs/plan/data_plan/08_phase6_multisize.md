# Phase 6 — Train đa size (bucketed) cho size-generalization

> Mục tiêu: train policy trên **nhiều size** (size ladder) thay vì 1 size, để test tốt trên cả thang
> `|A_r|`. Đòn bẩy chính cho generalization. Phụ thuộc Phase 1 (cap + bucketing). Tham chiếu `docs/data.md §8.E`.

## 6.1 — Vì sao

Test có **size ladder** `|A_r|∈{30,45,60,75,90}`. Train 1 size → kém ở biên (extrapolation). Train đa
size phủ phân phối test → size-generalization (đúng spec train của `docs/data.md §8.E`: `n∈{20,30,40,50}`).

Ràng buộc: encoder cấm trộn size trong 1 batch (`assert mask is None`). ⇒ **bucket theo size**: mỗi batch
1 size, các size luân phiên qua các batch.

## 6.2 — Thay đổi code (đã implement)

**File:** `env/generator.py`
- `generate_dataset(num_samples, num_loc, num_arc, num_vehicle)` — tách ra từ `save_cache`, sinh 1 size.
- `MultiSizeCARPGenerator(num_samples, sizes, num_vehicle)` — `sizes` = list `(num_loc, num_arc)`; gom data
  theo bucket size; phơi `bucket_ranges`.
- `SizeBucketBatchSampler(bucket_ranges, batch_size, shuffle)` — mỗi batch chỉ chứa index của **1 bucket**;
  xáo trộn trong bucket + thứ tự batch mỗi epoch.

**File:** `env/env.py`
- `CARPEnv(sizes=...)`: nếu có `sizes` → `dataset()` dùng `MultiSizeCARPGenerator` + `batch_sampler`.

**File:** `train.py` / `train.sh`
- `--sizes "20:40,30:60,40:80,50:100,40:120"` (`|A_r|∈{30,45,60,75,90}`). Fleet M cố định; topology
  unit_square. (M-conditioning để sau — xem Phase 3.)

## 6.3 — Phân vai (nhắc lại)

- **Train:** unit_square, **đa size** (ladder), M cố định, 1 model/variant.
- **Test in-dist:** unit_square cả ladder.
- **Test OOD:** cluster + OSM (Phase 4), M là trục eval (Phase 3 revised).

---

## ✅ Cổng test Phase 6 (đã xanh)

1. **Mỗi batch single-size:** iterate dataloader → mọi batch `adj` vuông cùng `n`; `collate=torch.cat` không lỗi.
2. **Đủ size:** tất cả size trong ladder đều xuất hiện qua các batch.
3. **Không batch nào vượt bucket:** mọi index trong 1 batch thuộc đúng 1 `bucket_range`.
4. **Train step thật:** dataloader đa size → `env.reset` → policy forward → reward → `loss.backward` chạy,
   có gradient (verified: sizes {31,46,61}).

### Checklist
- [x] `generate_dataset` + `MultiSizeCARPGenerator` + `SizeBucketBatchSampler`.
- [x] `CARPEnv(sizes=...)` → multi-size dataloader.
- [x] `train.py --sizes`, `train.sh` ladder.
- [x] `tests/test_multisize.py` xanh + train-step smoke.
- [x] Commit "Phase 6: multi-size bucketed training".
