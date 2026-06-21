# Plan tổng quan — Cải tổ Dataset HDCARP

> Nguồn: [`docs/data.md`](../../data.md). Mục tiêu: đưa data của repo **về đúng công thức gốc
> Hà 2024 (F1–F5)** rồi mở rộng các trục **KHÔNG đổi công thức** (quy mô, mật độ `d`, số xe `M`,
> topology cho test OOD, thống kê) để đủ mạnh cho bài báo Q1/A*.

## Nguyên tắc bất biến (đọc trước khi sửa bất cứ gì)

Mọi thay đổi phải **bảo toàn F1–F5** (xem `docs/data.md §1`):

| Mã | Bất biến |
|---|---|
| F1 | Đồ thị unit-square, `n` đỉnh, `\|A\| = n·d`, `d ∈ {1.5,2,2.5,3}` |
| F2 | ¼-split: mỗi lớp `⌊\|A\|/4⌋`; `P=3`; required ≈ 75%, **3 lớp cân bằng** |
| F3 | `d_a = max{1, ⌊d'_a + 0.5⌋}` (Euclid ×100, integer) |
| F4 | `q_a = ⌊d_a·0.5 + 0.5⌋`; `s_a = 2·d_a` |
| F5 | `Q = (Σ_{a∈A_r} q_a)/3 + 0.5` — **cộng `0.5` MỘT LẦN** (xem cảnh báo dưới) |

> ⚠️ **F5 — code đang SAI:** paper cộng `0.5` một lần `(Σq)/3 + 0.5`, nhưng cả 2 generator đang tính
> `Σ(q/3 + 0.5)` (cộng `0.5` mỗi arc). **Phase 0 §0.5 sửa bắt buộc.**
>
> ⚠️ **F1 + metric — đã chốt làm đúng (B-scoped, Phase 0 §0.6):** generator train hiện dùng
> **complete-Euclid metric** (`adj` = đường thẳng) trong khi test dùng **sparse floyd-warshall** (bám
> đường thật) → lệch bản chất, là **nguyên nhân lớn nhất** khiến test kém. Phase 0 sửa `sample_arcs` thành
> **đồ thị thưa + strongly-connected** và tính `adj` **dùng chung `floyd_warshall` với test**. Planarity &
> integer ×100 là nice-to-have. Train trên unit-square F1, test OOD trên OSM/cluster (Phase 4).

**Ngoài phạm vi (out-of-scope, sẽ đổi công thức — KHÔNG làm):** quét `p` lớp, tách demand/service
khỏi length, đổi dải demand, đổi hằng `/3` của capacity. Xem `docs/data.md §5 ❌`.
(Lưu ý: sửa `Σ(q/3+0.5)` → `Σq/3+0.5` **không phải** đổi công thức — nó là **sửa lỗi** để khớp paper.)

## Hiện trạng repo so với gốc (cái cần sửa)

| File | Hàm | Lệch gốc | Phải về |
|---|---|---|---|
| `env/generator.py` | `sample_priority_classes` | random 1–3 + rule "60–70 cố định" | ¼-split + 3 lớp cân bằng |
| `env/generator.py` | `sample_vehicle_capacity` | **`Σ(q/3+0.5)` (SAI)** | **`Σq/3 + 0.5`** (Phase 0 §0.5) |
| `env/generator.py` | `sample_arcs` | random, không Hamiltonian/SC | **đồ thị thưa + strongly-connected** (Phase 0 §0.6) |
| `env/generator.py` | `sample_traversal_time` | **`adj` = complete-Euclid** (≠ test) | **floyd-warshall thưa, dùng chung test** (§0.6) |
| `env/generator.py` | `sample_*` | giá trị liên tục `/d_max` | giữ `/d_max` (input-preproc); integer là nice-to-have |
| `env/generator.py` | `generate` | nhận 1 `num_loc/num_arc/num_vehicle` | nhận **dải** |
| `data/gen.py` | `required_count` | 60–70 / 75% | ¼-split (F2) |
| `data/gen.py` | `build_instance` clss | `rng.randint(1,4)` random | 3 lớp cân bằng |
| `data/gen.py` | `build_instance` C | **`Σ(q/3+0.5)` (SAI)** | **`Σq/3 + 0.5`** (Phase 0 §0.5) |
| `data/gen.py` | `main` per-M loop | sinh riêng `data/<M>m/` mỗi M | **M là tham số EVAL** — sinh arcs 1 lần, bỏ tầng `<M>m` (Phase 3 revised) |
| `common/ops.py` | `import_instance` | ép đọc `es['M']` | thêm **override `M`** lúc eval (Phase 3 revised) |
| `data/gen.py` | `np.savez` | thiếu metadata | thêm `d, M(nominal), τ, \|A_r\|, topology` |
| `train.sh` | const | `num_arc=20, num_vehicle=3` | nhất quán F1–F5; **size ladder** (Phase 6) |

## Ràng buộc compute (quyết định trần)

- Attention dense `O(n²)`, `attn = q·kᵀ * adj` → **không dùng flash/SDPA** (`policy/encoder.py`).
- `policy/encoder.py:257` → `assert mask is None, "Mask not yet supported!"` ⇒ **1 batch chỉ 1 size `n`**
  (bucket theo size). `collate_fn = torch.cat(adj (1,n,n))`.
- Trần cứng **`|A_r| ≤ 100` (n ≤ 101)** ⇒ mọi cấu hình fit 1×4090, không cần AMP/checkpointing.

## Các phase (làm tuần tự, có cổng test giữa mỗi phase)

| Phase | File | Nội dung | Phụ thuộc |
|---|---|---|---|
| **Pre** | [`00a_cleanup.md`](00a_cleanup.md) | Dọn test deprecated + data/checkpoint cũ (SAI công thức, vượt trần) | — (chạy đầu tiên) |
| **0** | [`01_phase0_restore_formula.md`](01_phase0_restore_formula.md) | Khôi phục ¼-split + lớp cân bằng + capacity + metric thưa | Pre (nền tảng) |
| **1** | [`02_phase1_size_cap.md`](02_phase1_size_cap.md) | Trần `\|A_r\|≤100`, dải size, bucket | P0 |
| **2** | [`03_phase2_density.md`](03_phase2_density.md) | Quét 4 mức `d`, report theo `d` | P1 |
| **3** | [`04_phase3_fleet.md`](04_phase3_fleet.md) | **M là tham số EVAL** (sinh arcs 1 lần, không per-M) | P1 |
| **4** | [`05_phase4_ood.md`](05_phase4_ood.md) | Test OOD (OSM nhiều city + cluster) | P0–P3 |
| **5** | [`06_phase5_stats.md`](06_phase5_stats.md) | Seeds ≥20–30, Wilcoxon/Friedman, τ, gap-BKS | P0–P4 |
| **6** | [`08_phase6_multisize.md`](08_phase6_multisize.md) | Train đa size (bucketed) cho size-generalization | P1 |

Bảng cổng test gộp: [`07_test_matrix.md`](07_test_matrix.md).

> 🔒 **Để sau (bạn xử lý):** đưa `M` vào **input của policy** (M-conditioning) — hiện policy không thấy M.
> Phase 3 chỉ làm M ở mức data/eval; training giữ M cố định.

## Quy tắc "qua phase" (BẮT BUỘC)

> **Chỉ chuyển sang phase kế tiếp khi TOÀN BỘ cổng test của phase hiện tại PASS.**

1. Chạy đủ test suite:
   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```
2. Phải **cập nhật cả test cũ** (các regression test đang assert "60–70" sẽ **fail có chủ đích** sau P0 —
   sửa chúng thành assert ¼-split, đừng chỉ thêm test mới).
3. Mỗi phase có checklist riêng ở cuối file — tick hết mới qua.
4. Không gộp 2 phase trong 1 lần sửa. Mỗi phase = 1 commit (hoặc 1 nhóm commit) độc lập, test xanh.
5. **Bắt đầu bằng Pre-Phase** ([`00a_cleanup.md`](00a_cleanup.md)): tạo tag `archive/pre-data-overhaul`
   rồi xóa data/checkpoint cũ (sinh từ công thức SAI + vượt trần). Không xóa khi chưa có tag khôi phục.
