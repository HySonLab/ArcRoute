# Plan tổng quan — Động hóa MODEL theo `M` (và `size`)

> Nguồn: [`docs/dynamic.md`](../../dynamic.md). Mục tiêu: biến model RL từ "cố định size & bỏ qua M"
> thành **nhận `M` (số xe) làm tham số điều khiển thật** và (tùy chọn) **size động trong 1 batch**.
>
> ⚠️ **Đây là plan về MODEL** (env + policy + reward + encoder), KHÔNG đụng khâu sinh data — data đã sẵn
> sàng (xem `docs/plan/data_plan/`: size ladder Phase 6, M-independent instance Phase 3). Test dùng lại
> **test grid `data/ood/`** + `eval/stats.py` đã có.

## Bối cảnh: cái gì đã động, cái gì chưa

Theo `dynamic.md §1`: mọi **nội dung số** của instance đã động (demand/service/traversal/adj/clss scalar).
Còn lại **2 trục cấu trúc**:

| Trục | Hiện trạng | Plan |
|---|---|---|
| **`M` (số xe)** | lưu vào td nhưng **no-op** — `cal_reward` bỏ qua, không cap route, policy không thấy M | **Cốt lõi plan** (Phase 1–3) |
| **`size \|A_r\|`** | đa size đã chạy qua **bucketing** (data_plan Phase 6) | Đường "rẻ" XONG; "thật" (trộn size/batch) = Phase 4 **tùy chọn** |
| `p` (số lớp) | hardcode 3 ở reward (`pos_val=[1,2,3]`) | **giữ 3** (theo Hà) — không trong plan |

## ⚠️ Phát hiện then chốt: M bị chặn bởi CAPACITY

`C = Σq/3 + 0.5` → demand chuẩn hóa `Σ(q/C) ≈ 3` (cap mỗi route = 1). ⇒ **cần ≥ 3 route** mới chở hết
demand. Hệ quả khi cap số route ở M:

| M | tổng tải `M·1` vs nhu cầu `≈3` | Khả thi? |
|---|---|---|
| 1, 2 | 1, 2 < 3 | ❌ **INFEASIBLE** (không chở hết) |
| **3** | 3 ≈ 3 | ✅ **khít** (đúng calib paper) |
| 5, 7, 10 | > 3 | ✅ lỏng (makespan giảm nhờ song song hơn) |

→ **Quyết định bắt buộc ở Phase 0:** sweep M thực sự là `{3,5,7,10}` (bỏ 1,2) HOẶC cho **multi-trip**
(1 xe về depot nạp lại, chạy nhiều chuyến) HOẶC scale `C` theo M (= đổi F5, **out-of-scope** theo data.md).
Đây cũng là lời cảnh báo cho `dynamic.md §2.2` ("M có cần là input không").

## Các phase

| Phase | File | Nội dung | Bắt buộc? |
|---|---|---|---|
| **0** | [`01_phase0_scope.md`](01_phase0_scope.md) | Chốt ngữ nghĩa M (cap route? multi-trip? dải M khả thi) | quyết định |
| **1** | [`02_phase1_M_constraint.md`](02_phase1_M_constraint.md) | M thành **ràng buộc thật** (env cap route + reward theo M) | ✅ nền tảng |
| **2** | [`03_phase2_M_conditioning.md`](03_phase2_M_conditioning.md) | Policy **thấy M** (reset truyền M + context feature) | ✅ |
| **3** | [`04_phase3_train_generalize.md`](04_phase3_train_generalize.md) | Train **đa M** để generalize | ✅ |
| **4** | [`05_phase4_dynamic_size.md`](05_phase4_dynamic_size.md) | Size động THẬT (mask encoder + ragged + masked-norm) | ⏹ **tùy chọn** |
| **5** | [`06_phase5_eval.md`](06_phase5_eval.md) | Eval & report theo M, size trên test grid | ✅ |

Bảng cổng test gộp: [`07_test_matrix.md`](07_test_matrix.md).

## Ràng buộc kế thừa (giữ nguyên)

- Trần `|A_r| ≤ 100`, attention `O(n²)`, không flash (data_plan §5.7).
- Metric train≈test (floyd-warshall thưa) — **không phá** khi sửa model.
- `p=3` cố định.

## Quy tắc "qua phase" (như data_plan)

1. Mỗi phase có cổng test riêng — **PASS hết mới qua phase sau**.
2. `uv run python -m unittest discover -s tests -p "test_*.py"` phải xanh.
3. Mỗi phase = 1 commit độc lập.
4. Đổi env/reward/policy dễ làm **lệch train≈test** hoặc **vỡ rollout** → mỗi phase có **smoke test rollout**
   (encoder+decoder+reward+backward) như đã làm ở data_plan Phase 0/6.
