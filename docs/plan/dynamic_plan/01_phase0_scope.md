# Phase 0 — Chốt ngữ nghĩa `M` (quyết định, code sau)

> Mục tiêu: trả lời dứt khoát **"M nghĩa là gì"** trước khi sửa env/policy. Sai ở đây thì Phase 1–3 vô nghĩa.
> Không có code; output là **quyết định ghi vào plan + comment**. Tham chiếu `dynamic.md §2.2`, `00_overview`.

## 0.1 — 3 câu hỏi phải chốt

### Q1. M là ràng buộc cứng hay chỉ report?
- `cal_reward` hiện **không cap** số route → policy được dùng route tùy ý (chỉ bị capacity chặn dưới ~3).
- Vì makespan **giảm khi nhiều route song song hơn**, nếu không cap M thì policy luôn muốn tách tối đa →
  bài toán dễ đi & **không phải HDCARP đúng**.
- **Khuyến nghị:** M là **ràng buộc cứng** = "tối đa M route song song". ⇒ phải cap (Phase 1).

### Q2. Dải M khả thi? (vì capacity chặn ≥3 route)
`Σ(demand) ≈ 3`, cap route = 1 ⇒ cần **≥3 route**. Chọn 1 trong 3:
- **(A) Dải `M ∈ {3,5,7,10}`** (bỏ 1,2). Đơn giản nhất, đúng capacity hiện tại. **Khuyến nghị.**
- **(B) Multi-trip:** 1 xe về depot nạp lại → chạy nhiều chuyến. M=1,2 khả thi nhưng **đổi mô hình
  makespan** (chuyến tuần tự trên 1 xe) + sửa `cal_reward` nặng. Cân nhắc kỹ.
- **(C) Scale `C` theo M** (`C=Σq/M+0.5`) → **đổi F5 = out-of-scope** (data.md). KHÔNG khuyến nghị.

> Lưu ý: data đã sinh với M nominal=3, instance **M-independent**. Đổi dải M chỉ là tham số eval/train,
> KHÔNG sinh lại data — TRỪ KHI chọn (C) (đổi C = đổi instance = phải sinh lại).

### Q3. "M route" map vào action thế nào?
- Action là 1 chuỗi, `0` = về depot, đoạn giữa 2 số `0` = 1 route (`cal_reward.action_to_tours`).
- Cap M = **chặn mở route thứ M+1** (số lần rời depot tới customer ≤ M).
- Chốt: makespan tính trên **đúng ≤ M tour song song**.

## 0.2 — Quyết định mặc định (đề xuất)

| Câu | Chốt mặc định |
|---|---|
| Q1 | M là **ràng buộc cứng** (cap ≤ M route song song) |
| Q2 | **(A)** dải `M ∈ {3,5,7,10}` |
| Q3 | route = đoạn depot-to-depot; cap = chặn route thứ M+1 |

> Nếu cần báo cáo `M∈{2,5}` như HRDA cũ: M=2 infeasible với C hiện tại → buộc chọn (B) multi-trip hoặc
> chấp nhận chỉ báo cáo M≥3. **Ghi rõ trong paper.**

---

## ✅ Cổng Phase 0 (quyết định, không phải test code)

- [ ] Ghi quyết định Q1/Q2/Q3 vào file này + 1 dòng trong `cal_reward.py`/`env.py` (comment).
- [ ] Xác nhận dải M chốt **không kéo theo sinh lại data** (nếu chọn A/B); nếu (C) → quay lại data_plan.
- [ ] Cập nhật `eval/stats.py`/báo cáo: chỉ tổng hợp M trong dải khả thi.
- [ ] (không có unittest cho phase này — nhưng Phase 1 sẽ test feasibility theo quyết định ở đây.)

> Sau khi chốt → sang [`02_phase1_M_constraint.md`](02_phase1_M_constraint.md).
