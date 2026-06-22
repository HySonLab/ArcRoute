# Phase 0 — Ngữ nghĩa `M` (CHỐT: policy M-agnostic + Scheduler tách riêng)

> Mục tiêu: chốt **"M nghĩa là gì"** và **kiến trúc** trước khi sửa code. Tham chiếu `dynamic.md §2.2`,
> [`00_overview`](00_overview.md), và paper `../HDCARP_with_ML/main.tex`.

## 0.1 — Kiến trúc đã chốt: M-agnostic policy + Scheduler

**Tách 2 thành phần** (khớp notation paper: policy dựng chuỗi `α`, toán tử `Φ` map ra solution `S`):

1. **Policy `π_θ`** (encoder/decoder): dựng **một chuỗi arc theo ưu tiên**, **KHÔNG thấy M** → 1 model cho
   mọi M ("train once").
2. **Scheduler `Φ`** (module riêng): `(α, M) → routes của M xe` + mục tiêu phân cấp `(T_1,…,T_p)`. **M chỉ
   vào đây.**

→ Headline đóng góp: **decouple construction (policy, M-agnostic) khỏi assignment/evaluation (Scheduler,
M-aware)**; đổi M = chạy lại `Φ`, không train lại.

**M-conditioning** (policy thấy M) = **enhancement HOÃN** → [Phase 2](03_phase2_M_conditioning.md), làm sau,
có ablation. Giữ trong plan để implement về sau.

## 0.2 — ✅ ĐÃ CHỐT: `Q = Σq/3 + 0.5` (theo paper gốc Ha 2024) + Scheduler **multi-trip**

**Nguồn chuẩn — Ha 2024 (`papers/2024_Ha_HDCARP_matheuristic.pdf`, tr.18):**
> *"There are **three** homogeneous vehicles with capacity `Q = Σ_{a∈A_r} q_a/3 + 0.5`."*

⇒ `Q = (Σq_a)/3 + 0.5` — **+0.5 MỘT lần**; số "3" = **số xe**. Đây là calib **KHÍT** (đo thực tế: Σdemand
chuẩn hoá = **2.87**, #route tối thiểu = **3**). **Code `generator.py:102` đang ĐÚNG** → **không đổi code**.

> ⚠️ **TYPO trong bản nháp** `HDCARP_with_ML/main.tex:632` viết `Σ_a(q_a/3 + 0.5)` (per-arc → hoá "lỏng").
> **Phải sửa nháp về `Σq/3 + 0.5`** (một lần) cho khớp paper gốc + code.

**Hệ quả: Q khít → pure-evaluator KHÔNG làm được M=2** (2 route × Q < Σq). ⇒ **Scheduler chạy chế độ
multi-trip**, **tự suy biến về single-trip-evaluator khi K≤M** (tức M≥3):

| M | #trip tối thiểu (Q khít) | Scheduler | = evaluator? |
|---|---|---|---|
| **2** | 3 | gán 3 trip → 2 xe (1 xe 2 trip) | ❌ multi-trip |
| **3** | 3 | 3 trip → 3 xe (1-1) | ✅ |
| **5,7,10** | 3 | chia order → M segment → M xe | ✅ |

> **`Q` CỐ ĐỊNH (`/3`, không theo M) ⇒ policy rollout độc lập M ⇒ rollout một lần, Scheduler chia/gán cho
> mọi M, KHÔNG rollout lại** — *"train once, M là eval-time param"* ở dạng sạch nhất. M=2 = gán 3 trip lên 2 xe.

## 0.3 — Câu hỏi phụ đã chốt

| Câu | Chốt |
|---|---|
| M là ràng buộc cứng hay report? | **Cứng**, thực thi trong **Scheduler** (không cap mask của policy) |
| Map action → solution | chuỗi α → `Φ` → routes của M xe; `T_k` = max completion time mỗi lớp (paper) |
| Variant | mặc định **P**; test lặp **U**. P/U ảnh hưởng **Scheduler** (precedence vs hierarchy-level) |
| `p` (số lớp) | giữ **3** |

## 0.4 — Xoá giả định cũ (deprecated)

- ❌ Bỏ ý tưởng cap số route bằng mask + `routes_used` (Hướng A mask-cap — không code).
- ✅ Giữ "cần ~3 route" (do `Q = Σq/3` khít, **đúng paper gốc**) — đây là lý do Scheduler phải multi-trip cho M=2.

---

## ✅ Cổng Phase 0 (quyết định)

- [x] Chốt **M-agnostic policy + Scheduler tách module**; M-conditioning hoãn (Phase 2).
- [x] **Chốt calib `Q = Σq/3 + 0.5`** (paper gốc Ha 2024 — **không đổi code**); Scheduler **mode multi-trip**.
- [ ] **Sửa TYPO bản nháp** `HDCARP_with_ML/main.tex:632`: `Σ_a(q_a/3+0.5)` → `Σq/3 + 0.5`.
- [ ] Ghi 1 dòng comment ở `common/scheduler.py` ("M-agnostic policy + multi-trip Φ; Q=Σq/3 cố định").
- [ ] Xác nhận **không sinh lại data, không đổi `Q` trong code** (code đã đúng).

> Sau khi chốt → sang [`02_phase1_M_constraint.md`](02_phase1_M_constraint.md).
