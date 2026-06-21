# Phase 1 — `M` thành ràng buộc THẬT (env + reward)

> Mục tiêu: làm `M` thực sự **giới hạn số route song song** (hiện no-op). Đây là nền tảng — phải xong
> trước khi cho policy "thấy" M (Phase 2). Phụ thuộc Phase 0 (đã chốt cap ≤ M, dải `{3,5,7,10}`).
> Tham chiếu `dynamic.md §2.2`; code: `env/env.py`, `common/cal_reward.py`.

## 1.1 — Đếm & cap số route trong env

**File:** `env/env.py`

Hiện `step`/`reset` theo dõi `used_capacity`, `visited` nhưng **không đếm route**. `step` reset
`used_capacity` về 0 khi về depot (`* (current_node != 0)`), nhưng số lần mở route **không bị chặn**.

Cần:
1. Thêm state **`routes_used`** vào td (reset = 0). Tăng 1 mỗi khi **rời depot tới customer** (transition
   `current_node==0 → action!=0`).
2. Trong `get_action_mask`: khi `routes_used == M` và xe **đang ở depot**, **cấm đi tới customer mới**
   (chỉ còn được phục vụ trong route hiện tại). Khi route hiện tại cũng hết chỗ → còn arc chưa phục vụ =
   **infeasible** (xem 1.3).
3. `M` lấy từ td (`num_vehicle`) — Phase 2 sẽ truyền vào `td_reset`; tạm thời đọc `self.num_vehicle`.

> ⚠️ Tinh tế: "mở route mới" = rời depot. Phải phân biệt depot-return (kết thúc route) với depot-departure
> (mở route). Cap đếm trên **departure**.

## 1.2 — Reward tính trên đúng ≤ M tour

**File:** `common/cal_reward.py`

`calc_reward` + `action_to_tours` hiện tách tour theo số `0` — **bao nhiêu cũng nhận**. Sau khi env cap M,
action hợp lệ sẽ có ≤ M tour, nên `cal_reward` **tự nhiên đúng** NẾU env chặn chuẩn. Nhưng cần:
1. **Assert/guard:** số tour ≤ M; nếu >M (do mask lỗi) → raise hoặc phạt nặng (bắt bug sớm).
2. Makespan song song giữ nguyên (max theo lớp qua các tour) — không đổi công thức.

## 1.3 — Xử lý infeasible (M quá nhỏ)

Với dải `{3,5,7,10}` (Phase 0-A) thì M≥3 → luôn đủ chở (`Σdemand≈3`). Nhưng để **chắc chắn**:
- Test feasibility: với M=3, tồn tại lời giải hợp lệ (mọi arc phục vụ, mỗi route ≤ cap, ≤ M route).
- Nếu cho M<3 (không khuyến nghị) → env phải báo infeasible rõ ràng, không treo vòng lặp decode.

---

## ✅ Cổng test Phase 1

**File test:** thêm `tests/test_env_fleet.py`.

1. **Cap số route:** chạy rollout với M=3 trên vài instance → **số tour ≤ 3** mọi lần
   (`len(action_to_tours(actions)) <= M`).
2. **M lớn cho phép nhiều route hơn:** M=7 → có instance dùng >3 tour (chứng tỏ cap nới theo M).
3. **Feasibility M=3:** tồn tại rollout hợp lệ phục vụ hết arc, mỗi route ≤ cap.
4. **Reward guard:** ép action có >M tour → `cal_reward` raise/phạt (không trả số "đẹp" sai).
5. **Mask không khóa cứng:** action_mask tại depot khi `routes_used<M` vẫn cho mở route (không deadlock).
6. **⭐ Rollout smoke (chống vỡ):** `env.reset → policy → reward → backward` vẫn chạy, reward finite
   (như data_plan Phase 0). Chạy ở M=3 và M=7.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `env`: state `routes_used`, tăng khi rời depot; `get_action_mask` cap ở M.
- [ ] `cal_reward`: guard ≤ M tour; makespan song song giữ nguyên.
- [ ] Feasibility M=3 OK; infeasible (M<3) báo rõ.
- [ ] Test cap route, nới theo M, feasibility, reward-guard, no-deadlock, **rollout smoke** — xanh.
- [ ] `unittest discover` xanh (gồm test data_plan cũ — không được vỡ).
- [ ] Commit "Dynamic Phase 1: M là ràng buộc thật (cap route + reward theo M)".

> Lưu ý: ở phase này policy **chưa thấy M** (vẫn chưa vào input) — M chỉ ràng buộc qua env/mask. Phase 2 mới
> đưa M vào model để policy **chủ động** dùng đúng số xe.
