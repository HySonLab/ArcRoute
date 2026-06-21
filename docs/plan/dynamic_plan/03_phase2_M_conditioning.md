# Phase 2 — Policy "thấy" `M` (M-conditioning)

> Mục tiêu: đưa `M` (và số route còn lại) thành **input của policy** để model **chủ động** lập lịch theo số
> xe, thay vì chỉ bị env chặn thụ động (Phase 1). Phụ thuộc Phase 1. Code: `env/env.py`, `policy/context.py`.

## 2.1 — Truyền `M` vào `td_reset` (mắt xích còn thiếu)

**File:** `env/env.py:reset`

⚠️ Hiện `reset` **KHÔNG đưa `num_vehicle` vào `td_reset`** (chỉ giữ demand/clss/service/traversal/adj/
visited/used_capacity/...). `generate()` có tạo `num_vehicle` nhưng `reset` bỏ. ⇒ Policy không có đường
nào thấy M.

Cần: thêm vào `td_reset`:
- `"num_vehicle"`: M (broadcast theo batch).
- `"routes_left"`: `M - routes_used` (cập nhật mỗi step trong `step`) — tín hiệu **động** hữu ích hơn M tĩnh.

## 2.2 — Thêm M vào context của policy

**File:** `policy/context.py:ARPContext`

Hiện `project_context = Linear(embed_dim + 1, embed_dim)`, với `+1` = `state_embedding`
(`vehicle_capacity − used_capacity`). Mở rộng:
- Nối thêm **`routes_left`** (và/hoặc `M` chuẩn hóa) vào context → `Linear(embed_dim + 2, embed_dim)`
  (hoặc +3 nếu thêm M).
- Chuẩn hóa M (vd `M/10`) để scale ổn định.

```python
# phác (đọc lại code quanh đó):
state = td["vehicle_capacity"] - td["used_capacity"]          # (B,1)
routes_left = td["routes_left"] / td["num_vehicle"]           # (B,1) tỉ lệ còn lại
context = torch.cat([cur_node_embedding, state, routes_left], -1)
return self.project_context(context)   # Linear(embed_dim + 2, embed_dim)
```

> Lựa chọn thiết kế: đưa M ở **context (động, mỗi bước)** tốt hơn ở init-embedding (tĩnh), vì quyết định
> "có nên mở route mới không" phụ thuộc **số route còn lại tại thời điểm decode**.

## 2.3 — (Không bắt buộc) M ở init-embedding

Có thể thêm M như global feature ở `policy/init.py` (broadcast vào mọi node) nếu muốn encoder cũng biết M.
Thường **không cần** — context đã đủ. Để lại như nice-to-have.

---

## ✅ Cổng test Phase 2

1. **`td_reset` có M:** sau `env.reset`, `td["num_vehicle"]` và `td["routes_left"]` tồn tại, đúng giá trị.
2. **`routes_left` giảm dần:** qua các step, `routes_left` giảm khi mở route mới, ≥ 0.
3. **Context ăn M (shape):** `ARPContext.forward` chạy với chiều mới; output `(B, embed_dim)` đúng.
4. **⭐ Output ĐỔI theo M:** cùng instance + cùng seed, chạy policy với M=3 vs M=7 → phân phối action
   (logits) **khác nhau** (chứng minh M thực sự ảnh hưởng, không bị bỏ qua).
5. **Gradient chảy tới context-M:** `loss.backward` → tham số `project_context` có grad.
6. **⭐ Rollout smoke:** reset→policy→reward→backward ở M=3 và M=7, reward finite.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `env.reset` truyền `num_vehicle` + `routes_left` vào `td_reset`; `step` cập nhật `routes_left`.
- [ ] `ARPContext` nối M/`routes_left`; `Linear` đổi in-dim tương ứng.
- [ ] Test: M trong td, routes_left giảm, context shape, **output đổi theo M**, gradient, rollout smoke — xanh.
- [ ] `unittest discover` xanh (không vỡ test cũ).
- [ ] Commit "Dynamic Phase 2: policy M-conditioning (reset truyền M + context feature)".

> Sau phase này M đã là input thật. Phase 3 train để model **dùng tốt** M tùy ý.
