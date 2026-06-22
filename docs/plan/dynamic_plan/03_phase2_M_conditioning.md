# Phase 2 — ⏸ *(HOÃN)* M-conditioning (policy "thấy" `M`) — enhancement

> **TRẠNG THÁI: HOÃN — không nằm trên critical path.** Headline dùng **policy M-agnostic** (Scheduler lo M,
> Phase 1). Phase này là **enhancement làm SAU**: cho policy *thấy* M để **tỉa chuỗi arc theo M** (thay vì
> để Scheduler tự xoay), kỳ vọng chất lượng mỗi-M tốt hơn. **Giữ trong plan để implement về sau**, kèm
> **ablation** chứng minh giá trị gia tăng so với M-agnostic. Phụ thuộc Phase 1 + 3. Code: `policy/context.py`,
> critic.

## Vì sao hoãn (không bỏ)

- M-agnostic đã cho **1 model chạy mọi M** ("train once") — đủ cho headline.
- M-conditioning **có thể** tốt hơn vì policy tỉa thứ tự/độ-lớn chuyến theo M, nhưng:
  - thêm chi phí: train phải **quét M trong input**, critic cũng phải M-aware;
  - giá trị **chưa chắc lớn** nếu Scheduler đã tốt → cần **đo bằng ablation** mới biết đáng làm.
- ⇒ Làm sau khi có baseline M-agnostic (Phase 3) để **so sánh sòng phẳng**.

## 2.1 — Khi implement: truyền M vào policy

**File:** `env/env.py:reset` — `num_vehicle` đã có trong `td` (Phase 1). Tùy chọn thêm `trips_opened` (số
chuyến đã mở) làm soft-signal động.

## 2.2 — Context của policy

**File:** `policy/context.py:ARPContext` — hiện `Linear(embed_dim + 1, embed_dim)` (`+1` = `cap − used_cap`).
Nối thêm **`M` chuẩn hóa** (vd `M/10`) → `Linear(embed_dim + 2, embed_dim)`:

```python
state  = td["vehicle_capacity"] - td["used_capacity"]   # (B,1)
m_feat = td["num_vehicle"] / 10.0                        # (B,1)
context = torch.cat([cur_node_embedding, state, m_feat], -1)
return self.project_context(context)                     # Linear(embed_dim + 2 → embed_dim)
```

## 2.3 — ⚠️ Critic cũng phải thấy M

Reward phụ thuộc M (qua Scheduler) ⇒ value baseline `V_φ` phải **M-conditioned**, nếu không advantage
`reward − value` bị nhiễu (baseline trung bình hóa qua M). **Nối M vào critic** y như context.

## 2.4 — (Không bắt buộc) M ở init-embedding

Thêm M như global feature ở `policy/init.py` nếu muốn encoder cũng biết M. Thường không cần.

---

## ✅ Cổng test Phase 2 (khi implement)

1. **`td_reset` có M** (đã từ Phase 1); (nếu làm) `trips_opened` tăng ≥0.
2. **Context + critic ăn M (shape):** forward chạy với chiều mới, output đúng.
3. **⭐ Output ĐỔI theo M:** cùng instance/seed, M=3 vs M=7 → logits khác (ở variant P).
4. **Gradient tới context-M + critic-M.**
5. **⭐ Rollout smoke** ở `M∈{2,3,7}` × `{P,U}`, reward finite.
6. **⭐ Ablation:** model M-conditioned vs M-agnostic trên cùng test grid → báo cáo chênh lệch (giá trị của
   enhancement). Nếu không thắng đáng kể → ghi nhận M-agnostic là đủ.

### Checklist (khi implement)
- [ ] `ARPContext` + **critic** nối `M` chuẩn hóa; `Linear` đổi in-dim.
- [ ] Train quét M trong input; test output-đổi-theo-M, gradient, rollout smoke `{P,U}`.
- [ ] **Ablation M-cond vs M-agnostic** → bảng so sánh.
- [ ] `unittest discover` xanh; commit "Dynamic Phase 2 (enhancement): M-conditioning + ablation".

> Ghi chú: đây là **TODO hoãn lại** — critical path headline là `0 → 1 → 3 → 5` (M-agnostic).
