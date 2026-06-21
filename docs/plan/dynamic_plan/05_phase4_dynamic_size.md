# Phase 4 — Size động THẬT (mask encoder + ragged) — ⏹ TÙY CHỌN

> Mục tiêu: cho **trộn nhiều size trong 1 batch** (ragged) thay vì bucket. Đây là **sửa kiến trúc, khó
> nhất**, và **KHÔNG bắt buộc** — đa size đã chạy qua bucketing (data_plan Phase 6). Chỉ làm nếu cần
> throughput cao hơn hoặc infer batch ragged. Tham chiếu `dynamic.md §2.1`.

## 4.0 — Cân nhắc trước khi làm

| | Bucketing (đã có) | Ragged + mask (Phase 4) |
|---|---|---|
| Đa size train | ✅ | ✅ |
| Trộn size/batch | ❌ (1 size/batch) | ✅ |
| Công sức | 0 (xong) | **cao** (mask khắp encoder/norm/decoder) |
| Rủi ro rò padding | 0 | cao nếu mask sót |

> **Khuyến nghị:** bỏ qua Phase 4 trừ khi đo được bucketing là nút cổ chai. Nếu làm, theo đúng 5 bước dưới.

## 4.1 — Bỏ `assert mask is None` + truyền mask qua MHA

**File:** `policy/encoder.py`
- `GraphAttentionNetwork.forward` (dòng ~257): `assert mask is None` → nhận `mask` (padding) và truyền
  xuống từng `MultiHeadAttentionLayer`.

## 4.2 — Mask TRƯỚC softmax (quan trọng nhất)

**File:** `policy/encoder.py:MultiHeadAttention.forward` (dòng ~166–170)
- Hiện: `attn = q·kᵀ * adj` rồi `clamp` rồi `softmax`. Padding hiện chỉ bị `*adj` (adj=0) → logit=0 nhưng
  **softmax(0) vẫn rò** sang token padding.
- Sửa: trước softmax, `attn = attn.masked_fill(pad_mask, -inf)` (vị trí key padding → −∞). Đảm bảo hàng
  toàn −∞ (token padding làm query) không tạo NaN (xử lý riêng: zero-out output các hàng đó).

## 4.3 — Ragged / padding collate

**File:** `env/generator.py` (collate) + `env/env.py`
- `collate_fn` hiện `torch.cat` (yêu cầu cùng `n`). Thêm bản **pad** `adj/features/mask` về `n_max` của
  batch; trả kèm `pad_mask`.
- `SizeBucketBatchSampler` → đổi sang sampler **trộn size** khi bật ragged.

## 4.4 — Chuẩn hóa loại padding (masked-norm)

**File:** `policy/encoder.py:Normalization` (BatchNorm1d / InstanceNorm1d, dòng ~22)
- BatchNorm tính mean/var **gồm cả token padding** → lệch. Đổi sang **LayerNorm** hoặc **masked-norm**
  (chỉ thống kê trên token thật).

## 4.5 — Decoder + action_mask tôn trọng padding

**File:** `policy/decoder.py`, `env/env.py:get_action_mask`
- Bảo đảm bước decode (pointer) và `action_mask` **không chọn token padding**; logit padding → −∞.

---

## ✅ Cổng test Phase 4 (nếu làm)

1. **Mask = bucket (tương đương):** 1 batch ragged (vài size) cho **cùng kết quả** với chạy từng size riêng
   (cùng seed/trọng số) — chứng minh padding không ảnh hưởng số học. (sai số ≤ 1e-4)
2. **Không rò padding:** đổi giá trị ở vùng padding → output token thật **không đổi**.
3. **Không NaN:** hàng query toàn-padding không tạo NaN trong softmax/norm.
4. **Masked-norm đúng:** thống kê norm không phụ thuộc số token padding (thêm padding → output token thật giữ nguyên).
5. **⭐ Rollout ragged:** reset→policy→reward→backward trên batch trộn size, reward finite, gradient OK.

### Checklist
- [ ] Encoder nhận + truyền mask; `masked_fill(-inf)` trước softmax; xử lý hàng toàn-padding.
- [ ] Padding collate + sampler trộn size.
- [ ] LayerNorm/masked-norm thay BatchNorm.
- [ ] Decoder/action_mask loại padding.
- [ ] Test tương-đương-bucket, không-rò, không-NaN, masked-norm, rollout ragged — xanh.
- [ ] `unittest discover` xanh.
- [ ] Commit "Dynamic Phase 4 (optional): true dynamic size via masking + ragged batch".

> Nếu bỏ Phase 4: ghi rõ trong paper rằng đa size đạt qua **bucketing** (1 size/batch) — vẫn hợp lệ.
