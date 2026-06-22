# Phase 3 — Train **M-agnostic** ("train once, any M")

> Mục tiêu: train **một** policy M-agnostic chạy tốt với **mọi M** — bằng cách quét M trong **reward
> (Scheduler)**, KHÔNG đưa M vào policy. Policy học **thứ tự arc tốt trung bình qua dải M**; Scheduler `Φ`
> map ra solution cho từng M. Phụ thuộc **Phase 1 (Scheduler)** — KHÔNG phụ thuộc Phase 2 (đã hoãn). Code:
> `train.py`, `train.sh`, `env/generator.py` (**sửa pick M per-instance**).

> **Vì sao quét M trong reward dù policy M-agnostic:** mỗi instance sample 1 M → reward = `Scheduler(α, M)` →
> policy tối ưu `E_M[Φ(α,M)]` ⇒ học **thứ tự robust cho cả dải M** (chính là "train once"). M **không** vào
> input policy, nhưng phải đa dạng trong reward để thứ tự không overfit một M.

## 3.1 — Cho reward quét M (per-instance)

**File:** `train.py`
- `--num_vehicle` hiện `type=int` (1 giá trị). Cho nhận **list rời rạc** giống `--sizes`:
  `--num_vehicle "3,5,7,10"` → parse thành `[3,5,7,10]`, truyền vào `CARPEnv`.

**File:** `env/generator.py` — ⚠️ **PHẢI SỬA (plan cũ nói sai "không sửa generator")**
- Hiện `generate_dataset` (dòng ~196) gọi `_pick(num_vehicle)` **MỘT lần cho cả dataset** → mọi instance
  cùng 1 M; trong `MultiSizeCARPGenerator` (dòng ~272) gọi 1 lần/size ⇒ **mỗi size 1 M cố định, KHÔNG trộn**.
- **Sửa độ hạt:** giữ `num_loc/num_arc` pick một lần (ràng buộc shape bucket), **để `num_vehicle` pick
  PER-INSTANCE**:
  ```python
  num_loc, num_arc = _pick(num_loc), _pick(num_arc)        # bỏ _pick(num_vehicle) ở đây
  def __getitem__(self, idx):
      return generate(num_loc, num_arc, num_vehicle)       # truyền nguyên LIST → generate:141 _pick mỗi instance
  ```
- **`num_vehicle` lưu dạng tensor `(B,1)`** (hiện là python int, dòng ~185) để `torch.cat` collate + slice
  `td[i]` trong `run_parallel` chạy đúng khi M khác nhau giữa các hàng.
- **Scheduler (`calc_reward`) đọc M từ `td["num_vehicle"]` (per-instance)**, KHÔNG từ `env.self.num_vehicle`
  (scalar) — nếu không cả batch sụp về một M.

> Vì reward chạy **per-instance** (`run_parallel` cắt `td[i]`) và M **không vào policy/mask/decode**
> (M-agnostic), nên batch trộn M **không phá `torch.cat`**; mỗi instance được Scheduler tính theo M riêng.
> (Contrast: size đổi shape → vẫn phải bucket.)

**File:** `train.sh`
- Thêm biến `FLEET="3,5,7,10"` (dải Phase 0-A) + truyền `--num_vehicle "$FLEET"`.
- Giữ size ladder (`--sizes`) của data_plan Phase 6 → train **đa size × đa M** cùng lúc.
- **Variant mặc định `P`** (Phase 0-Q4): `train.py` đang default `U` → **đổi sang `P`** (hoặc truyền
  `--variant P` trong `train.sh`). Train mỗi variant 1 model; plan chốt báo cáo chính trên `P`.

> Lưu ý bucketing: size vẫn bucket (1 size/batch). M **không** cần bucket (M là scalar, không đổi shape) →
> trộn M tự do trong 1 batch. ✓

## 3.2 — Cân nhắc huấn luyện

- **Curriculum (tùy chọn):** M nhỏ nhất (nhiều chuyến nối tiếp nhất → makespan cao nhất, khó cân nhất) là
  ca khó; có thể bắt đầu M lớn rồi siết dần — hoặc random đều ngay. Random đều thường đủ; ghi chú nếu dùng.
- **Baseline so sánh:** 1 model **M-agnostic** (train once) vs vài model **mỗi M một cái** (hoặc M-conditioned
  Phase 2) → chứng minh M-agnostic không thua nhiều. Đây là **ablation** chính của headline.

## 3.3 — Lưu ý train≈test & feasibility

- Mọi M feasible (theo calib đã chốt Phase 0) → không gặp instance vô nghiệm, không deadlock khi train.
- `Q=Σq/3` khít + Scheduler multitrip: `>M` chuyến **không phải lỗi**, chỉ là multi-trip (M=2 → 3 trip/2 xe).
- `FLEET` nên gồm **M=2** (báo cáo so HRDA) — chỉ là tham số reward, không đổi shape.

---

## ✅ Cổng test Phase 3

1. **Parse `--num_vehicle` list:** `"3,5,7,10"` → `[3,5,7,10]`; 1 giá trị `"3"` vẫn chạy (backward-compat).
2. **⭐ Pick M per-instance:** sau khi sửa generator, **MỘT batch (cùng size)** có `td["num_vehicle"]` chứa
   **≥2 giá trị M khác nhau** (chứng minh đã pick per-instance, không còn 1 M/bucket). `num_vehicle` shape
   `(B,1)`.
3. **Batch trộn M không vỡ:** batch mixed-M (cùng size) → encoder+decoder forward + `run_parallel(calc_reward)`
   chạy, **`torch.cat` không lỗi**, mỗi instance lập lịch theo M riêng (kiểm 1 ca M=3 và 1 ca M=7 trong cùng
   batch cho makespan khác nhau hợp lý).
4. **⭐ Train step M-agnostic × đa size:** dataloader (sizes + fleet) → vài step PPO thật (encoder+decoder+
   Scheduler-reward+backward), gradient OK, reward finite. **Policy KHÔNG nhận M** (kiểm: bỏ M khỏi input
   vẫn chạy). **Chạy ở `variant='P'`.**
5. **Scheduler chạy khi train:** reward mỗi batch hữu hạn ở mọi M; không deadlock.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
# smoke train ngắn (vài epoch) để chắc pipeline:
#   sửa train.sh MAX_EPOCH nhỏ, FLEET="3,5,7,10", rồi ./train.sh ; theo dõi logs/
```

### Checklist
- [ ] `train.py --num_vehicle` nhận list; `train.sh` thêm `FLEET` + truyền vào.
- [ ] **`env/generator.py`: pick `num_vehicle` per-instance** (bỏ `_pick` ở `generate_dataset`, để `generate`
      tự pick); `num_vehicle` lưu tensor `(B,1)`; reward/context đọc M từ `td` per-instance.
- [ ] `train.py`/`train.sh` đặt **`variant='P'`**; **policy M-agnostic** (không nối M vào input).
- [ ] Test: parse list, M per-instance, batch trộn M, **train-step M-agnostic×size**, reward hữu hạn — xanh.
- [ ] (tùy chọn) chạy smoke train ngắn, log vào `logs/`.
- [ ] `unittest discover` xanh.
- [ ] Commit "Dynamic Phase 3: train M-agnostic (train once, any M)".
