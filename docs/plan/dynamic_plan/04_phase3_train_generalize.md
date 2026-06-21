# Phase 3 — Train đa `M` để generalize

> Mục tiêu: train policy (đã M-conditioned, Phase 2) trên **nhiều M** để 1 model chạy tốt với mọi `M` trong
> dải. Phụ thuộc Phase 2. Code: `train.py`, `train.sh`, `env/generator.py` (đã hỗ trợ list M).

## 3.1 — Cho train sweep M

**File:** `train.py`
- `--num_vehicle` hiện `type=int` (1 giá trị). Cho nhận **list rời rạc** giống `--sizes`:
  `--num_vehicle "3,5,7,10"` → parse thành `[3,5,7,10]`, truyền vào `CARPEnv`.
- Generator đã hỗ trợ `num_vehicle=list` (data_plan Phase 3, `_pick` chọn discrete) → mỗi instance random
  1 M trong dải. **Không sửa generator.**

**File:** `train.sh`
- Thêm biến `FLEET="3,5,7,10"` (dải Phase 0-A) + truyền `--num_vehicle "$FLEET"`.
- Giữ size ladder (`--sizes`) của data_plan Phase 6 → train **đa size × đa M** cùng lúc.

> Lưu ý bucketing: size vẫn bucket (1 size/batch). M **không** cần bucket (M là scalar, không đổi shape) →
> trộn M tự do trong 1 batch. ✓

## 3.2 — Cân nhắc huấn luyện

- **Curriculum (tùy chọn):** bắt đầu M=3 (khít, khó nhất) rồi mở dần — hoặc random đều ngay. Random đều
  thường đủ; ghi chú nếu dùng curriculum.
- **Baseline so sánh:** train 1 model **đa M** vs vài model **mỗi M một cái** → chứng minh model đa M không
  thua nhiều (giá trị của M-conditioning).

## 3.3 — Lưu ý train≈test & infeasible

- M trong dải `{3,5,7,10}` đều feasible (Phase 0) → không gặp instance vô nghiệm lúc train.
- Reward guard (Phase 1) bắt lỗi nếu mask cho ra >M tour giữa lúc train.

---

## ✅ Cổng test Phase 3

1. **Parse `--num_vehicle` list:** `"3,5,7,10"` → `[3,5,7,10]`; 1 giá trị `"3"` vẫn chạy (backward-compat).
2. **Generator sweep M:** với `num_vehicle=[3,5,7,10]`, qua nhiều instance, `td["num_vehicle"]` phủ cả dải.
3. **Batch trộn M, đồng size:** 1 batch (cùng size) có thể chứa **nhiều M khác nhau** → forward không lỗi
   (M là scalar/sample, không phá `torch.cat`).
4. **⭐ Train step đa M×đa size:** dataloader (sizes + fleet) → vài step PPO thật (encoder+decoder+reward+
   backward), gradient OK, reward finite — như smoke data_plan Phase 6.
5. **Cap tôn trọng khi train:** trong các batch, số tour ≤ M tương ứng từng instance.

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
# smoke train ngắn (vài epoch) để chắc pipeline:
#   sửa train.sh MAX_EPOCH nhỏ, FLEET="3,5,7,10", rồi ./train.sh ; theo dõi logs/
```

### Checklist
- [ ] `train.py --num_vehicle` nhận list; `train.sh` thêm `FLEET` + truyền vào.
- [ ] Test: parse list, generator sweep M, batch trộn M, **train-step đa M×size**, cap tôn trọng — xanh.
- [ ] (tùy chọn) chạy smoke train ngắn, log vào `logs/`.
- [ ] `unittest discover` xanh.
- [ ] Commit "Dynamic Phase 3: train đa M (generalize fleet)".
