# Bảng cổng test gộp — dynamic_plan

> Mỗi phase chỉ qua khi **toàn bộ** cổng test của nó + các phase trước **xanh**.
> ```bash
> uv run python -m unittest discover -s tests -p "test_*.py" -v
> ```

## Lưu ý xuyên suốt

- **Mỗi phase phải kèm "rollout smoke"** (`env.reset → policy → reward → backward`, reward finite) ở các M
  liên quan — vì sửa env/reward/policy dễ **vỡ rollout** hoặc **lệch train≈test**.
- **Không phá test data_plan cũ** (65 test hiện có phải vẫn xanh).
- File test mới gợi ý: `tests/test_env_fleet.py` (Phase 1–2), mở rộng `tests/test_multisize.py` (Phase 3),
  `tests/test_encoder_mask.py` (Phase 4).
- `p=3` cố định; trần `|A_r|≤100`; metric thưa (floyd-warshall) **không đổi**.

## Ma trận

| Phase | Bất biến phải giữ | Test trọng yếu (PASS mới qua) |
|---|---|---|
| **0** Scope | — (quyết định) | Ghi quyết định Q1/Q2/Q3; xác nhận dải M không kéo theo sinh lại data |
| **1** M-constraint | capacity, makespan song song | (1) số tour ≤ M; (2) M lớn → nới route; (3) feasibility M=3; (4) reward-guard >M tour; (5) no-deadlock mask; (6) ⭐ rollout smoke M=3 & M=7 |
| **2** M-conditioning | train≈test không vỡ | (1) `td_reset` có `num_vehicle`+`routes_left`; (2) routes_left giảm ≥0; (3) context shape mới; (4) ⭐ **output đổi theo M**; (5) gradient tới context-M; (6) rollout smoke |
| **3** Train đa M | bucketing size giữ; M trộn tự do | (1) parse `--num_vehicle` list; (2) generator sweep M; (3) batch trộn M cùng size OK; (4) ⭐ train-step đa M×size (backward); (5) cap tôn trọng khi train |
| **4** Dynamic size *(tùy chọn)* | không rò padding | (1) ragged == bucket (≤1e-4); (2) đổi padding → token thật không đổi; (3) không NaN; (4) masked-norm đúng; (5) rollout ragged |
| **5** Eval | dùng test grid sẵn có | (1) cùng instance đa M → makespan đơn điệu theo M; (2) break-down theo size/d/topology; (3) stats p∈[0,1], gap đúng; (4) report gap in-dist vs OOD |

## Quan hệ với data_plan

- **Không sinh data mới.** Dùng `data/ood/` + `import_instance(M=)` + `eval/stats.py`.
- Nếu Phase 0 chọn (C) scale `C` theo M → **quay lại data_plan** (đổi F5, sinh lại) — tránh nếu được.

## Checklist tổng

- [ ] Phase 0: chốt ngữ nghĩa M (Q1/Q2/Q3) → ghi vào plan + comment code
- [ ] Phase 1: M là ràng buộc thật (env cap + reward) → commit
- [ ] Phase 2: policy M-conditioning (reset + context) → commit
- [ ] Phase 3: train đa M → commit
- [ ] Phase 4 *(tùy chọn)*: dynamic size masking/ragged → commit
- [ ] Phase 5: eval & report theo M/size → commit
- [ ] Cập nhật `docs/dynamic.md` nếu quyết định khác plan (giữ dynamic.md là nguồn sự thật về mục tiêu).
