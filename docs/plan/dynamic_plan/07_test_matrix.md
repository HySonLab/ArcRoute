# Bảng cổng test gộp — dynamic_plan

> Mỗi phase chỉ qua khi **toàn bộ** cổng test của nó + các phase trước **xanh**.
> ```bash
> uv run python -m unittest discover -s tests -p "test_*.py" -v
> ```

## Lưu ý xuyên suốt

- **Kiến trúc headline: policy M-agnostic + Scheduler `Φ` tách module** (`common/scheduler.py`). M **chỉ vào
  Scheduler**, KHÔNG vào encoder/decoder/mask → rollout không phụ thuộc M, không deadlock. Critical path
  `0→1→3→5`; **Phase 2 (M-conditioning) HOÃN**; Phase 4 (size) tùy chọn.
- **Calib `Q = Σq/3 + 0.5` (ĐÃ CHỐT — paper gốc Ha 2024, code đúng, không đổi).** Khít ⇒ Scheduler **multi-trip**
  (tự thành single-trip khi K≤M, tức M≥3). M=2 = gán ~3 trip lên 2 xe. *(Bản nháp main.tex per-arc = typo, sửa.)*
- **Mỗi phase phải kèm "rollout smoke"** (`env.reset → policy → Scheduler-reward → backward`, reward finite).
- **Variant mặc định `P`** (hierarchical/HDCARP). P/U ảnh hưởng **Scheduler** (precedence vs hierarchy-level
  khi tính `T_k`), không phải mask. **Mọi rollout-smoke chạy ở `P` trước, lặp ở `U`**. `train.py` default `P`;
  eval/baselines ép `--variant P` khi so sánh (default baseline lệch P/U).
- **Không phá test data_plan cũ** (65 test hiện có phải vẫn xanh).
- File test mới gợi ý: `tests/test_scheduler.py` (Phase 1), mở rộng `tests/test_multisize.py` (Phase 3),
  `tests/test_encoder_mask.py` (Phase 4).
- `p=3` cố định; trần `|A_r|≤100`; metric thưa (floyd-warshall) **không đổi**; `variant='P'` mặc định.

## Ma trận

| Phase | Bất biến phải giữ | Test trọng yếu (PASS mới qua) |
|---|---|---|
| **0** Scope | — (quyết định) | Chốt **M-agnostic + Scheduler**; **`Q=Σq/3` (paper gốc) + multi-trip**; variant P; không sinh lại data |
| **1** Scheduler | policy M-agnostic; mask+`Q` giữ nguyên; `Φ` tôn trọng P/U | (1) ⭐ `T_k` đúng định nghĩa paper (ví dụ tay); (2) ⭐ hierarchy-level **P vs U** khác đúng; (3) ⭐ `T_1` **đơn điệu theo M**; (4) ⭐ continuity `K≤M` (M≥3) == song song cũ; (5) ⭐ **M=2 multi-trip** (3 trip→2 xe) feasible, không deadlock; (6) ⭐ rollout smoke M∈{2,3,5} **× {P,U}** |
| **2** M-cond *(HOÃN)* | train≈test không vỡ | (1) context+critic ăn M (shape); (2) ⭐ **output đổi theo M** (ở P); (3) gradient tới context-M+critic-M; (4) rollout smoke **× {P,U}**; (5) ⭐ **ablation** M-cond vs M-agnostic |
| **3** Train M-agnostic | size bucket giữ; M trộn tự do trong **reward**; **variant=P** | (1) parse `--num_vehicle` list; (2) ⭐ **pick M per-instance** (1 batch ≥2 M, `num_vehicle` `(B,1)`); (3) batch mixed-M không vỡ `torch.cat`; (4) ⭐ train-step **M-agnostic** (policy không nhận M) ×size (backward) **ở P**; (5) reward hữu hạn mọi M |
| **4** Dynamic size *(tùy chọn)* | không rò padding | (1) ragged == bucket (≤1e-4); (2) đổi padding → token thật không đổi; (3) không NaN; (4) masked-norm đúng; (5) rollout ragged |
| **5** Eval | dùng test grid sẵn có; **báo cáo chính ở P** | (1) cùng instance đa M → makespan đơn điệu theo M; (2) break-down theo size/d/topology **× variant**; (3) stats p∈[0,1], gap đúng; (4) report gap in-dist vs OOD; (5) baselines ép cùng `--variant P` |

## Quan hệ với data_plan

- **Không sinh data mới, không đổi `Q`** (đã đúng paper gốc). Dùng `data/ood/` + `import_instance(M=)` +
  `eval/stats.py`. M chỉ vào Scheduler.

## Checklist tổng

- [x] Phase 0: chốt **M-agnostic + Scheduler**; **`Q=Σq/3` (paper gốc), Scheduler multi-trip**; variant P. **CÒN: sửa typo main.tex.**
- [ ] Phase 1: tách `common/scheduler.py` (Φ); `calc_reward` mỏng lại → commit
- [ ] Phase 2 *(HOÃN)*: M-conditioning (context + critic) + ablation → commit khi làm sau
- [ ] Phase 3: train **M-agnostic** (train once, M quét trong reward) → commit
- [ ] Phase 4 *(tùy chọn)*: dynamic size masking/ragged → commit
- [ ] Phase 5: eval & report theo M/size + **ablation M-agnostic vs M-cond/per-M** → commit
- [ ] Cập nhật `docs/dynamic.md` nếu quyết định khác plan (giữ dynamic.md là nguồn sự thật về mục tiêu).
