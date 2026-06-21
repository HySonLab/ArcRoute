# Bảng cổng test gộp — điều kiện "qua phase"

> Một chỗ tra nhanh: mỗi phase chỉ được chuyển tiếp khi **toàn bộ** test của nó + các phase trước **xanh**.
> Lệnh chung (chạy từ project root):
> ```bash
> uv run python -m unittest discover -s tests -p "test_*.py" -v
> ```

## Lưu ý xuyên suốt

- **Sửa test cũ, không chỉ thêm test mới.** Phase 0 đổi công thức ⇒ các test đang assert "60–70"
  (`test_required_count_large_instances`, `test_required_count_large`, `TestRequiredCount`) sẽ
  **fail có chủ đích** — phải cập nhật chúng sang ¼-split, coi đó là một phần của cổng Phase 0.
- **2 test capacity đang khóa công thức SAI** (`test_capacity_sums_over_required_arcs`,
  `test_capacity_formula`) phải sửa `ref` từ `Σ(q/3+0.5)` → `Σq/3+0.5` — nếu không, fix đúng ở §0.5 sẽ
  bị test cũ báo "đỏ" oan.
- **Mỗi phase = commit riêng**, test xanh trước khi commit. Không gộp phase.
- Các test cần OSMnx (`osm` branch ở Phase 4) phải `@unittest.skipUnless` để CI offline không đỏ.
- File test: `tests/test_generator.py` (env), `tests/test_gen.py` (benchmark). Thêm file mới nếu cần
  (vd `tests/test_density.py`, `tests/test_stats.py`) — vẫn theo pattern `unittest`, prefix `test_`.

## Ma trận

| Phase | Bất biến phải giữ | Test trọng yếu (PASS mới qua) |
|---|---|---|
| **Pre** Cleanup | test/import không phụ thuộc data cũ | (1) `unittest discover` xanh **sau khi** xóa `data/2m,5m`; (2) import `env.generator`/`common.ops`/`data/gen.py` OK; (3) không tham chiếu chết data cũ ở đường chạy chính; (4) tag `archive/pre-data-overhaul` tồn tại |
| **0** Restore | F2 ¼-split, lớp cân bằng, **F5 capacity `Σq/3+0.5`**, **đồ thị thưa+SC, metric dùng chung test** | (1) `\|A_r\|=3·⌊\|A\|/4⌋`; (2) 3 lớp cân bằng ±1; (3) ratio≈0.75 mọi size (chống scaling defect); (4) depot lớp 0; (5) **capacity `Q==Σq/3+0.5` và KHÁC `Σ(q/3+0.5)` cũ** (sửa 2 test đang khóa sai); (6) **⭐ train≈test metric: `adj_train≈import_instance(adj)` cùng tập arc**; (7) đồ thị train strongly-connected, `adj` finite; (8) round-trip integer finite |
| **1** Size cap | `\|A_r\| ≤ 100`, 1 size/batch | (1) trần `3·(num_arc//4)≤100`; (2) dải size random hợp lệ; (3) `torch.cat` khác `n` → raise (bucket); (4) ladder ra {20,40,60,80,100} |
| **2** Density | F1, trần | (1) `\|A\|≈n·d`; (2) trần giữ với mọi (n,d); (3) 4 mức d sinh được; (4) metadata `d` |
| **3** Fleet (revised) | **C bất biến theo M; M là tham số eval** | (1) **C không đổi khi đổi M**; (2) `import_instance(f, M=k)` override M, arcs/C không đổi; (3) sinh **không** tầng `<M>m`; (4) `τ(M)=Σq/(M·C)` report-time giảm theo M; (5) baseline `--M` parse |
| **4** OOD | F2–F5 đồng nhất mọi topology; layout `<topology>/<\|A\|>` | (1) cluster strongly-connected + hợp lệ; (2) F2–F5 giống nhau qua cycle/cluster/grid; (3) round-trip OOD npz; (4) metadata `topology`; (5) OSM skip-unless |
| **5** Stats | không đụng data | (1) `τ = Σq/(M·C)` report-time đơn điệu giảm theo M; (2) metadata đủ `d,M,topology,tau,n_req`; (3) seed độc lập; (4) stats script p-value∈[0,1], gap=0 khi obj=BKS |
| **6** Multi-size | 1 batch 1 size, các size luân phiên | (1) mỗi batch single-size; (2) đủ size ladder; (3) không batch vượt bucket; (4) train-step thật chạy (encoder+reward+backward) |

## Smoke test thủ công (ngoài unittest)

```bash
# Sinh on-the-fly 1 instance nhỏ (không GPU), xem shape adjacency:
uv run python -c "from env.generator import generate; td=generate(20,20,3); print({k:td[k].shape for k in td.keys()})"

# Benchmark physics offline (không OSMnx) — qua test_gen pathway:
uv run python -m unittest tests.test_gen -v
```

## Checklist tổng (toàn bộ kế hoạch)

- [x] Pre-Phase: tag archive + xóa data/checkpoint cũ, test xanh → commit
- [x] Phase 0 xanh → commit
- [x] Phase 1 xanh → commit
- [x] Phase 2 xanh → commit
- [x] Phase 3 (bản đầu, per-M) → commit. **Cần làm lại (revised): M là tham số eval, sinh 1 lần.**
- [x] Phase 4 xanh → commit
- [x] Phase 5 xanh → commit
- [x] Phase 6 (multi-size training) xanh → commit
- [ ] **Phase 3 revised**: bỏ per-M, `import_instance(M=...)`, baseline `--M`, regenerate layout mới → commit
- [ ] Cập nhật `docs/data.md` nếu có quyết định khác so với plan (giữ data.md là nguồn sự thật về thiết kế).
