# Phase 4 — Test OOD: đổi topology (giữ F2–F5)

> Mục tiêu: tạo **test set out-of-distribution** bằng cách áp **đúng F2–F5** lên topology khác unit-square:
> (a) **OSM thật nhiều thành phố**, (b) **clustered/grid synthetic**. Train trên unit-square (F1) → test OOD.
> Đây là deviation **tối thiểu & chỉ ở test** (per-arc formula y hệt). Phụ thuộc Phase 0–3.
> Tham chiếu `docs/data.md §5 P4, §8.D`.

## 4.1 — Nguyên tắc: chỉ đổi NGUỒN GRAPH, không đổi physics

`data/gen.py:build_instance(edges, coords, M, rng)` đã **tách rời** physics khỏi nguồn graph — nó chỉ
nhận `edges + coords`. Vì vậy OOD = **cấp `edges/coords` từ nguồn khác** rồi gọi đúng `build_instance`
(đã chứa F2–F5 sau Phase 0). Không viết lại physics.

## 4.2 — Các nguồn topology

| Cell | Nguồn | Cách tạo `edges/coords` |
|---|---|---|
| O1 | unit-square (in-dist) | F1 gốc — baseline so sánh |
| O2 | OSM city A | `load_osm_graph` + `get_random_connected_subgraph` (đã có) |
| O3 | OSM city B | như O2, bbox khác (thành phố khác) |
| O4 | clustered/grid | sinh synthetic: K cụm Gaussian + nối cạnh; hoặc lưới đều |

**File:** `data/gen.py`
- Thêm `--topology {unit_square, osm, cluster}` (hoặc nhiều bbox cho multi-city).
- `unit_square`: random coords trong `[0,1]²`, dựng min Hamiltonian + thêm cạnh tới `|A|=n·d` (như F1).
- `osm`: dùng pipeline OSMnx hiện có; cho nhận **nhiều bbox** (`--bbox` lặp) để nhiều city.
- `cluster`: sinh K tâm cụm, mỗi node thuộc 1 cụm; nối trong cụm dày + vài cạnh liên cụm; đảm bảo
  strongly-connected (lọc như `get_random_connected_subgraph`).
- Mọi nguồn → **chung `build_instance`** → đảm bảo F2–F5 đồng nhất. Lưu `topology` vào metadata `.npz`.

> ⚠️ OSMnx cần `uv add osmnx` (xem header `data/gen.py`). Phần `build_instance` + `cluster` **không** cần
> OSMnx ⇒ test được offline; chỉ nhánh `osm` cần network/lib.

## 4.3 — Quy ước thư mục & metadata

> ⚠️ **Cập nhật theo Phase 3 (revised): KHÔNG có tầng `<M>m`** — M là tham số eval, không phải trục sinh.

- `data/ood/<topology>/<|A|>/*.npz` (vd `data/ood/cluster/80/...`, `data/ood/osm_cityA/80/...`).
- O1 (unit_square) là in-distribution; eval script so O2–O4 vs O1 dưới **cùng M** (truyền `--M` lúc eval).
- Metadata lưu: `topology, d, M(nominal), τ(reference), n_req`. (`M`/`τ` chỉ là nominal — eval override M.)

---

## ✅ Cổng test Phase 4

### Test cần THÊM (offline, không cần OSMnx)
1. **Cluster sinh hợp lệ:** generator cluster cho ra graph **strongly-connected**, `build_instance`
   chạy được, ¼-split + cân bằng lớp giữ nguyên, không NaN.
2. **F2–F5 đồng nhất qua topology:** chạy `build_instance` trên (a) cycle synthetic, (b) cluster,
   (c) grid — cả ba đều thỏa: `service==2·d`, `q==0.5·d+0.5`, `C==Σ q/3+0.5`, lớp cân bằng.
   → chứng minh OOD **không** lén đổi công thức.
3. **Round-trip OOD npz:** `.npz` cluster load qua `common.ops.import_instance` finite, `demands==req[:,2]/C`.
4. **Metadata topology:** `.npz` chứa `topology` đúng nhãn.

### Test cần ĐÁNH DẤU SKIP (cần OSMnx)
5. Nhánh `osm` multi-bbox: bọc `@unittest.skipUnless(has_osmnx, ...)` để CI offline không fail.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `data/gen.py --topology` + nhánh unit_square / osm(multi-bbox) / cluster.
- [ ] Mọi nguồn dùng chung `build_instance` (F2–F5 bất biến).
- [ ] Lưu `topology` metadata; thư mục `data/ood/...`.
- [ ] Test cluster/grid offline xanh; test OSM skip-unless-osmnx.
- [ ] `unittest discover` xanh.
- [ ] Commit "Phase 4: OOD test sets (OSM multi-city + cluster), keep F2–F5".
