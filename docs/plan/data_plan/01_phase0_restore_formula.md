# Phase 0 — Khôi phục công thức gốc Hà 2024 (NỀN TẢNG)

> Mục tiêu: sửa các chỗ HRDA **lệch công thức** về đúng F2 (¼-split + lớp cân bằng) và F3/F4 (integer).
> Đây là phase quan trọng nhất — nó vá **scaling defect** (required tụt còn ~30% ở `|A|=200`).
> **Không thêm trục mới** ở phase này.

## 0.1 — ¼-split + lớp cân bằng trong `env/generator.py`

**File:** `env/generator.py` → hàm `sample_priority_classes(num_arc)` (dòng 64–73).

Hiện tại: random 1–3 + rule "60–70 cố định" khi `num_arc > 80`. Cần thay bằng:

- `per_class = num_arc // 4` (≈ `⌊|A|/4⌋`).
- Gán **đúng `per_class` arc cho mỗi lớp {1,2,3}** (cân bằng), phần dư = lớp 0 (non-required).
- Index 0 (depot self-loop) luôn lớp 0.
- Dùng `torch.randperm` để xáo trộn arc nào vào lớp nào (giữ tính ngẫu nhiên vị trí, nhưng **số lượng
  mỗi lớp cố định**).

Gợi ý logic (không copy nguyên si — đọc lại code quanh đó):
```python
def sample_priority_classes(num_arc):
    per_class = num_arc // 4                      # ⌊|A|/4⌋ mỗi lớp (F2)
    clss = torch.zeros(num_arc + 1, dtype=torch.int64)   # mặc định lớp 0
    perm = torch.randperm(num_arc) + 1            # bỏ index 0 (depot)
    for c in (1, 2, 3):
        idx = perm[(c - 1) * per_class : c * per_class]
        clss[idx] = c
    return clss                                   # index 0 vẫn = 0
```
> `|A_r| = 3·per_class ≈ 0.75·num_arc` mọi size → hết scaling defect. Không còn nhánh `num_arc > 80`.

## 0.2 — ¼-split + lớp cân bằng trong `data/gen.py`

**File:** `data/gen.py`.

1. `required_count(num_arc, rng)` (dòng 39–43): thay cả 2 nhánh bằng `3 * (num_arc // 4)`.
   ```python
   def required_count(num_arc, rng=np.random):
       return 3 * (num_arc // 4)        # F2: 3 lớp, mỗi lớp ⌊|A|/4⌋
   ```
2. `build_instance` (dòng 87): `clss = rng.randint(1, 4, size=n_req)` → gán **cân bằng**:
   ```python
   per_class = n_req // 3
   clss = np.repeat([1, 2, 3], per_class)
   clss = np.concatenate([clss, np.full(n_req - len(clss), 3)])  # dư cho lớp cuối
   rng.shuffle(clss)
   ```
   > Vì `n_req = 3·(|A|//4)` chia hết cho 3 nên thường `per_class` chẵn; vẫn xử lý phần dư cho an toàn.

## 0.3 — Integer F3/F4 cho instance LƯU RA (tùy chọn nhưng nên làm)

> F3: `d_a = max{1, ⌊d'_a + 0.5⌋}` với `d'_a` = Euclid **×100**. F4: `q = ⌊d·0.5+0.5⌋`, `s = 2·d`.
> Chuẩn hóa `/d_max` hay `/Q` **chỉ ở bước input NN**, KHÔNG lưu vào instance → không tính là đổi công thức.

- `data/gen.py:build_instance`: thay `d = d_eucl / d_max` (liên tục) bằng
  `d = np.maximum(1, np.round(d_eucl * 100)).astype(int)` rồi tính `q, s` integer. Lưu integer vào `.npz`.
  Chuẩn hóa `/C` đã làm sẵn ở `common/ops.import_instance:demands = req[:,2]/C` (giữ nguyên).
- `env/generator.py`: là **nice-to-have** (xem 0.6). `traversal_time` per-arc giữ **chuẩn hóa `/d_max` ở
  input** (input-preprocessing, hợp lệ) — KHÔNG bắt buộc integer-hóa. Lưu ý: việc sửa **metric `adj`**
  (sparse + floyd-warshall dùng chung với test) là MUST-FIX nằm ở 0.6, **không** thuộc mục 0.3 này.

> ⚠️ Lưu ý độ rủi ro: integer hóa làm `import_instance`/`floyd_warshall` đổi đơn vị (×100). Phải chạy
> round-trip test (mục test 4) để chắc adjacency vẫn finite & demands chuẩn hóa đúng.

## 0.4 — Thống nhất `train.sh`

**File:** `train.sh`. `NUM_ARC=20` + `NUM_LOC=20` → `|A|=20`, quá nhỏ và lệch. Đặt về cấu hình F1
hợp lệ (vd `NUM_LOC=40, NUM_ARC=80` cho `|A_r|≈60`). Để Phase 1 lo dải size; ở P0 chỉ cần **một**
cấu hình nhất quán F1–F5 thay cho giá trị cũ. Ghi chú lý do ngay trong comment của `train.sh`.

## 0.5 — Sửa công thức CAPACITY F5 (BẮT BUỘC — code đang SAI)

> ⚠️ **Đây là sai lệch công thức thật, không phải "khác nhẹ".** Paper trang 18:
> **`Q = (Σ_{a∈A_r} q_a)/3 + 0.5`** — cộng `0.5` **một lần** (3 xe chia đều tổng demand + slack nhỏ).
> Code hiện tính `Σ(q_a/3 + 0.5) = (Σq)/3 + 0.5·|A_r|` — cộng `0.5` **mỗi arc** → với `|A_r|=75` thì
> capacity lỏng gấp ~75 lần → ràng buộc tải gần như không kích hoạt → instance dễ hơn paper. Đây cũng là
> `Q` trong `τ = Σq/(M·Q)` mà Phase 5 report → nếu không sửa thì **τ và toàn bộ độ khó đều sai**.

**File:** `env/generator.py` → `sample_vehicle_capacity` (dòng 59–62):
```python
def sample_vehicle_capacity(demand, priority_classes):
    # paper Q = (Σ_{a∈A_r} q_a) / 3 + 0.5   (cộng 0.5 MỘT LẦN)
    required = priority_classes > 0
    return demand[required].sum() / 3 + 0.5
```

**File:** `data/gen.py` → `build_instance` (dòng 89–90):
```python
# Step 7: vehicle capacity Q = (Σ_{a∈A_r} q_a) / 3 + 0.5
C = float(q_req.sum() / 3.0 + 0.5)
```

> ⚠️ Cả comment cũ ("sum over required arcs of (q_a/3 + 0.5)") cũng sai — sửa luôn comment để khỏi
> tái phạm. Việc này đổi `C` của instance → ảnh hưởng `demands = q/C` (đã normalize trong code). Chạy
> lại round-trip test sau khi sửa.

## 0.6 — Làm generator TRAIN đúng F1 (HƯỚNG B-scoped — ĐÃ CHỐT)

> **Lý do (quan trọng nhất của cả Phase 0):** train và test hiện đang dùng **2 loại metric khác hẳn nhau**,
> đây nhiều khả năng là nguyên nhân lớn nhất khiến điểm test kém — lớn hơn cả lỗi F2/F5.
>
> - **Train** (`sample_traversal_time`): `dists = cdist(coords, coords)` = ma trận Euclid **ĐẦY ĐỦ** giữa
>   mọi cặp đỉnh → min-plus là no-op (Euclid đã thỏa tam giác) → `adj` = **khoảng cách đường thẳng** giữa
>   2 arc. Tức đồ thị train thực chất là **complete graph, đi lại = teleport theo đường chim bay**. Việc
>   `sample_arcs` bốc arc ngẫu nhiên cũng không tạo ra cấu trúc đường thật.
> - **Test** (`common/ops.import_instance` → `floyd_warshall`): `adj` chỉ dựng từ **các arc thật** (còn lại
>   `inf`) rồi floyd-warshall → **shortest-path thật trên đồ thị thưa, có hướng**, dài hơn đường thẳng & bất
>   đối xứng.
>
> ⇒ Model học hình học "đường thẳng đối xứng" rồi test trên "bám đường, có hướng" → `adj` lệch bản chất
> → generalize tệ. **Phải đóng khoảng cách này.**

### MUST-FIX (tác động cao lên điểm test — đây là phần chính)

1. **Đồ thị train = sparse + strongly-connected.** Viết lại `sample_arcs(num_loc, num_arc)`:
   - Dựng **min Hamiltonian circuit** qua tất cả đỉnh (đảm bảo liên thông mạnh) — tái dùng được ý tưởng/
     logic kiểm tra `nx.is_strongly_connected` như `data/gen.py`.
   - Thêm arc ngẫu nhiên tới `|A| = round(num_loc·d)` (xem Phase 2 cho `d`), KHÔNG trùng, depot (đỉnh 0)
     là tail của ≥1 arc.
   - Trả về tập arc đúng như đồ thị đường thưa (không phải để encoder "thấy" complete graph).

2. **`adj` tính trên ĐÚNG tập arc thưa, DÙNG CHUNG pipeline với test.** Sửa `sample_traversal_time` (hoặc
   tách hàm) sao cho:
   - `adj` = floyd-warshall trên ma trận chỉ chứa **arc thật** (`inf` ngoài tập arc), **không** dùng
     `cdist` đầy đủ.
   - **Tái dùng `common/ops.convert_adjacency_matrix` + `floyd_warshall` + `dist_edges`** (đã có sẵn, đang
     dùng cho test) thay vì tự viết min-plus riêng → đảm bảo train và test đi **chung một đường tính**.
     Đây là điều kiện để test "train≈test metric" (mục test 9) pass.
   - `traversal_time` per-arc vẫn = độ dài Euclid của chính arc đó (chuẩn hóa — xem dưới).

### NICE-TO-HAVE (tác động thấp — làm nếu rẻ, bỏ được, KHÔNG chặn Phase 0)

3. Lọc "arc không cắt nhau / không quá dài" (planarity F1). Ảnh hưởng nhỏ tới điểm test.
4. Integer ×100 cho F3/F4 ở generator train (xem 0.3). Vì input vẫn normalize nên ít đổi điểm test;
   để khớp chữ paper thì làm, còn không thì **giữ chuẩn hóa `/d_max` như input-preprocessing** (hợp lệ).

### Phân vai train vs test (để vừa dễ train vừa generalize)

- **Train:** unit-square F1 (đã sparse + strongly-connected sau khi sửa) — rẻ, on-the-fly, đa dạng "miễn
  phí" qua 4 mức mật độ `d` (Phase 2). **KHÔNG tải OSM khi train.**
- **Test in-distribution:** unit-square F1 (cùng họ).
- **Test OOD:** OSM thật nhiều city + cluster (Phase 4).
- Compute: n≤101, floyd-warshall ~10⁶ ops/instance, chạy trong dataloader workers → thoải mái.

> Ghi 1 dòng giả định vào `00_overview.md` + comment đầu `env/generator.py`: generator train dựng
> **đồ thị thưa strongly-connected (đúng tinh thần F1)**, `adj` dùng chung floyd-warshall với test; planarity
> & integer là tùy chọn (nice-to-have).

---

## ✅ Cổng test Phase 0 (PHẢI PASS mới qua Phase 1)

**File test:** `tests/test_generator.py` (cho `env/generator.py`), `tests/test_gen.py` (cho `data/gen.py`).

### Test cần SỬA (đang fail có chủ đích sau khi đổi công thức)
1. `tests/test_generator.py::TestPriorityClasses::test_required_ratio_small_instances` —
   đổi assert thành `req == 3*(num_arc//4)`.
2. `tests/test_generator.py::TestPriorityClasses::test_required_count_large_instances` —
   bỏ assert "55–72"; thay bằng `req == 3*(100//4) == 75`.
3. `tests/test_gen.py::TestBuildInstance::test_required_count_large` — `|A|=120` → `req == 3*(120//4)==90`.
4. `tests/test_gen.py::TestBuildInstance::test_required_count_small` & `TestRequiredCount` — cập nhật
   sang công thức ¼-split (vd `required_count(40) == 30`? Không: `3*(40//4)=30` ✓; `required_count(20)=15` ✓
   — trùng giá trị cũ ở size nhỏ, nhưng `required_count(150)` giờ = `3*37=111`, KHÔNG còn 60–70).
5. **⚠️ Capacity (đang KHÓA công thức SAI — phải sửa, xem 0.5):**
   - `tests/test_generator.py::TestVehicleCapacity::test_capacity_sums_over_required_arcs` — `ref` hiện là
     `(demand[clss>0]/3 + 0.5).sum()` (SAI) → đổi thành `demand[clss>0].sum()/3 + 0.5`.
   - `tests/test_gen.py::TestBuildInstance::test_capacity_formula` — `ref` hiện là `(req[:,2]/3 + 0.5).sum()`
     (SAI) → đổi thành `req[:,2].sum()/3 + 0.5`.
   - `test_capacity_not_constant` giữ nguyên (vẫn đúng với công thức mới).

### Test cần THÊM MỚI
6. **Cân bằng lớp** (cả 2 generator): mỗi lớp {1,2,3} có số arc bằng nhau ± 1.
   ```python
   counts = [int((clss == c).sum()) for c in (1, 2, 3)]
   assert max(counts) - min(counts) <= 1
   ```
7. **Scaling không vỡ:** với `num_arc ∈ {20,40,80,120,200}`, `|A_r| / |A| ≈ 0.75` (sai số ≤ 0.05).
   → đây là test trực tiếp chống regression của "scaling defect".
8. **Depot vẫn lớp 0** (đã có — giữ).
9. **Capacity đúng F5 (0.5):** với demand giả định cụ thể, `Q == Σq/3 + 0.5` (một +0.5), và **KHÁC**
   giá trị cũ `Σ(q/3+0.5)` ít nhất `0.5·(|A_r|-1)` → test chống tái phạm:
   ```python
   q = demand[clss > 0]
   assert abs(float(cap) - (float(q.sum())/3 + 0.5)) < 1e-5
   assert abs(float(cap) - float((q/3 + 0.5).sum())) > 0.5   # phải khác công thức cũ
   ```
10. **⭐ Train≈test metric (CỔNG CHÍNH của B-scoped, 0.6):** chứng minh `adj` sinh ở train **trùng** với
    `adj` mà test (`import_instance`) tính trên **cùng tập arc**. Đây là test trực tiếp chống lỗi
    complete-vs-sparse:
    ```python
    # 1) sinh 1 instance train, lấy arcs (req+nonreq) + adj của nó
    # 2) đóng gói chính các arc đó thành .npz (req/nonreq, P, M, C)
    # 3) chạy qua common.ops.import_instance -> dms
    # 4) assert adj_train ≈ dms (cùng floyd-warshall, cùng tập arc thưa)
    assert np.allclose(adj_train, dms, atol=1e-4)
    # và đảm bảo KHÔNG còn là complete-Euclid:
    # với đồ thị thưa, tồn tại cặp arc mà shortest-path > đường thẳng Euclid
    assert (dms > euclid_straight + 1e-6).any()
    ```
11. **Strongly-connected (0.6 must-fix #1):** đồ thị arc do `sample_arcs` sinh ra là liên thông mạnh
    (dựng `nx.DiGraph` từ arcs → `nx.is_strongly_connected` True); `adj` finite, không `inf`.
12. (Nếu làm 0.3 integer) `tests/test_gen.py`: `d, q, s` là số nguyên ≥ 1; round-trip `import_instance`
    vẫn `finite`, `demands[1:] == req[:,2]/C`.

### Lệnh chạy
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

### Checklist
- [ ] `sample_priority_classes` dùng ¼-split + cân bằng, bỏ nhánh `num_arc>80`.
- [ ] `required_count` = `3*(num_arc//4)`; `build_instance` gán lớp cân bằng.
- [ ] **(BẮT BUỘC) Sửa capacity F5 → `Σq/3 + 0.5`** ở cả 2 generator + sửa comment.
- [ ] **(B-scoped MUST-FIX) `sample_arcs` dựng đồ thị thưa + strongly-connected** (Hamiltonian circuit + thêm arc tới `|A|=n·d`).
- [ ] **(B-scoped MUST-FIX) `adj` train dùng chung `convert_adjacency_matrix`+`floyd_warshall`+`dist_edges`** với test (bỏ `cdist` complete).
- [ ] Ghi giả định F1 generator train vào `00_overview.md` + comment đầu `env/generator.py`.
- [ ] (nice-to-have) planarity / integer F3/F4 — bỏ được, không chặn phase.
- [ ] `train.sh` về cấu hình F1–F5 nhất quán, có comment.
- [ ] **Sửa** 5 nhóm test cũ ở trên (gồm 2 test capacity đang khóa sai) + **thêm** test cân bằng, scaling, capacity, **train≈test metric**, strongly-connected.
- [ ] `unittest discover` xanh toàn bộ.
- [ ] Commit riêng "Phase 0: restore Hà-2024 formula + sparse train graph (¼-split, balanced, Q=Σq/3+0.5, shared floyd-warshall metric)".
