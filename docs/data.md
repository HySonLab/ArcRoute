# Đánh giá & Hướng cải thiện Dataset cho HDCARP

> Mục tiêu: đánh giá data của project và đề xuất thiết kế đủ mạnh cho bài báo **Q1 / hạng A***,
> với **ràng buộc bắt buộc: GIỮ NGUYÊN công thức sinh instance của Hà et al. 2024** (bài báo gốc
> đề ra bài toán HDCARP). Mọi đề xuất dưới đây chỉ cải thiện các trục **KHÔNG làm đổi công thức**
> (quy mô, mật độ, số xe, đa dạng topology cho test, số seed, thống kê).
>
> Bài toán: **HDCARP** — required arcs chia thành priority class; biến thể **P** (precedence
> nghiêm ngặt) và **U** (thứ tự phục vụ linh hoạt). Mục tiêu: **lexicographic makespan** — min max
> completion time theo từng lớp (KHÔNG phải tổng quãng đường).

---

## 1. Công thức gốc Hà 2024 — phần BẤT BIẾN (không được đổi)

Trích từ `papers/2024_Ha_HDCARP_matheuristic.pdf` §5.1:

| Mã | Thành phần | Công thức (giữ nguyên) |
|---|---|---|
| **F1** | Đồ thị | `n` đỉnh trong unit square (`n` bội số 10, 10→100), đỉnh 1 = depot; dựng min Hamiltonian circuit → thêm cung ngẫu nhiên tới `\|A\| = n·d`, **mật độ `d ∈ {1.5, 2, 2.5, 3}`**; cung không quá dài, không cắt nhau |
| **F2** | Chia required | `A_r^1, A_r^2, A_r^3` mỗi lớp `⌊\|A\|/4⌋`; `A_nr` = phần dư; **`P = {1,2,3}`** → required ≈ 75%, non-required ≈ 25%, **3 lớp cân bằng (~¼ mỗi lớp)** |
| **F3** | Distance | `d_a = max{1, ⌊d'_a + 0.5⌋}`, `d'_a` = Euclid × 100 → **số nguyên ≥ 1** |
| **F4** | Demand / service | `q_a = ⌊d_a·0.5 + 0.5⌋` (số nguyên); `s_a = 2·d_a` |
| **F5** | Fleet / capacity | 3 xe đồng nhất; `Q = Σ_{a∈A_r} q_a/3 + 0.5` |

> Đây là "hợp đồng" không đổi. Mọi cải thiện ở §5 đều phải bảo toàn F1–F5.

---

## 2. Hiện trạng & độ lệch của HRDA so với bản gốc

| Trục | **Hà 2024 (gốc)** | **HRDA 2025 (repo này)** | Nhận xét |
|---|---|---|---|
| Số xe | 3 cố định | test {2,5}; train |M|=5; code default 3 | HRDA đổi (nguồn gốc "2,3,5") |
| Required | **¼-split, ~75% mọi size** | 75% nếu \|A\|<80, **60–70 cố định** nếu ≥80 | ⚠️ HRDA **lệch công thức** → gây scaling defect |
| Classes | 3 cân bằng (¼ mỗi lớp) | 3, **gán ngẫu nhiên 1–3** (không cân bằng) | HRDA lệch |
| Distance/demand | **số nguyên** (×100, ⌊⌋) | **liên tục** + chuẩn hóa `/d_max` | HRDA lệch |
| Đồ thị | unit square + **4 mức mật độ d** | OSMnx 1 bbox, **mất biến thiên d** | HRDA lệch |
| Capacity | `Σq/3 + 0.5` | `Σ(q/3 + 0.5)` | khác nhẹ |
| Seeds | 5 version | 20/cell | HRDA tốt hơn |

**Kết luận then chốt:** phần lớn vấn đề data của repo đến từ việc **HRDA đi chệch công thức gốc**, chứ
không phải công thức gốc sai. Đặc biệt **scaling defect** (tỉ lệ required tụt còn ~30% ở \|A\|=200)
là do rule "60–70 cố định" của HRDA — **bản gốc Hà giữ ¼-split nên KHÔNG dính lỗi này**.

**3 nguồn không nhất quán trong repo:** Hà (3 xe, ¼-split, integer) ≠ HRDA paper (|M|∈{2,5}, 60–70) ≠
`train.sh` (`num_arc=20, num_vehicle=3`) ≠ `data/2m` (req=20 ở \|A\|=100). Cần thống nhất **về bản gốc**.

---

## 3. Trả lời 3 câu hỏi (đã cập nhật theo bản gốc + ràng buộc giữ công thức)

### 3.1 Số xe {2,3,5} — ổn chưa?
Vì mục tiêu là **makespan**, số xe là **input hợp lệ** (thêm xe → makespan giảm) — khác CARP min-cost
nơi fleet để free. Bản gốc dùng **3 cố định**; HRDA dùng {2,5}. Vấn đề là **lưới quá hẹp & train≠test**.
→ Cải thiện **không đổi công thức**: **quét `M` như tham số** {1,2,3,5,7,10}, train/test nhất quán.
Giữ nguyên F5 (Q vẫn `Σq/3+0.5`; chỉ số xe khả dụng thay đổi). Thêm `M=1` để nối lý thuyết HCPP.

### 3.2 Priority classes = 3 — ổn chưa?
`p=3` là của **bản gốc** và **gắn chặt với công thức ¼-split (F2)**. Đổi `p` (vd 4–5 lớp) sẽ **đổi F2**
→ **NGOÀI phạm vi** nếu giữ công thức Hà. ⇒ **Giữ `p=3`**. (Literature như HMRPP quét 2–5 lớp; nếu sau
này muốn, đây là một extension RIÊNG, có đánh dấu là "đổi công thức" — xem §5 Out-of-scope.)

### 3.3 Rule required arcs — ổn chưa?
Bản gốc **¼-split (≈75% mọi size) là ĐÚNG và scale chuẩn**. Lỗi nằm ở **HRDA đổi sang "60–70 cố định"**
→ tỉ lệ tụt theo size → độ khó không tăng theo \|A\|. 
→ Cải thiện = **quay về đúng công thức gốc F2** (¼-split). Vừa khắc phục scaling defect, vừa **không
đổi công thức** (thực ra là khôi phục công thức).

---

## 4. Đánh giá theo literature (phân loại theo "có đụng công thức không")

| Trục | Hiện trạng | Chuẩn literature | Cải thiện được mà KHÔNG đổi công thức? |
|---|---|---|---|
| Fleet | hẹp {2,5}/3 | makespan → fleet là input; CARP min-cost thì free | ✅ quét M (giữ F5) |
| Required ratio | HRDA lệch (60–70) | egl/BMCV subset; gdb/val 100% | ✅ về ¼-split gốc (F2) |
| Quy mô | ≤150–300 arcs | NCO test tới 1000–7000 (LEHD, TAM); LSCARP EGL-G/Hefei | ⏹ **chốt trần `\|A_r\|≤100`** (fit 4090, §5.7); large-scale = future work |
| Mật độ đồ thị | HRDA bỏ; gốc có `d∈{1.5..3}` | đa dạng cấu trúc là yêu cầu (Bossek 2019; Smith-Miles 2023) | ✅ khai thác đủ 4 mức d của F1 |
| Đa phân phối | 1 nguồn | uniform+cluster+real (AMDKD, Omni-VRP, RRNCO) | ⚠️ chỉ ở **test OOD**, giữ F2–F5 (xem I4) |
| Demand/service | coupled theo length | gdb/val: demand & cost ĐỘC LẬP | ❌ tách ra = đổi F4 → out-of-scope |
| Classes p | 3 | HMRPP quét 2–5 | ❌ đổi p = đổi F2 → out-of-scope |
| Seeds/rigor | 5–20, 1 seed | ≥20–30 seed; Wilcoxon/Friedman; gap-to-BKS (Accorsi 2022) | ✅ thuần quy trình |
| Capacity | đúng tinh thần | `Q=⌈r·Σq/n⌉` (Uchoa 2017) | ✅ giữ F5, chỉ **report tightness** |

---

## 5. Đề xuất — GIỮ NGUYÊN công thức Hà 2024

> **Nguyên tắc:** không chạm F1–F5. Chỉ thay đổi: quy mô `n`, mật độ `d` (vốn đã có), số xe `M` (input),
> nguồn topology cho **test OOD**, số seed, và phần thống kê/đo đạc.

### P0 — Thống nhất về công thức gốc (bắt buộc, KHÔNG thêm công thức)
1. **Khôi phục ¼-split (F2)** trong HRDA/code thay cho rule "60–70" → khắc phục scaling defect.
2. **Class gán cân bằng ~¼ mỗi lớp** (F2) thay cho random 1–3.
3. **Dùng số nguyên F3/F4** cho instance lưu ra (`d=max{1,⌊·⌋}`, `q=⌊·⌋`). *Chuẩn hóa `/d_max` hay
   `/Q` chỉ là bước tiền xử lý INPUT cho mạng NN — không lưu vào instance, nên không tính là đổi công thức.*
4. Thống nhất `train.sh`, generator và benchmark về cùng F1–F5.

### P1 — Quy mô: TRẦN CỨNG `|A_r| ≤ 100` (giữ F1/F2)
5. **Chốt `|A_r| ≤ 100`.** Vì giữ ¼-split (F2), `|A_r| ≈ 0.75·|A|` → **`|A| ≤ ~133`** (|A_nr| ≤ ~33).
   Toàn bộ dataset (train **và** test) nằm trong trần này → vừa khớp cấu hình train của paper
   (|A_r|=100), vừa **fit gọn 4090** (n≤101 → batch ~2000, KHÔNG OOM — xem §5.7); bỏ hẳn rủi ro
   large-scale & nhết vấn đề compute.
6. **Size ladder = quét `|A_r|`** trong `{20, 40, 60, 80, 100}` (≈ `|A| ∈ {27,53,80,107,133}`).
   Train ở 1 mức (vd `|A_r|=60`, hoặc mixed ≤100), test cả thang → **size-generalization trong [≤100]**.
7. **Bucket theo size**: mỗi batch chỉ 1 giá trị `n` (code ghép `adj (B,n,n)` bằng `torch.cat`,
   encoder chưa hỗ trợ mask) → không trộn size trong 1 batch.

### P2 — Khai thác mật độ `d` sẵn có trong F1
6. Sinh đủ **cả 4 mức `d ∈ {1.5,2,2.5,3}`** và **report kết quả theo `d`** (HRDA hiện bỏ trục này).
   Đây là đa dạng cấu trúc "miễn phí" vì đã nằm trong công thức gốc.

### P3 — Quét số xe như tham số (F5 nguyên)
7. `M ∈ {1,2,3,5,7,10}`, train/test nhất quán; `Q` vẫn `Σq/3+0.5`. (M=1 nối HCPP.)

### P4 — Đa phân phối CHO TEST OOD (giữ F2–F5, chỉ đổi nguồn graph)
8. Tạo **test set OOD** bằng cách áp **đúng F2–F5** lên topology khác: (a) **OSM thật nhiều thành phố**,
   (b) **clustered/grid**. Train trên unit-square (F1) → test OOD. Đây là deviation **tối thiểu & chỉ ở
   test** (per-arc formula y hệt), cần thiết để claim distribution-generalization. *Nếu muốn tuyệt đối
   không rời F1, bỏ P4 và chỉ dựa vào P1+P2 (scale + density) — vẫn hợp lệ nhưng yếu hơn về OOD.*

### P5 — Chặt chẽ thống kê (thuần quy trình, không đụng data)
9. **≥20–30 seed/cell** (gốc chỉ 5); report mean±std.
10. **Wilcoxon signed-rank** (cặp), **Friedman + CD diagram** (≥3 thuật toán), α=0.05.
11. **gap-to-BKS** (best & mean) + runtime chuẩn hóa (Accorsi-Lodi-Vigo 2022).
12. **Report tightness** `τ = Σq/(M·Q)` mỗi instance (mô tả phân bố độ khó — Smith-Miles 2023).

### ❌ Out-of-scope (sẽ ĐỔI công thức Hà — chỉ làm nếu chấp nhận rời bản gốc)
- Quét `p ∈ {2,4,5}` (đổi F2).  • Demand/service độc lập với length (đổi F4).  • Đổi dải demand / bỏ
  số nguyên (đổi F3/F4).  • Đổi hằng `/3` trong capacity (đổi F5).

### Bảng tham số đề xuất (giữ công thức)
| Tham số | Bản gốc Hà | Đề xuất (giữ công thức) | Đổi công thức? |
|---|---|---|---|
| `p` (classes) | 3 | **giữ 3** | giữ |
| split required | ¼ mỗi lớp (~75%) | **giữ ¼-split** | giữ |
| distance/demand | integer | **giữ integer** | giữ |
| capacity | `Σq/3+0.5` | **giữ**, thêm report `τ` | giữ |
| `M` (fleet) | 3 | **{1,2,3,5,7,10}** | ✅ không |
| `d` (mật độ) | {1.5,2,2.5,3} | **dùng đủ 4 + report theo d** | ✅ không |
| **`\|A_r\|` (TRẦN CỨNG)** | ≤~225 | **≤ 100** (train & test) | ✅ không |
| `\|A\|` (= `\|A_r\|`/0.75) | ≤~300 | **≤ ~133** | ✅ không |
| size ladder | — | quét `\|A_r\| ∈ {20,40,60,80,100}` | ✅ không |
| topology | unit square | + OSM nhiều city + cluster (**chỉ test OOD**) | ⚠️ chỉ đổi nguồn graph |
| seeds | 5 | **≥20–30** | ✅ không |
| batch | — | **bucket theo size** (1 size / batch) | ràng buộc compute |

---

### 5.7 Ràng buộc compute (O(n²)) — quyết định trần quy mô

Đã kiểm tra code (`policy/encoder.py`, `env/env.py`, `collate_fn`). Model scale theo
`n = |A_r| + 1` (số required arc + depot); tour decode dài theo `|A_r|`. Ba ràng buộc cứng:

1. **Attention dense O(n²), KHÔNG flash:** `attn_weights = q·kᵀ` rồi `* adj` (adj là ma trận
   shortest-path, không phải 0/1) → phải materialize `(batch, heads, n, n)`, không dùng được
   flash/SDPA tiết kiệm RAM.
2. **`adj` là `(batch, n, n)`** + decode **n bước tuần tự** (pointer).
3. **Batch phải cùng `n`:** `collate_fn = torch.cat(adj (1,n,n))` và encoder `assert mask is None`
   ("Mask not yet supported!") → không trộn nhiều size trong 1 batch (phải bucket theo size).

**Ước lượng training** (4090 ~24GB): bộ nhớ chi phối ≈ `batch × 8 × n² × 4B × 12 layers × ~2`
→ **batch tối đa ≈ 2.3×10⁷ / n²**. Với trần đã chốt **`|A_r| ≤ 100` (n ≤ 101)** thì mọi cấu hình
nằm trong vùng ✅:

| \|A_r\| | n=\|A_r\|+1 | batch khả thi (train) | Đánh giá |
|---|---|---|---|
| 20 | 21 | ~4096 | ✅ |
| 40 | 41 | ~4096 | ✅ |
| 60 | 61 | ~4096 (giảm nhẹ nếu cần) | ✅ |
| 80 | 81 | ~3000 | ✅ |
| **100 (trần)** | 101 | ~2000 | ✅ sát giới hạn ở batch 4096 |
| *(vượt trần — không làm)* 150 | 151 | ~1000 | ⚠️ cần AMP+ckpt |
| *(vượt trần)* 200 | 201 | ~512 | ❌ batch 4096 OOM |

**Hệ quả (với `|A_r| ≤ 100`):**
- **Không còn vấn đề compute**: toàn bộ train/test fit 1×4090, không cần AMP/checkpointing.
- Vẫn **bucket theo size** (mỗi `n` một batch) vì encoder chưa hỗ trợ mask.
- Nếu sau này muốn vượt trần (≥150) để claim large-scale: cần **AMP (bf16) + gradient checkpointing**
  hoặc viết lại attention (sparse/flash thay `*adj`, thêm mask cho batch ragged) — **future work**.

## 6. Map vào code (chỉ sửa tham số, không sửa công thức)
- `env/generator.py`: thay rule required về **¼-split (F2)**; gán class cân bằng; cho `num_loc/num_arc`
  và `num_vehicle` nhận dải; (tùy chọn) dùng integer như F3/F4 cho instance lưu, chuẩn hóa chỉ ở input.
- `data/gen.py`: `required_count` → ¼-split; `--vehicles` → dải `M`; thêm tham số mật độ `d`; (P4)
  cho phép nguồn graph = unit-square / OSM / cluster nhưng dùng chung F2–F5.
- Thêm metadata vào `.npz` (`d, M, τ, |A_r|`) để report.

---

## 7. Tài liệu tham khảo
Xem `papers/index.md` (đã tải 24 PDF). Các mục chính:

**HDCARP gốc & RL**
- Hà, Dang, Le, Nguyen, Langevin (2024) — *On the HDCARP* (bài gốc, MILP + matheuristic; nguồn F1–F5).
  `papers/2024_Ha_HDCARP_matheuristic.pdf`; instances tại orlab.com.vn/materials.
- Nguyen, Nguyen, Dang, Hy (2025) — HRDA, arXiv:2501.00852.

**Benchmark & survey arc routing**
- Golden/DeArmon/Baker (1983) gdb; Benavent et al. (1992) val; Li & Eglese (1996), Brandão & Eglese (2008) egl/EGL-G;
  Beullens et al. (2003) BMCV; Belenguer et al. (2006) MCARP; Tang et al. (2017) large-scale Hefei/Beijing;
  Corberán & Laporte (2014) SIAM; Corberán et al. (2021) Networks survey.

**Hierarchical / priority**
- Dror et al. (1987) HCPP; Letchford & Eglese (1998); Cabral et al. (2004); Korteweg & Volgenant (2006);
  Colombi et al. (2017) HMRPP (quét 2–5 lớp); Afanasev et al. (2021).

**Neural routing & chuẩn đánh giá**
- Nazari (2018), Kool (2019), Kwon POMO (2020); Luo LEHD (2023), Drakulic BQ-NCO (2023), Hou TAM (2023);
  Bi AMDKD (2022), Zhou Omni-VRP (2023); Joshi (2021), Liu (2023); Uchoa (2017) capacity/X-instances;
  Bossek (2019) tspgen; Smith-Miles & Muñoz (2023) ISA; Accorsi-Lodi-Vigo (2022) guidelines; Carrasco (2020).

**Neural arc-routing & data thật**
- DaAM (2024), Arc-DRL/CPP-LC (2023), Agricultural Spraying CARP (2023), RRNCO (2025).

---

## 8. Phụ lục — Cấu hình ví dụ (để kiểm tra trực quan)

Mọi số tính sẵn từ công thức gốc: `|A| = n·d` (n = số đỉnh, d = bậc trung bình / mật độ),
¼-split (`mỗi lớp = ⌊|A|/4⌋`, `|A_r| = 3×`), `s=2d`, `Q=Σ_{A_r} q/3+0.5`; lọc `n·d` để `|A_r| ≤ 100`.
Mỗi bảng **cô lập 1 trục** (kiểu benchmark chuẩn).

### A. Test — lưới quy mô (vary size; base M=5; variant P & U)
| Cell | n | d | \|A\| | /lớp | **\|A_r\|** | \|A_nr\| | variant | M | seeds | #inst |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 20 | 2.0 | 40 | 10 | **30** | 10 | P,U | 5 | 20 | 80 |
| S2 | 30 | 2.0 | 60 | 15 | **45** | 15 | P,U | 5 | 20 | 80 |
| S3 | 40 | 2.0 | 80 | 20 | **60** | 20 | P,U | 5 | 20 | 80 |
| S4 | 50 | 2.0 | 100 | 25 | **75** | 25 | P,U | 5 | 20 | 80 |
| S5 | 40 | 3.0 | 120 | 30 | **90** | 30 | P,U | 5 | 20 | 80 |

### B. Test — sub-study mật độ d (fix n=40; variant U; M=5)
| Cell | n | d | \|A\| | /lớp | **\|A_r\|** | \|A_nr\| | seeds | #inst |
|---|---|---|---|---|---|---|---|---|
| D1 | 40 | 1.5 | 60 | 15 | **45** | 15 | 20 | 20 |
| D2 | 40 | 2.0 | 80 | 20 | **60** | 20 | 20 | 20 |
| D3 | 40 | 2.5 | 100 | 25 | **75** | 25 | 20 | 20 |
| D4 | 40 | 3.0 | 120 | 30 | **90** | 30 | 20 | 20 |

### C. Test — quét fleet M (fix \|A\|=80, \|A_r\|=60; variant P & U)
| Cell | M | \|A_r\| | variant | seeds | #inst | ghi chú |
|---|---|---|---|---|---|---|
| F1 | 1 | 60 | P,U | 20 | 40 | = HCPP single-vehicle |
| F2 | 2 | 60 | P,U | 20 | 40 | |
| F3 | 3 | 60 | P,U | 20 | 40 | = gốc Hà |
| F4 | 5 | 60 | P,U | 20 | 40 | |
| F5 | 7 | 60 | P,U | 20 | 40 | |
| F6 | 10 | 60 | P,U | 20 | 40 | |

### D. Test OOD — đổi topology (giữ F2–F5; fix \|A_r\|≈60, M=5; P & U)
| Cell | topology | nguồn | \|A_r\| | seeds | #inst |
|---|---|---|---|---|---|
| O1 | unit-square | F1 gốc (in-distribution) | 60 | 20 | 40 |
| O2 | OSM city A | đường thật | ~60 | 20 | 40 |
| O3 | OSM city B | đường thật khác | ~60 | 20 | 40 |
| O4 | clustered | synthetic cụm | 60 | 20 | 40 |

### E. Training (on-the-fly, unit-square F1)
| Tham số | Giá trị (random mỗi instance) |
|---|---|
| n (nodes) | {20, 30, 40, 50} |
| d (mật độ) | {1.5, 2.0, 2.5, 3.0}, lọc `n·d` sao cho **\|A_r\| ≤ 100** |
| \|A_r\| dải | ~21 → ~93 (≤100) |
| p (classes) | **3** (cố định, ¼-split) |
| M (fleet) | {2, 3, 5, 7} random |
| variant | 1 model / variant (P hoặc U) |
| #instances | ~1M on-the-fly; batch **bucket theo n** |

**Tổng test ví dụ:** A(400) + B(80) + C(240) + D(160) ≈ **880 instance** (≥20 seed/cell), tất cả
`|A_r| ≤ 100` → train/test fit gọn 1×4090.

**Cách đọc nhanh:** cột `|A_r|` đều ≤100 ✓; `|A| = /lớp × 4` (¼-split) ✓; 3 lớp ưu tiên bằng nhau ✓;
A = quét **size**, B = quét **mật độ d**, C = quét **số xe M**, D = quét **phân phối topology**.
