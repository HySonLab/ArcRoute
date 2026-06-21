# RL Dynamic — hiện trạng & việc muốn làm

> Ghi lại: model RL hiện tại đang "động" được những gì, còn cố định gì, và mục tiêu sắp tới là
> **động hóa `|A_r|` (size) và `M` (số xe)**. Liên quan: [[data.md]] (cấu hình dataset).

---

## 1. Hiện trạng

### ✅ Đã dynamic (model nhận như input, xử lý được giá trị bất kỳ)
| Khía cạnh | Cơ chế | Vị trí code |
|---|---|---|
| demand / service / traversal / adj | nạp dạng scalar qua `Linear(4)` + adj trong attention | `policy/init.py:ARPInitEmbedding` |
| **giá trị priority class** | `clss` đưa vào **dạng SCALAR** (không one-hot) → input không chốt cứng số lớp | `policy/init.py` |
| decode theo trạng thái | context cập nhật `current_node` + `capacity − used` mỗi bước | `policy/context.py:ARPContext` |
| số route nổi lên động | policy tự quyết khi quay depot (action=0) | `common/cal_reward.py:action_to_tours` |
| mask capacity / precedence (P,U) | che cung vượt tải & sai thứ tự lớp theo từng bước | `env/env.py:get_action_mask` |
| reload data động | regenerate instance train mỗi `reload_train_dataloader=4` epoch | `rl/ppo.py:291` |

### ❌ Chưa dynamic (cố định / không thực dùng)
| Khía cạnh | Hiện trạng | Nguyên nhân (code) |
|---|---|---|
| **size `\|A_r\|` / `n`** | cố định cả run lẫn batch | `adj (B,n,n)` ghép `torch.cat`; encoder `assert mask is None` ("Mask not yet supported") → không trộn size / không ragged |
| **số xe `M`** | lưu vào td nhưng **không dùng** | `calc_reward` bỏ qua `num_vehicle`; `refine_routes(max_vehicles=3)` hardcode & **không được gọi**; không có cap số route |
| số lớp `p` (ở reward) | hardcode 3 | `cal_reward` `pos_val=[1,2,3]` (input `clss` thì linh hoạt) |
| mật độ `d` | không có khái niệm | train generator lấy cung từ tổ hợp node, không mô hình hóa `d` |
| dynamic node embedding | tắt | `decoder.py:118 is_dynamic_embedding=False`, `StaticEmbedding`→`0,0,0` |
| đa phân phối topology | 1 kiểu (unit-square ngẫu nhiên) | train generator 1 distribution |

### Nhận định cốt lõi
Mọi **nội dung số** của instance đã động sẵn. Chỉ còn **2 trục cấu trúc** chưa hỗ trợ: **`|A_r|` (size)**
và **`M` (số xe)**. `p` để cố định 3 (theo công thức gốc Hà — xem [[data.md]]) nên không phải blocker.

---

## 2. Việc muốn làm

**Mục tiêu:** động hóa **size `|A_r|`** và **số xe `M`** → model nhận instance kích thước & số xe tùy ý.
Hai trục này **KHÔNG đối xứng** về khối lượng việc:

### 2.1 Dynamic size `|A_r|` — SỬA KIẾN TRÚC (khó nhất)
Không chỉ là "cho phép `n` thay đổi". Cần:
1. **Mask trong encoder**: bỏ `assert mask is None`; truyền padding mask qua các lớp MHA.
2. **Mask trước softmax** trong `attn_weights * adj`: vị trí padding → `−inf` (không chỉ nhân 0), để
   softmax không rò sang token padding.
3. **Batch ragged / padding**: sửa `collate_fn` (hiện `torch.cat`) để pad `adj`, features, mask về `n_max`.
4. **Chuẩn hóa**: `BatchNorm/InstanceNorm` (`encoder.py:Normalization`) phải **loại token padding** khỏi
   thống kê, nếu không mean/var bị lệch.
5. **Decoder + action mask**: bảo đảm bước decode & `env.get_action_mask` tôn trọng padding.
> Phương án thay thế (rẻ hơn): **bucket theo size** (mỗi batch 1 `n`) — không cần mask, nhưng không
> trộn size trong 1 batch. Hiện code đã chạy được kiểu này nếu sinh data per-size.

### 2.2 Dynamic số xe `M` — THÊM MỚI (không phải mở khóa)
`M` hiện là no-op. Muốn M thật sự là tham số điều khiển:
1. **Đưa M vào model** (context/global feature) để policy "biết" có mấy xe.
2. **Cap số route ở M** trong `env.get_action_mask` (chặn mở route thứ M+1).
3. **Reward theo đúng M xe**: `cal_reward` tính makespan trên M tour (hiện tính trên số tour tự sinh).
> ⚠️ Cân nhắc trước: với mục tiêu makespan + route tự nổi lên, **có thật sự cần M là input không?**
> Nếu bài toán không ép đúng M xe thì có thể bỏ; nếu cần báo cáo theo `M∈{2,5,...}` thì bắt buộc wire vào.

---

## 3. Lưu ý / rủi ro
- **Động kiến trúc ≠ tổng quát tốt**: cho phép size/M thay đổi chỉ là điều kiện cần; muốn generalize
  phải **train trên đa dạng size/M** (xem ladder & quét M trong [[data.md]] §5, §8).
- **BatchNorm + size đổi**: chuẩn hóa theo batch nhạy với padding & phân bố size → cân nhắc LayerNorm
  hoặc masked-norm.
- **`p` vẫn fixed ở reward** (`pos_val=[1,2,3]`): giữ p=3 theo Hà thì OK; đổi p phải sửa reward.
- Tôn trọng ràng buộc compute `O(n²)` và trần `|A_r| ≤ 100` (xem [[data.md]] §5.7).

---

## 4. Thứ tự đề xuất
1. **(Rẻ)** Bucket theo size + sinh data per-size → có ngay "đa size" mà không sửa kiến trúc.
2. **(Quyết định)** Chốt có cần `M` là input không. Nếu có → wire M (input + cap + reward).
3. **(Khó)** Dynamic size thật sự: mask encoder + ragged batch + masked-norm.
4. Train đa dạng size/M để generalize; đánh giá theo [[data.md]].
