# Plan tổng quan — Động hóa MODEL theo `M` (và `size`)

> Nguồn: [`docs/dynamic.md`](../../dynamic.md). Mục tiêu: biến model RL từ "cố định size & bỏ qua M"
> thành **nhận `M` (số xe) làm tham số điều khiển thật** và (tùy chọn) **size động trong 1 batch**.
>
> ⚠️ **Đây là plan về MODEL** (env + policy + reward + encoder), KHÔNG đụng khâu sinh data — data đã sẵn
> sàng (xem `docs/plan/data_plan/`: size ladder Phase 6, M-independent instance Phase 3). Test dùng lại
> **test grid `data/ood/`** + `eval/stats.py` đã có.

## Bối cảnh: cái gì đã động, cái gì chưa

Theo `dynamic.md §1`: mọi **nội dung số** của instance đã động (demand/service/traversal/adj/clss scalar).
Còn lại **2 trục cấu trúc**:

| Trục | Hiện trạng | Plan |
|---|---|---|
| **`M` (số xe)** | lưu vào td nhưng **no-op** | **Scheduler lo M** (Phase 1); policy **M-agnostic** (headline). M-conditioning = enhancement **hoãn** (Phase 2) |
| **`size \|A_r\|`** | đa size đã chạy qua **bucketing** (data_plan Phase 6) | Đường "rẻ" XONG; "thật" (trộn size/batch) = Phase 4 **tùy chọn** |
| `p` (số lớp) | hardcode 3 ở reward (`pos_val=[1,2,3]`) | **giữ 3** (theo Hà) — không trong plan |
| **`variant` P/U** | mask ưu tiên lớp trong `get_action_mask` (P chặn lớp thấp; U không) | **trực giao với M — mặc định `P`**; xem ⚠️ dưới |

### ⚠️ Trục `variant` P/U (trực giao với M) — **mặc định `P`**

- **`P` (priority/hierarchical = HDCARP đúng):** `env/env.py:get_action_mask` chặn phục vụ customer có lớp
  **thấp hơn** lớp hiện tại (`clss − clss_min < 0`). Đây là **mặc định của plan này**.
- **`U` (unrestricted = CARP thường):** bỏ ràng buộc thứ tự lớp.
- P/U **không phải trục M**. Vẫn phải kiểm ở cả hai: **mọi rollout-smoke chạy ở `variant='P'`** (mặc định)
  và lặp lại ở `'U'`. Lưu ý **Scheduler phải tôn trọng thứ tự lớp / hierarchy-level** khi tính `T_k` (P enforce
  precedence; U dùng hierarchy-level như paper định nghĩa) — đây là chỗ P/U ảnh hưởng **Scheduler**, không
  phải mask của policy M-agnostic.
- ⚠️ Lưu ý lệch default trong code: `train.py` đang default `U`, các baseline default `P`. Plan này
  **chốt `P`** → Phase 3 đổi `train.py`/`train.sh` sang `variant='P'`.

## ✅ Quyết định kiến trúc: **policy M-agnostic + Scheduler tách riêng** (headline)

Tách bài toán làm **2 thành phần** (theo notation paper: policy dựng chuỗi `α`, toán tử `Φ` map ra solution):

1. **Policy `π_θ` (encoder/decoder)** — dựng **một chuỗi arc theo ưu tiên**, **KHÔNG thấy M** (M-agnostic).
   **Train một lần.**
2. **Scheduler `Φ` (concept/module RIÊNG — `common/scheduler.py`)** — map `(chuỗi α, M) → routes của M xe`
   + tính mục tiêu phân cấp `(T_1,…,T_p)`. **M chỉ tham gia tại đây.**

> **"Train once, any fleet size M":** đổi M = **chạy lại Scheduler**, không train lại policy. Khớp commit
> `84ea3df` *"M is an eval-time parameter"*. Không sinh lại data (chỉ đổi khâu Scheduler/reward).

**M-conditioning** (cho policy *thấy* M để tỉa chuỗi theo M) = **enhancement, HOÃN lại** → [Phase 2](03_phase2_M_conditioning.md),
implement sau, kèm ablation so với M-agnostic.

### ✅ ĐÃ CHỐT: `Q = Σq/3 + 0.5` (paper gốc Ha 2024) + Scheduler **multi-trip**

Nguồn chuẩn **Ha 2024 tr.18**: *"three vehicles, `Q = Σ_{a∈A_r} q_a/3 + 0.5`"* → calib **KHÍT** (~3 route),
**code đã đúng, không đổi**. (Bản nháp `HDCARP_with_ML/main.tex:632` viết per-arc → **TYPO, phải sửa**.)

Q khít ⇒ M=2 không khả thi single-trip ⇒ **Scheduler chế độ multi-trip**, **tự suy biến về single-trip khi
K≤M** (M≥3):

| M | #trip tối thiểu | Scheduler | = single-trip? |
|---|---|---|---|
| **2** | 3 | 3 trip → 2 xe (1 xe 2 trip) | ❌ multi-trip |
| **3 / 5 / 7 / 10** | 3 | M segment → M xe | ✅ |

> **`Q` cố định (`/3`, không theo M) ⇒ policy rollout độc lập M ⇒ rollout MỘT lần, Scheduler chia/gán cho
> mọi M (không rollout lại)** — *"train once, M = eval-time param"* dạng sạch nhất. Khớp commit `84ea3df`.
> Không sinh lại data, không đổi `Q` code.

## Các phase

**Critical path (headline):** `0 → 1 → 3 → 5`. Phase 2 (M-conditioning) là **nhánh hoãn**; Phase 4 (size) tùy chọn.

| Phase | File | Nội dung | Bắt buộc? |
|---|---|---|---|
| **0** | [`01_phase0_scope.md`](01_phase0_scope.md) | Chốt M-agnostic + Scheduler; **`Q=Σq/3` (paper gốc) + Scheduler multi-trip** | ✅ quyết định |
| **1** | [`02_phase1_M_constraint.md`](02_phase1_M_constraint.md) | **Tách `Scheduler` thành module** (`common/scheduler.py`) — M chỉ vào đây; `calc_reward` mỏng lại | ✅ nền tảng |
| **2** | [`03_phase2_M_conditioning.md`](03_phase2_M_conditioning.md) | *(HOÃN)* Policy **thấy M** (context + critic) + ablation vs M-agnostic | ⏸ **hoãn — làm sau** |
| **3** | [`04_phase3_train_generalize.md`](04_phase3_train_generalize.md) | Train **M-agnostic** ("train once"); M quét trong **reward/Scheduler**, không vào policy | ✅ |
| **4** | [`05_phase4_dynamic_size.md`](05_phase4_dynamic_size.md) | Size động THẬT (mask encoder + ragged + masked-norm) | ⏹ **tùy chọn** |
| **5** | [`06_phase5_eval.md`](06_phase5_eval.md) | Eval & report theo M, size trên test grid + ablation scheduler/M-cond | ✅ |

Bảng cổng test gộp: [`07_test_matrix.md`](07_test_matrix.md).

## Ràng buộc kế thừa (giữ nguyên)

- Trần `|A_r| ≤ 100`, attention `O(n²)`, không flash (data_plan §5.7).
- Metric train≈test (floyd-warshall thưa) — **không phá** khi sửa model.
- `p=3` cố định.
- **`variant='P'` mặc định** (HDCARP đúng); test phải xanh ở P, và không vỡ ở U.

## Quy tắc "qua phase" (như data_plan)

1. Mỗi phase có cổng test riêng — **PASS hết mới qua phase sau**.
2. `uv run python -m unittest discover -s tests -p "test_*.py"` phải xanh.
3. Mỗi phase = 1 commit độc lập.
4. Đổi env/reward/policy dễ làm **lệch train≈test** hoặc **vỡ rollout** → mỗi phase có **smoke test rollout**
   (encoder+decoder+reward+backward) như đã làm ở data_plan Phase 0/6.
