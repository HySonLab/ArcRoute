# Phase 0 — Scope & quyết định (chốt trước khi sửa code)

> Mục tiêu: chốt **K, công thức rank, bỏ-critic-vs-cờ, biểu diễn vector reward, cái gì đóng băng**, và
> **chiến lược A/B + rollback**, trước khi đụng code. Tham chiếu [`00_overview`](00_overview.md).

## 0.1 — ✅ ĐÃ CHỐT: kiến trúc D2

Stack: **policy (GRPO) → Scheduler `Φ` (GIỮ NGUYÊN) → eval lexicographic best-of-K → within-class LS**.
D2 đổi **3 chỗ**: (a) selector best-of-K (Phase 1), (b) reward env lộ T-vector (Phase 2), (c) advantage
GRPO theo rank (Phase 3). **Không** đổi Scheduler, global-class P mask (`env/env.py:82-93`), hay tính
M-agnostic của policy.

## 0.2 — ✅ Quyết định cụ thể

| Câu | Chốt | Lý do / vị trí code |
|---|---|---|
| **K (group size)** | **mặc định 8**, dải 8–16; cờ `--group_size` | dùng `batchify` (`common/ops.py:76`) nhân td K lần (POMO). ⚠️ ràng buộc `run_parallel` (xem 0.4) |
| **Công thức rank** | xếp K bằng `np.lexsort((T_3,T_2,T_1))` → rank `r∈{0..K-1}` → **center+scale**: `adv = (r - (K-1)/2) / ((K-1)/2)` ∈ [-1,1], mean≈0 | scale-free, zero-mean (group mean = baseline), unit-scale; **rank tốt nhất → adv âm nhất hay dương nhất? xem 0.3** |
| **Critic** | **bỏ value term** (`vf_lambda→0`), critic không tham gia loss; **giữ object critic** trong code (khỏi vỡ `configure_optimizers`) nhưng **không backward qua nó** ở path GRPO | `rl/critic.py` giữ nguyên; `rl/ppo.py:226-233` value_loss = 0 khi `--algo grpo` |
| **Biểu diễn vector reward** | `get_reward` trả **(B,3) T-vector** khi cờ `reward_mode='vector'`; mặc định/`'scalar'` giữ `-(rs*w).sum` cũ | `env/env.py:150-159`; `common/cal_reward.py` đã trả full vector (`return torch.tensor(rs)`) |
| **Đóng băng** | Scheduler `Φ`, P mask, encoder/decoder, M-agnostic, calib `Q=Σq/3`, p=3 | — |
| **A/B** | cờ `--algo {ppo,grpo}` chọn path; `--reward_mode {scalar,vector}`; default = path cũ (ppo+scalar) | `train.py`, `rl/ppo.py` |
| **Rollback** | mỗi đổi sau 1 cờ; tắt cờ ⇒ về hành vi cũ **byte-identical**; mỗi phase 1 commit | — |

## 0.3 — ⚠️ Quy ước dấu reward/rank (chốt rõ để khỏi sai dấu)

- `calc_reward` trả **T-vector dương** (completion time; **nhỏ hơn = tốt hơn**), `common/cal_reward.py:35`.
- PPO **maximize reward** (`monitor="val/reward", mode="max"`, `train.py:91`); surrogate dùng `adv` mà
  **adv lớn = hành động tốt** (`rl/ppo.py:215-223`).
- ⇒ Rank phải để **nghiệm tốt (T thấp) có adv DƯƠNG**. Cụ thể: `order = np.lexsort((T_3,T_2,T_1))` cho
  **index nghiệm tốt→xấu**; nghiệm tốt nhất nhận rank cao nhất. Công thức chốt:
  - cho mỗi phần tử i tính `rank_i ∈ {0..K-1}` = **vị trí trong thứ tự GIẢM của độ tốt** (tốt nhất = K-1),
  - `adv_i = (rank_i - (K-1)/2) / ((K-1)/2)` → tốt nhất `+1`, tệ nhất `-1`, mean = 0.
- Test Phase 3 phải khoá dấu này bằng **ví dụ tay** (xem `04_phase3_grpo_core.md §gate`).

## 0.4 — ⚠️ Ràng buộc kỹ thuật phải tôn trọng

- **`run_parallel` cần batch ≥ num_epochs(=10)** để chia loader (`common/ops.py:35`, `env/env.py:147,153`
  gọi `num_epochs=10`). Với GRPO, batch hiệu dụng = `B_instances × K`. ⇒ **`B_instances × K ≥ 10`**;
  smoke test phải đặt B,K thoả (vd B=4,K=8 → 32 ✓). Ghi rõ trong cổng test mỗi phase.
- **`batchify` thứ tự**: `batchify(td, K)` lặp **mỗi instance K lần liên tiếp** theo
  `_batchify_single` (`common/ops.py:71-74`: `expand(repeats,...).view(s[0]*repeats,...)` → layout
  `(K, B) → (K*B)`? **PHẢI verify layout** ở Phase 3 và reshape `(B,K)` đúng trục trước khi rank —
  test rank dùng layout thật, không giả định).
- **`store_all_logp=True`** đã bật (`policy/policy.py:73`) → có logp đầy đủ để PPO ratio.

## 0.5 — Xoá giả định cũ

- ❌ Bỏ ý "tune `obj_weights` cho khéo" — GRPO rank thay thế hẳn weighted-sum ở path train.
- ✅ Giữ `obj_weights`/scalar reward **sau cờ** cho A/B + rollback (không xoá code cũ).

---

## ✅ Cổng Phase 0 (quyết định — không code)

- [ ] Chốt **K=8 (8–16)**, công thức **centered lexicographic rank** (0.2) + **quy ước dấu** (0.3).
- [ ] Chốt **bỏ value term sau cờ** (giữ object critic), `reward_mode` vector/scalar, `--algo {ppo,grpo}`.
- [ ] Chốt **A/B + rollback**: default = path cũ; tắt cờ ⇒ hành vi cũ.
- [ ] Ghi ràng buộc **`B×K ≥ 10`** (run_parallel) + **verify layout batchify** vào plan Phase 3.
- [ ] Xác nhận **đóng băng** Scheduler/P-mask/M-agnostic/p=3/Q.

> Sau khi chốt → [`02_phase1_lex_selector.md`](02_phase1_lex_selector.md).
