# Phase 5 — **Within-class local search** trên winner best-of-K *(TÙY CHỌN)*

> Mục tiêu: mạnh hoá khâu **polish within-class** và chỉ chạy nó trên **nghiệm thắng lexicographic best-of-K**
> ở eval (không trong vòng train). Cải thiện T_2/T_3 (và T_1 nếu được) **không train lại**. Phụ thuộc Phase 1
> (selector) + Phase 6 (đo). **Tùy chọn** — bỏ qua được nếu gain nhỏ. Code: `common/local_search.py`,
> `eval/run_grid.py`, `baseline/rl_hyb.py`.

## 5.1 — Hiện trạng (đã xác nhận)

- `common/local_search.py`: chỉ có **`lsRL`** (`:58-60`) → `intraU` = **intra-route 2-opt** (`once_intraU`
  flip đoạn, `:9-27`), polish **trong từng tour**, tôn trọng prefix theo lớp (`intraU` cắt theo `pos_ids`,
  `:29-56`). Inter-route `ls` **bị bỏ/comment** (không tồn tại bản inter).
- `common/cal_reward.py:36-37`: nhánh `local_search` **đang comment** (`# tours = lsRL(...)`).
- `env/env.py:144 get_objective(..., local_search=True)` truyền cờ nhưng `calc_reward` chưa dùng (Scheduler
  path). ⇒ LS hiện **không thực sự chạy** ở đường eval mới.

## 5.2 — Thay đổi cụ thể (chỉ ở EVAL, chỉ trên WINNER)

> **Nguyên tắc:** LS **đắt** ⇒ chỉ chạy **1 lần** trên nghiệm thắng best-of-K (sau Phase 1 lexsort),
> **KHÔNG** trên cả K, **KHÔNG** trong train (giữ GRPO sạch + nhanh). Within-class ⇒ **không phá precedence
> P** (chỉ hoán vị **trong cùng lớp**).

1. **Polish hàm** `common/local_search.py`: giữ `intraU` (intra 2-opt within-class — đã đúng). *(Tùy chọn
   nâng cao:* thêm inter-route within-class move — di chuyển arc **cùng lớp** giữa các route của cùng lớp —
   nhưng **chỉ** nếu chứng minh không vi phạm P và cải thiện đo được; nếu rủi ro, **giữ intra-only**.)

2. **Gắn vào eval winner:**
   - `eval/run_grid.py` (sau lex `best` ~`:111`): với winner action, gọi LS within-class rồi **re-evaluate**
     `T` qua `calc_reward`/Scheduler; nếu (lexicographic) tốt hơn thì thay. Cờ `--local_search`.
   - `baseline/rl_hyb.py` (sau `idx` lex ~`:37`): tương tự trên `out['actions'][idx]`.
   - ⚠️ **Re-evaluate bắt buộc**: LS đổi action ⇒ phải tính lại `T` qua **cùng Scheduler** rồi so lexicographic;
     chỉ nhận khi **không regress T_1** (giữ bất biến no-T_1-regression).

3. **KHÔNG** bật LS trong `get_reward`/train (GRPO không cần — rank tự lo).

## 5.3 — Gắn cờ
- `--local_search` (default off) ở `eval/run_grid.py`/`baseline/rl_hyb.py`. Off ⇒ hành vi Phase 1 nguyên.

---

## ✅ Cổng test Phase 5

**File test mới:** `tests/test_within_class_ls.py`.

1. **⭐ Within-class không phá precedence:** dựng tour có lớp [1,1,2,2,3]; sau LS, **thứ tự lớp giữ
   monotone** (không có lớp cao chen trước lớp thấp) → P-feasible.
2. **⭐ Không regress (lexicographic):** trên winner, `T_after` **≤** `T_before` theo lexsort
   (T_1,T_2,T_3); đặc biệt `T_1_after ≤ T_1_before` (zero T_1 regression).
3. **⭐ Idempotent/hội tụ:** LS chạy 2 lần == 1 lần (đã tối ưu cục bộ), không vòng lặp vô hạn (`it<100` guard
   như `intraU` `:52`).
4. **Cờ off = no-op:** `--local_search` off → winner **không đổi** so Phase 1.
5. **Smoke eval winner** *(skip nếu thiếu ckpt/data)*: best-of-K + LS → `(3,)` finite, T_1 không tăng.
6. **Suite cũ xanh.**

### Lệnh
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python -m unittest tests.test_within_class_ls -v
```

### Checklist
- [ ] `common/local_search.py`: within-class polish (intra-only an toàn; inter chỉ nếu chứng minh P-safe).
- [ ] `eval/run_grid.py` + `baseline/rl_hyb.py`: chạy LS **trên winner** + **re-evaluate** + nhận khi lex-tốt-hơn; cờ `--local_search`.
- [ ] **Không** bật LS trong train.
- [ ] `tests/test_within_class_ls.py`: ⭐ precedence, ⭐ no-regress, ⭐ idempotent, off=no-op, smoke.
- [ ] `unittest discover` xanh.
- [ ] Commit.

### Commit message
```
D2 Phase 5 (optional): within-class local search on best-of-K winner

Polish only the lexicographic best-of-K winner at eval (not in training, not
on all K), within-class moves so precedence-P holds; re-evaluate via Scheduler
and accept only on lexicographic improvement (no T_1 regression). Flagged.
```
