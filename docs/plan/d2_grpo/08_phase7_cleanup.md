# Phase 7 — Dọn **deprecated + dead code**

> Mục tiêu: xoá code chết/hỏng và (sau khi D2 thắng) khâu weighted+critic đã deprecated. Chia **2 mức**:
> **(A) dead code rõ ràng** — làm BẤT KỲ LÚC NÀO (độc lập GRPO); **(B) deprecated** — chỉ sau khi GRPO
> validate xong (Phase 6) để không mất đường A/B/rollback.

## 7.A — Dead code rõ ràng (an toàn, làm sớm được)

Đã xác nhận bằng grep:

| Mục | Vị trí | Hành động |
|---|---|---|
| File **comment 100%** (0 dòng code thật) | `common/intra.py` (90 dòng), `common/inter.py` (137 dòng) | **XOÁ file** |
| Import **thừa** `lsRL` (chỗ dùng đã comment) | `common/cal_reward.py:3` | xoá dòng import |
| Debug **comment** `# print/# exit` | `env/env.py` (6 dòng quanh `get_action_mask`/`get_reward`) | xoá |
| Import **HỎNG** (hàm không tồn tại) | `baseline/meta.py:6,8,9` (`run_parallel2`, `get_Ts`, `ls`) | **⚠️ baseline ngoài scope D2** — chỉ GHI CHÚ "baseline non-runnable", sửa khi hồi sinh baseline; KHÔNG đụng ở D2 |
| (tùy chọn) typo tên file | `policy/decode_stragegy.py` → `decode_strategy.py` | rename + sửa import — **tùy chọn**, vì đụng import (rủi ro nhỏ) |

> ⚠️ **GIỮ** `from common.ops import gather_by_index` ở `env/env.py:7` — **vẫn dùng** ở `step` (`:43`),
> KHÔNG xoá (chỉ chỗ dùng trong mask cũ đã bỏ).

### ✅ Cổng test 7.A
- [ ] `grep -rn "intra\|inter" common/ tests/ baseline/` không còn tham chiếu tới file đã xoá (ngoài comment lịch sử).
- [ ] `uv run python -m unittest discover -s tests -p "test_*.py"` **xanh** (xoá dead code không được vỡ gì).
- [ ] `uv run python -c "import common.cal_reward, env.env"` import sạch.
- [ ] Commit "D2 Phase 7A: remove dead code (intra/inter.py, unused imports, debug comments)".

## 7.B — Deprecated weighted+critic (CHỈ sau khi GRPO validate — Phase 6)

Sau khi Phase 6 chứng minh GRPO ≥ weighted (win-rate, 0 regress T_1), khâu weighted-sum + critic trở thành
**deprecated**. Lúc đó (và CHỈ lúc đó) dọn:

| Mục | Vị trí | Hành động |
|---|---|---|
| Đường reward scalar weighted | `env/env.py` `get_reward` nhánh `reward_mode='scalar'` + `obj_weights` | xoá nhánh scalar; `get_reward` chỉ còn vector |
| Critic | `rl/critic.py`, tham chiếu trong `rl/ppo.py` | nếu **bỏ hẳn PPO** → xoá; nếu **giữ PPO** làm baseline → để nguyên |
| `--algo ppo` path | `train.py`, `rl/ppo.py` | giữ nếu cần A/B vĩnh viễn; xoá nếu chốt chỉ GRPO |

> **Quy tắc:** chỉ xoá khi **không còn ai cần A/B**. Trước đó GIỮ để rollback. Ghi rõ trong commit "đã validate
> GRPO ở Phase 6, mục X".

### ✅ Cổng test 7.B
- [ ] Phase 6 PASS (GRPO ≥ weighted, 0 regress T_1) — **điều kiện tiên quyết**.
- [ ] Sau khi xoá: `unittest discover` xanh; smoke train GRPO (`--algo grpo`) vẫn EXIT=0.
- [ ] Không còn tham chiếu `obj_weights`/`reward_mode='scalar'` (nếu đã chốt bỏ).
- [ ] Commit "D2 Phase 7B: drop deprecated weighted+critic path (GRPO validated)".

---

> **Lưu ý:** 7.A có thể làm **ngay sau Phase 0** (độc lập, giảm nhiễu khi đọc code). 7.B **bắt buộc** sau
> Phase 6. Tách 2 commit.
