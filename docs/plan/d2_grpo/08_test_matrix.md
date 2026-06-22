# Bảng cổng test gộp — d2_grpo

> Mỗi phase chỉ qua khi **toàn bộ** cổng test của nó + các phase trước **xanh**.
> ```bash
> uv run python -m unittest discover -s tests -p "test_*.py" -v
> ```

## Lưu ý xuyên suốt (bất biến CỨNG)

- **KHÔNG phá 90 unit test hiện có.** Default mọi cờ = **path cũ** (`--algo ppo`, `reward_mode='scalar'`,
  selector lex là free-win không retrain) ⇒ suite cũ xanh suốt.
- **Scheduler `Φ` (`common/scheduler.py`), global-class P mask (`env/env.py:82-93`), policy M-agnostic =
  GIỮ NGUYÊN.** D2 chỉ đổi **tín hiệu học (GRPO rank) + selector (lex best-of-K) + đo**.
- **Zero T_1 regression** ở mọi nơi: selector lex không chọn T_1 tệ hơn argmin cũ; LS chỉ nhận khi T_1 không
  tăng; win-rate đo phải báo **T_1 regression count = 0**.
- **Mỗi phase kèm "rollout smoke"** (`env.reset → policy → reward → backward`, finite) ở **variant `P`** rồi
  lặp **`U`**.
- **`run_parallel` cần batch ≥ num_epochs(=10)** (`common/ops.py:35`, `env/env.py:147,153`): GRPO batch hiệu
  dụng = `B_instances × K` ⇒ test đặt `B×K ≥ 10` (vd B=2,K=8).
- **A/B + rollback:** mỗi đổi sau 1 cờ; tắt cờ ⇒ hành vi cũ. **Mỗi phase = 1 commit độc lập.**
- File test mới: `test_lex_selector.py` (P1), `test_vector_reward.py` (P2), `test_grpo.py` (P3),
  `test_train_config.py` (P4), `test_within_class_ls.py` (P5), mở rộng `test_eval_grid.py`/`test_stats.py` (P6).

## Ma trận

| Phase | Bất biến phải giữ | Test trọng yếu (PASS mới qua) |
|---|---|---|
| **0** Scope | — (quyết định) | Chốt **K=8**, công thức **centered lex rank** + dấu, bỏ-critic-sau-cờ, `reward_mode`, A/B/rollback, ràng buộc `B×K≥10` + layout batchify |
| **1** Lex selector | Scheduler/mask/policy nguyên; **no T_1 regress** | (1) ⭐ lex argmin đúng (ví dụ tay); (2) ⭐ **T_1 chọn == min** (no-regress); (3) tie-break thật (lex ≤ argmin-cũ theo T_2,T_3); (4) smoke selector (skippable) |
| **2** Vector reward | `calc_reward`/Scheduler nguyên; default scalar=cũ | (1) ⭐ shape (scalar→(B,1), vector→(B,3)); (2) ⭐ **scalar == -(vector·w)**; (3) ⭐ vector finite/≥0; (4) default=scalar; (5) ⭐ rollout smoke PPO cũ {P,U}; (6) suite cũ xanh |
| **3** GRPO core | path PPO cũ byte-identical; critic giữ trong optim, không vào loss GRPO | (1) ⭐ **rank tay** (best→adv max, dấu đúng, mean≈0); (2) ⭐ **layout batchify/unbatchify** (B,K); (3) ⭐ adv finite + zero-mean/group; (4) ⭐ **backward: policy có grad, critic không**; (5) ⭐ **PPO cũ vẫn chạy** (critic có grad); (6) ⭐ rollout smoke GRPO {P,U}, `B×K≥10`; (7) suite cũ xanh |
| **4** Train config | default ppo/scalar; `nohup→logs/` | (1) ⭐ parse `--algo/--group_size`; (2) ⭐ **auto reward_mode** (grpo→vector); (3) ⭐ build+`shared_step` grpo finite; (4) ⭐ per-obj metrics `T1/T2/T3_mean`; (5) ⭐ ppo build chạy (A/B); (6) smoke {ppo,grpo}×{P,U}; smoke train: T_2/T_3↓, T_1 phẳng |
| **5** Local search *(tùy chọn)* | within-class ⇒ P-precedence giữ; **no T_1 regress**; KHÔNG ở train | (1) ⭐ precedence giữ; (2) ⭐ no-regress lexicographic (T_1 không tăng); (3) ⭐ idempotent/hội tụ; (4) cờ off=no-op; (5) smoke winner (skippable) |
| **6** Eval & measure | cùng selector lex cho cả 2 nhánh A/B; **T_1 regress=0** | (1) ⭐ **win-rate lexicographic** đúng + regression count; (2) ⭐ zero-T_1-regress synthetic; (3) grid rows + monotone-M (schema không vỡ); (4) best-of-K yield shape (đơn điệu mềm); (5) stats p∈[0,1], gap dấu đúng; (6) suite cũ xanh |

## Quan hệ với dynamic_plan (TRỰC GIAO)

- **dynamic_plan** đổi *Scheduler/M* (M-agnostic policy + `Φ` multi-trip; M = eval-time param).
- **d2_grpo** đổi *tín hiệu học (weighted-sum → GRPO lex rank) + selector + đo*. **Không** đụng Scheduler,
  P mask, hay M-agnostic.
- ⇒ Hai plan **độc lập, ghép được**: GRPO train ra policy tốt hơn về T_2/T_3; Scheduler vẫn map (α,M)→routes
  như dynamic_plan Phase 1 đã làm. Reward GRPO **vẫn gọi cùng `calc_reward`→Scheduler** (chỉ khác: lấy
  T-vector ra rank thay vì scalarize).

## Checklist tổng

- [ ] Phase 0: chốt K/rank/dấu/cờ/rollback + ràng buộc B×K≥10 & layout batchify.
- [ ] Phase 1: lex best-of-K selector (`run_grid.py`+`rl_hyb.py`) → commit.
- [ ] Phase 2: vector reward mode (`env.py`, sau cờ) → commit.
- [ ] Phase 3: GRPO core (`ppo.py`, sau `--algo grpo`, bỏ critic term) → commit.
- [ ] Phase 4: train config (`train.py/sh`) + per-obj log + smoke train → commit.
- [ ] Phase 5 *(tùy chọn)*: within-class LS trên winner → commit.
- [ ] Phase 6: per-obj curve + lex win-rate + gap-to-LP + best-of-K yield → commit.
- [ ] Giữ **90 test cũ xanh** xuyên suốt; mọi cờ default = path cũ.
