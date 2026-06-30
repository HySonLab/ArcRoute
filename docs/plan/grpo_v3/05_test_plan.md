# Test Plan tổng hợp — GRPO v3

## Status: PLANNED

Tổng hợp TẤT CẢ test gate. Dùng built-in `unittest` (không pytest, theo `CLAUDE.md`).

```bash
uv run python -m unittest discover -s tests -p "test_*.py"   # toàn bộ
uv run python -m unittest tests.test_phase1 -v               # 1 file
```

---

## 1. Bảng tổng hợp test gate

| # | Test name | Phase | File | Pass criterion (numerical) | Blocking? |
|---|---|---|---|---|---|
| 1 | `TestJointMxNInfeasibility` | P1 | `test_phase1.py` | 100 random instance, decode tới done: KHÔNG step nào all-inf logits | Yes |
| 2 | `TestLogProbCorrectness` | P1 | `test_phase1.py` | Σlog_p_step == log_p(traj) (atol 1e-5) | Yes |
| 3 | `TestMaskCorrectness::test_served` | P1 | `test_phase1.py` | sau (m*,j*): served[j*]=True; không chọn lại j* | Yes |
| 4 | `TestMaskCorrectness::test_capacity` | P1 | `test_phase1.py` | không step nào chọn arc vượt capacity xe m* | Yes |
| 5 | `TestJointShapes` | P1 | `test_phase1.py` | logits shape `(B,M,N+1)`; flat `(B,M*(N+1))`; m*,j* split đúng | Yes |
| 6 | `TestGRPOIntegrationP1` | P1 | `test_grpo.py` | train 2 step no crash, grads finite, entropy finite | Yes |
| 7 | `TestBenchmarkP1` | P1 | `test_phase1.py` | epoch 50: mean T1 ≤ mean T1_ILS (5 instance) | No (perf) |
| 8 | `TestCurriculumSwitch` | P2 | `test_phase2.py` | callback đổi buckets sau PATIENCE epoch plateau | Yes |
| 9 | `TestHRDAEvalStep` | P2 | `test_grpo.py` | eval LS reward finite, ≤ RL-only lex | Yes |
| 10 | `TestBenchmarkMedium` | P2 | `test_eval_grid.py` | RL+HRDA mean T1 ≤ ILS mean T1 (medium instances) | No (perf) |
| 11 | `TestScaleNoCrash` | P2 | `test_phase2.py` | train 2 step bucket large no crash | Yes |

---

## 2. Thứ tự chạy & dependencies (CI rule)

```
Phase 1: tests 1–7   ── blocking 1–6 PASS trước khi code Phase 2 (7 = perf, non-block)
Phase 2: tests 8–11  ── blocking 8,9,11 PASS; 10 = target perf cuối
```

**CI rule:** Phase 2 không implement nếu blocking gate Phase 1 (test 1–6) chưa xanh.
Perf gate (7, 10) không chặn merge nhưng phải log số liệu vào `00_overview.md`.

Regression toàn cục sau mỗi phase:
```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```
Test hiện có phải vẫn xanh: `test_grpo.py`, `test_multisize.py`, `test_train_config.py`.

---

## 3. Dữ liệu test

| Dùng cho | Nguồn | Ghi chú |
|---|---|---|
| Unit (synthetic) | `env.generator.generate_dataset(B, n, n, M, num_workers=0)` | giống `test_grpo.py:_synthetic_dl` (line 33) |
| Joint decode toy | tay: M=2, N=6, adj/service/demand/clss đặt cứng | manual-calc T để verify |
| Benchmark | `.npz` OSM (vd `data/osm_test/20_30/*.npz`) | `import_instance` (`ops.py:178`) |
| Curriculum | `env.dataset(data_dir, buckets=[...])` (`env.py:143`) | |

---

## 4. Lưu ý implement test

- **Test 1 (infeasibility)**: sau flatten + add mask, assert
  `torch.isfinite(logits_flat).any(dim=-1).all()` tại mọi step trong rollout.
- **Test 2 (log_p)**: chạy 1 rollout `store_all_logp=True`, lấy `log_p_flat` mỗi
  step, gather theo `action_flat`, sum; so với `get_log_likelihood` output.
- **Test 3–4 (mask)**: dùng instance toy M=2 N=6, trace từng step, assert constraints.
- **Test 6 (integration)**: tái dùng `_fit_one_step` / `_wire_and_batch`
  (`test_grpo.py:46,235`) — khung Trainer CPU đã có sẵn.
