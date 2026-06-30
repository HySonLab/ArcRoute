# Phase 3: Scale (curriculum + benchmark)

## Status: PLANNED

Mục tiêu: scale policy Joint M×N (Phase 2) lên medium/large bằng curriculum, bật
HRDA trong eval, và benchmark đầy đủ vs ILS / ACO / EA.

---

## 1. Curriculum small → medium → large

Dữ liệu đã có cơ chế bucket: `env.dataset(..., buckets=...)` (`env.py:143`) +
`SizeBucketBatchSampler` + `train_buckets` (`base.py:36`, `grpo.py:135`). Curriculum
= đổi `train_buckets` theo tiến độ.

| Stage | buckets (`n_req` range) | M range | Khi nào chuyển |
|---|---|---|---|
| S1 small | `20_30` | 2–5 | val/lex_best plateau |
| S2 medium | `20_30`, `30_40` | 2–8 | val/lex_best plateau |
| S3 large | `+ 40_60` | 2–10 | hết epoch budget |

### Plateau detection (criteria chuyển stage)

Theo dõi `val/lex_best` (`grpo.py:311`, đã log mỗi epoch). Chuyển stage khi:

```
no_improve_epochs ≥ PATIENCE (vd 10)
  với improvement = (lex_best_t - best_so_far) > MIN_DELTA (vd 1e-3)
```

Cài bằng một Lightning `Callback` (`on_validation_epoch_end`) đọc
`trainer.callback_metrics["val/lex_best"]`, đếm epoch không cải thiện, rồi cập nhật
`model.data_cfg["train_buckets"]` + gọi lại `load_dataloader()` (hoặc reload
trainer datamodule). Không can thiệp vào loss.

---

## 2. HRDA trong eval (`_eval_step`, `grpo.py:294`)

Hiện `_eval_step` chỉ greedy decode → reward. Thêm refine:

```python
def _eval_step(self, batch, phase, dataloader_idx):
    with torch.no_grad():
        td0 = self.env.reset(batch)
        out = self.policy(td0.clone(), self.env, phase=phase)
    reward = out["reward"]                              # (B, P) — RL-only
    if getattr(self, "eval_local_search", True):
        # refine từng instance: ls() trên (action, action_vehicle) → re-score Scheduler
        reward = self._hrda_rescore(td0, out["actions"], out["action_vehicle"], reward)
    ... # log T_k_best, lex_best như cũ
```

`_hrda_rescore`: gọi `local_search.ls(vars, variant, actions=[...])`
(`local_search.py:254`) trên best action per instance, re-score bằng Scheduler
`_completion_times`, **giữ kết quả lex tốt hơn** (LS không được làm xấu). Tái dùng
hệ wiring đã mô tả ở `01_phase0_quick_win.md` §P0b nhưng batched.

Lưu ý: HRDA chỉ ở **eval/test/inference**, KHÔNG ở train (train giữ on-policy để
gradient sạch).

---

## 3. Benchmark plan

| Solver | Nguồn | Cấu hình |
|---|---|---|
| RL Phase 2 + HRDA | `solvers/rl.py` (`RLHCARP`, đã wire LS) | num_sample 100, greedy+sample |
| ILS | `solvers/meta.py` (`InsertCheapestHCARP`) | 500 iter (baseline hiện tại) |
| ACO | `solvers/` (test_aco.py) | mặc định |
| EA | `solvers/` (test_ea.py) | mặc định |
| LP/MIP | `solvers/lp.py` (gurobi) | small only (ground truth khi giải được) |

Sets: small (`20–30`), medium (`30–40`), large (`40–60`), mỗi set ≥ 20 instance,
M ∈ {2,5,10}. Báo cáo: mean/median T1 (primary), T2/T3, %instances RL ≤ ILS, wall
time. Dùng `tests/test_eval_grid.py` làm khung benchmark (đã tồn tại).

---

### Test Gate

| Test name | File | Pass criterion | Blocking? |
|---|---|---|---|
| `TestCurriculumSwitch` | `tests/test_phase3.py` | plateau callback đổi `train_buckets` đúng sau PATIENCE epoch không cải thiện | Yes |
| `TestHRDAEvalStep` | `tests/test_grpo.py` | `_eval_step` với `eval_local_search=True`: reward finite, ≤ RL-only (lex) | Yes |
| `TestBenchmarkMedium` | `tests/test_eval_grid.py` | RL+HRDA mean T1 ≤ ILS mean T1 trên medium set | No (perf, target) |
| `TestScaleNoCrash` | `tests/test_phase3.py` | train 2 step trên bucket large không crash (OOM guard) | Yes |
</content>
