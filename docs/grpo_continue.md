# GRPO / D2 — Handoff để tiếp tục ở session khác

> Cập nhật: 2026-06-23. Tóm tắt trạng thái D2 (vector reward + GRPO lexicographic rank),
> kết quả A/B đã đo, và việc cần làm tiếp. Plan đầy đủ: [`docs/plan/d2_grpo/`](plan/d2_grpo/).

---

## 1. TL;DR — đã chốt gì

- **Kiến trúc:** policy **M-agnostic** (chỉ sinh thứ tự arc) → **Scheduler `Φ`** (`common/scheduler.py`, map
  order+M → routes + (T₁,T₂,T₃)) → reward. M chỉ vào Scheduler. Mục tiêu **lexicographic** (T₁ ≺ T₂ ≺ T₃).
- **D2 = đổi tín hiệu học** từ weighted-sum `−(T·w)` sang **GRPO rank lexicographic** (xếp K sample/instance
  theo `lexsort(T₃,T₂,T₁)`, advantage = centered rank, **bỏ critic**). KHÔNG đụng Scheduler/mask/policy.
- **Kết luận A/B (đã đo, xem §4): GRPO THẬT SỰ tốt hơn weighted** — ~2-2.8% cả 3 lớp, **lợi thế tăng theo
  scale**, cải thiện cả T₁. → **Quyết định: dùng GRPO cho model cuối**, giữ PPO làm baseline A/B.

---

## 2. Đã implement + commit (suite 122 xanh, `rl/ppo.py` BẤT BIẾN)

| Phase | Commit | Nội dung |
|---|---|---|
| 1 | `8d40f56` | **Lexicographic best-of-K selector** (`eval/run_grid.py`, `baseline/rl_hyb.py`): `lexsort((T3,T2,T1))` thay vì T₁-only. **Free win, không retrain.** |
| 2 | `0932674` | `CARPEnv(reward_mode={scalar,vector})`: `get_reward` lộ (B,3) T-vector khi `vector`; scalar = cũ byte-identical. |
| 3 | `0d805a4` | **`rl/grpo.py` MỚI** (`class GRPO(PPO)`): batchify K → `unbatchify` (B,K,3) → centered lex rank → adv=rank, **critic-free**. `rl/ppo.py` không đổi. |
| 4 | `1fbade8` | `train.py` chọn lớp `--algo {ppo,grpo}` + `--group_size`; `train.sh` `ALGO`/`GROUP_SIZE`; log T₁/T₂/T₃ riêng. |
| 6 | `6d06d1e` | `eval/stats.py:win_rate/paired_win_rate` (lexicographic + đếm T₁-regression); `run_grid` cột `algo` + yield-curve. |
| 7A | `8dcbb28` | Xoá dead code (`common/intra.py`, `common/inter.py`, import `lsRL` thừa, debug comments). |
| — | `5835152` | `grpo.py` log **held-out best-of-K** `val/T1_best,T2_best,T3_best` (vào `lightning_logs/.../metrics.csv`). |

**Bất biến đã giữ:** `git diff <phase1> HEAD -- rl/ppo.py` rỗng; default `--algo ppo` + `reward_mode scalar`
= path cũ; Scheduler/global-class-P-mask/M-agnostic **đóng băng**.

**Bỏ qua/hoãn:** Phase 5 (within-class LS) **skip** (tùy chọn, bridging policy-tour↔Scheduler-route phức tạp).
**Phase 7B** (xoá weighted+critic) **HOÃN** — chỉ làm SAU khi full-scale A/B xác nhận GRPO thắng.

---

## 3. Validation (cơ chế đúng — đã xác nhận)

GRPO-only, 30 epoch, model tí hon. Held-out lex-best:
- T₁ bão hoà sớm (~epoch 7), rồi **T₂ tăng tốc giảm** (epoch 18→28: 21.10→20.72) — **chữ ký lexicographic**:
  khi T₁ hoà trong nhóm K → rank bị T₂ phá hoà → policy mới học T₂. Weighted thì T₂/T₃ phẳng lì.
- 0 NaN, T₁<T₂<T₃ mọi epoch.

---

## 4. A/B đã đo: GRPO vs PPO (weighted) — paired, cùng seed, lex-best-of-K

| Scale | Δ T₁ | Δ T₂ | Δ T₃ | Win-rate | T₁-regress |
|---|---|---|---|---|---|
| Nhỏ (2-layer, 20ep, |A_r|=30/45) | −1.8% | −1.6% | −1.3% | 21/40 | — |
| **Điểm-ngọt (4-layer, 35ep, |A_r|=45, K=8 train; N=50, K=64 eval)** | **−2.8%** | **−2.2%** | **−2.0%** | 27/50 | 21/50 |

- **Lợi thế TĂNG theo scale** (~1.5% → ~2.5%) → ngoại suy full-scale ~3-4%+.
- GRPO cải thiện **cả 3 lớp** kể cả T₁ (rank là optimizer sạch hơn weighted-scalar).
- **Sắc thái:** per-instance còn nhiễu (54% win, 42% T₁-regress) → lợi thế ở **trung bình**, chưa áp đảo sạch.

→ **Phán quyết: GRPO đáng dùng** (chất lượng tốt hơn thật + tăng theo scale). Còn việc: **xác nhận full-scale**.

---

## 5. VIỆC CẦN LÀM TIẾP (theo thứ tự)

### 5.1 — Train full-scale (việc chính, cần GPU lâu)
```bash
MODE=full ./train.sh                 # GRPO (ALGO=grpo default) → checkpoint cuối
MODE=full ALGO=ppo ./train.sh        # baseline weighted (cho A/B + paper)
```
- Full = 1000 epoch / 100k data / 12 layer (xem `train.sh`). **~10-20 GPU-giờ/run.** Cân nhắc giảm epoch
  (vd 200-400) nếu T₁ bão hoà sớm.
- ⚠️ **Dọn cache trước:** `rm -f data/*_data.data` (cache cũ size/M khác sẽ bị nạp nhầm — đã dính bug này).
- Theo dõi `val/T1_best,T2_best,T3_best` trong `lightning_logs/version_*/metrics.csv` (T₂/T₃ phải GIẢM sau khi
  T₁ bão hoà — KHÔNG hiện ở stdout vì `self.log` không `prog_bar`).

### 5.2 — Phase 6: đo trên `data/ood/` (đã có code)
```bash
uv run python eval/run_grid.py --ckpt <grpo_ckpt> --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 --out eval/grpo_P.csv
uv run python eval/run_grid.py --ckpt <ppo_ckpt>  --path data/ood --M 2,3,5,7,10 --variant P --num_sample 100 --out eval/ppo_P.csv
# rồi paired_win_rate(grpo_rows, ppo_rows) trong eval/stats.py → win-rate lexicographic + T₁-regress count
```
Tiêu chí PASS: GRPO **win-rate ≥ 50%** với **T₁-regression thấp** + T₂/T₃ trung bình thấp hơn rõ.

### 5.3 — Phase 7B: dọn deprecated (CHỈ sau khi 5.2 xác nhận GRPO thắng)
Xem [`docs/plan/d2_grpo/08_phase7_cleanup.md`](plan/d2_grpo/08_phase7_cleanup.md) §7B: bỏ nhánh weighted+critic
nếu chốt chỉ GRPO. Giữ nếu vẫn cần baseline A/B trong paper.

---

## 6. Vận hành / cạm bẫy (QUAN TRỌNG cho session sau)

- **GPU SHARE:** máy có job khác `python wgp.py --i2v-14B` (~2GB, KHÔNG phải của ta) — **đừng kill**. Kiểm
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`.
- **Job sống sót session-restart:** dùng `setsid bash -c '...' < /dev/null > log 2>&1 &` (nohup KHÔNG đủ —
  session-restart giết cả nohup; setsid tạo session mới → sống). Kết quả ghi file log → đọc lại bất kỳ lúc nào.
- **Cache trap:** `MultiSizeCARPGenerator` LOAD `data/*_data.data` nếu tồn tại → nếu size/M khác sẽ sai. **Luôn
  `rm -f data/*_data.data`** trước run mới (hoặc dùng path riêng).
- **Load checkpoint:** PyTorch 2.6 default `weights_only=True` → checkpoint chứa `CARPEnv` sẽ lỗi. Dùng
  `torch.load(ckpt, weights_only=False)` (hoặc `PPO.load_from_checkpoint`).
- **run_parallel cần batch ≥ 10** (num_epochs=10 hardcode trong `env.get_reward/get_objective`). GRPO batch
  hiệu dụng = `B×K` → luôn thoả; eval `num_sample ≥ 10` (clamp sẵn).
- **GRPO chậm 8×** PPO/epoch (batchify K=8). |A_r| lớn càng chậm. Validation: |A_r|=45, 4-layer, K=8 ≈ 3
  phút/epoch.
- **Test suite ~30s** (94→122 test). Khi dev: chạy `unittest tests.test_X` riêng; `discover` 1 lần cuối phase.
- `eval/_ab_compare.py` là **script tạm** (đã xoá sau A/B) — viết lại từ §4 nếu cần A/B nhanh: load policy
  weights (class-agnostic, `weights_only=False`), generate fixed eval set (seed cố định), batchify K, lex-best,
  so mean + paired win-rate.

---

## 7. Liên hệ với plan khác

- **`docs/plan/dynamic_plan/`** (Scheduler/M, B+ precedence) — ĐÃ implement (Phase 0/1/3 + B+ + reward phân
  cấp + eval scaffold). **Trực giao** với D2: D2 đổi *learning signal*, dynamic_plan đổi *Scheduler/M*.
- Scheduler còn TODO chất lượng (`scheduler.py`): `[assign]` LPT/chunking là heuristic; "staged + min-max T₁
  + load-aware T₂/T₃" đã bàn (chưa code) — sẽ giúp T₂/T₃ mượt hơn, độc lập GRPO.

---

## 8. Trạng thái git khi handoff

- Branch `dev`. D2 commit cuối: `5835152`. Suite 122 xanh. Working tree sạch (chỉ còn thay đổi pre-existing
  không liên quan: `.gitignore`, `data/ood/`, `setup.sh`, `data/cache` xoá, `requirements.txt`).
- `rl/ppo.py` byte-identical từ trước D2 → A/B + rollback an toàn.
