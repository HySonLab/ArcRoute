# Pre-Phase — Dọn deprecated & data cũ (chạy TRƯỚC Phase 0)

> Mục tiêu: **clean slate** trước khi cải tổ. Mọi data/checkpoint hiện tại đều sinh từ **công thức SAI**
> (rule 60–70, capacity `Σ(q/3+0.5)`, metric complete-Euclid) và phần lớn **vượt trần `|A_r|≤100`** →
> không dùng lại được. Giữ lại chỉ gây nhầm lẫn khi so kết quả. Phase này KHÔNG sửa công thức, chỉ xóa.

## ⚠️ An toàn TRƯỚC khi xóa (bắt buộc)

`data/2m`, `data/5m` đang **được git track** (commit từ initial). Tôi không phải người tạo chúng → trước khi
xóa phải **lưu vết để có thể khôi phục**:

```bash
git tag archive/pre-data-overhaul        # mốc khôi phục toàn bộ trạng thái cũ
# hoặc: git branch archive/old-hrda-data
```
> Sau khi tag, mọi `git rm` đều khôi phục được qua `git checkout archive/pre-data-overhaul -- <path>`.
> Nếu cần **tái lập kết quả HRDA cũ** sau này, dùng tag này — KHÔNG dựa vào working tree.

## C1 — Commit các test deprecated đã xóa (đang staged `D`)

Đã bị xóa khỏi đĩa, chỉ chờ commit (xem `git status`):
`tests/data.py`, `tests/lin.py`, `tests/ox_data.py`, `tests/sampler.py`, `tests/stuff.py`,
`tests/temp.py`, `tests/cache/ce1d…json`.

```bash
git add -A tests/
git status --short        # xác nhận chỉ còn test_generator.py, test_gen.py
```
> Giữ lại: `tests/test_generator.py`, `tests/test_gen.py` (đang dùng cho cổng test Phase 0).

## C2 — Xóa data benchmark cũ (SAI công thức + vượt trần)

`data/2m` (360 npz) và `data/5m` (280 npz): sinh với 60–70 split, capacity sai, **bucket 100→640 arcs**
(trần mới `|A_r|≤100` ⇔ `|A|≤~133`) → **toàn bộ vô hiệu**. Sẽ **tái sinh** ở Phase 1–4 bằng generator đã sửa.

```bash
git rm -r data/2m data/5m       # tracked → git rm (đã có tag archive ở trên)
rm -rf data/cache/*             # cache cũ (gitignored); data/cache giữ lại folder rỗng
```

## C3 — Xóa checkpoint & log train trên data sai (gitignored — chỉ `rm`)

Train trên data sai → vô nghĩa, lại nặng (~430MB):
```bash
rm -rf checkpoints/cl123        # ~350MB, 6 ckpt
rm -rf lightning_logs/*         # ~84MB, version_0..8
rm -rf logs/* 2>/dev/null || true
```
> `checkpoints/`, `lightning_logs/`, `logs` đều đã trong `.gitignore` → không ảnh hưởng git, chỉ giải phóng đĩa.

## C4 — Dọn tham chiếu code/script trỏ tới data cũ (sửa, không xóa file)

Không xóa file code, chỉ **cập nhật đường dẫn/comment chết** để khỏi chạy nhầm vào data cũ:

- `baseline/aco.py:18`, `baseline/ea.py:18` — default `--path data/5m60` (thư mục **không tồn tại**) →
  đổi sang đường dẫn data mới (sẽ chốt ở Phase 1, vd `data/5m/60`) hoặc bỏ default, bắt buộc truyền `--path`.
- `baseline/benchmark.sh` — hàng loạt dòng (phần lớn đã comment) trỏ `data/2m`, `data/5m`, `data/5m40`,
  `data/5m60`, `../cpkts/*.ckpt`. Đánh dấu **"cần cập nhật sau khi tái sinh data (Phase 1–4)"** hoặc dọn
  hẳn các dòng chết. Không để script chạy ra data cũ.
- `train.py:32` default `data/train_data.data`, `env/env.py:102` & `env/generator.py` default
  `carp_data.pt` — **auto-regenerate** nên không cần xóa file (hiện không có trên đĩa). Chỉ cần đảm bảo
  **không còn `*.data`/`carp_data.pt` cũ** sót lại:
  ```bash
  find . -maxdepth 2 \( -name "*.data" -o -name "carp_data.pt" \) -not -path "./.venv/*" -delete
  ```

## C5 — Dọn rác build

```bash
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
rm -rf temp/ 2>/dev/null || true     # thư mục tạm do data/gen.py tạo (thường đã tự xóa)
```

---

## ✅ Cổng test Pre-Phase (PASS mới qua Phase 0)

> Xóa data KHÔNG được làm hỏng test/import. Các test hiện tại (`test_generator`, `test_gen`) dùng dữ liệu
> **on-the-fly / tempdir**, KHÔNG phụ thuộc `data/2m`,`data/5m` → phải vẫn xanh sau khi xóa.

1. **Test vẫn chạy độc lập với data đã xóa:**
   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py" -v   # PHẢI xanh
   ```
2. **Import không gãy** (không file nào hard-require data cũ lúc import):
   ```bash
   uv run python -c "import env.generator, common.ops; import importlib.util; \
   s=importlib.util.spec_from_file_location('g','data/gen.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('imports OK')"
   ```
3. **Không còn tham chiếu chết tới data cũ trong đường chạy chính** (chỉ còn trong comment/benchmark.sh đã đánh dấu):
   ```bash
   grep -rn "data/2m\|data/5m60\|data/5m40\|carp_data.pt" --include=*.py train.py baseline/aco.py baseline/ea.py
   ```
4. **Tag khôi phục tồn tại:** `git tag | grep archive/pre-data-overhaul`.

### Checklist
- [ ] Tạo tag/branch `archive/pre-data-overhaul` TRƯỚC khi xóa.
- [ ] Commit xóa test deprecated (C1); `tests/` chỉ còn 2 file test_*.
- [ ] `git rm -r data/2m data/5m`; xóa `data/cache/*` (C2).
- [ ] `rm -rf checkpoints/cl123 lightning_logs/* logs/*` (C3).
- [ ] Cập nhật default path `aco.py`/`ea.py` + đánh dấu `benchmark.sh`; xóa `*.data`/`carp_data.pt` sót (C4).
- [ ] Dọn `__pycache__`, `temp/` (C5).
- [ ] `unittest discover` xanh; import OK; không tham chiếu chết.
- [ ] Commit "Pre-Phase: remove deprecated tests + obsolete data/checkpoints (archived at tag)".

> Sau khi PASS toàn bộ → sang [`01_phase0_restore_formula.md`](01_phase0_restore_formula.md).
