# ILS (InsertCheapestHCARP) — Handoff & usage guide

> Cập nhật: 2026-06-24. Local search inter/intra đã implement và verify đầy đủ.
> ILS chạy được, solution hợp lệ, scripts sẵn sàng tái sử dụng.

---

## 1. TL;DR — trạng thái hiện tại

- **`InsertCheapestHCARP`** (`src/solvers/meta.py`): construction heuristic (cheapest insertion
  theo class order) + local search → giải HDCARP-P/U trên instance `.npz`.
- **Local search** (`src/utils/local_search.py`): swap-based intra-route + inter-route,
  vectorized numpy (15× so với Python loops). Đã verify 0 delta mismatch, 161 tests pass.
- **Bug đã fix:** `get_once` dùng `idx` (index vào `costs`) trực tiếp để index vào `routes` —
  sai khi một số vehicle infeasible. Fix: `chosen = idxs[idx]`.
- **Scripts:** `ils_log.py` (giải + log), `validate_solution.py` (verify từ file).

---

## 2. Kiến trúc local search

### Intra-route swap

Đổi vị trí hai arcs trong cùng một route. Delta O(1):

- **Adjacent** (j = i+1): `Δ = adj[a,bj]+adj[bj,bi]+adj[bi,d] − adj[a,bi]+adj[bi,bj]+adj[bj,d]`
- **Non-adjacent**: `Δ = adj[a,bj]+adj[bj,a2]+adj[c,bi]+adj[bi,d] − old`

`_intra_delta_matrix` tính toàn bộ ma trận m×m bằng numpy broadcasting thay vì O(m²) Python loops.

**Variant P:** chỉ swap arcs cùng class. **Variant U:** swap tự do.

### Inter-route swap

Đổi một arc từ route r1 với một arc từ route r2. Delta O(1), kiểm tra capacity trước:

```
feas = (load1 - dx + dy ≤ 1.0) & (load2 - dy + dx ≤ 1.0)
# Variant P thêm: clss[x] == clss[y]
```

### Pipeline trong `ls()`

```
action (flat) → gen_tours → routes
  ↓ intra_route_opt per class (variant P) hoặc global (variant U)
  ↓ inter_route_opt all pairs
→ routes (improved)
```

### Lưu ý: Scheduler re-partition

`get_Ts` gọi `Scheduler._build_P` — nó **strip toàn bộ 0-separator** và **re-partition lại**
arcs theo class-balanced chunks, bỏ qua vehicle boundaries của `ls()`. Do đó:

- T1/T2/T3 từ `get_Ts` ≠ T1/T2/T3 tính trực tiếp từ `ls()` routes.
- ILS chọn best solution theo Scheduler T (dùng để rank), không phải ls() T.
- `validate_solution.py` báo cả hai, dùng `[INFO]` cho chênh lệch này (không phải bug).

---

## 3. Files liên quan

```
src/
├── solvers/
│   ├── meta.py            InsertCheapestHCARP — construction + ls orchestration
│   │                      get_once(): cheapest insertion (bug fix: idxs[idx])
│   │                      vars dict phải có 'nv' để Scheduler biết M
│   ├── cal_reward.py      get_Ts(vars, actions) — batch T1/T2/T3 qua Scheduler
│   └── scheduler.py       Scheduler._build_P / _completion_times (timing model)
│
├── utils/
│   ├── local_search.py    intra/inter operators, ls(), lsRL()
│   ├── nb_utils.py        gen_tours, deserialize_tours, deserialize_tours_batch
│   └── ops.py             run_parallel2, import_instance, floyd_warshall
│
scripts/
├── ils_log.py             Giải 1 instance → route_log.txt + route_log.sol
└── validate_solution.py   Verify solution từ .sol + instance .npz
```

---

## 4. Timing model (Scheduler)

Với một trip `[arc1, arc2, ..., arcN]`:

```
path = [depot(0), arc1, ..., arcN, depot(0)]
t[0] = service[depot] = 0
t[k] = service[arc_k] + adj[arc_{k-1}, arc_k]   (deadhead + service)
cum  = cumsum(t)

completion[k] = cum[k+1]          # thời điểm arc_k xong
duration      = cum[-1]            # bao gồm return to depot
```

`adj[i, j]` = Floyd-Warshall shortest path từ **head của arc i** đến **tail của arc j**
(node-to-node, đã verify khớp 100% với rebuild từ req+nonreq).

**T_k** = max completion time của arc thuộc class k, trên tất cả vehicles và trips.
Multi-trip: completion được cộng thêm `offset` = tổng duration các trips trước.

---

## 5. Format file solution (`.sol`)

```
instance: data/ood/osm_cityB/40/34_13_632.npz
variant: P
vehicles: 3
T1: 3.644184  T2: 9.865761  T3: 14.580141
route 1: 0 13 3 24 20 18 23 7 16 0
route 2: 0 1 21 4 5 11 10 15 17 0
route 3: 0 6 8 12 9 19 2 22 14 0
```

- Route dùng **arc index** (1-based, 0 = depot sentinel).
- Mỗi route bắt đầu và kết thúc bằng `0`.
- T1/T2/T3 là giá trị **Scheduler** (re-partitioned), không phải ls() routes.

---

## 6. Workflow chuẩn

```bash
# Giải 1 instance
uv run python scripts/ils_log.py \
    --file data/ood/osm_cityB/40/34_13_632.npz \
    --variant P \
    --vehicles 3 \
    --num_sample 20 \
    --log outputs/my_solution.txt
# → outputs/my_solution.txt   (human-readable: node chain + timing table)
# → outputs/my_solution.sol   (machine-readable: parse lại được)

# Validate bất cứ lúc nào
uv run python scripts/validate_solution.py \
    --instance data/ood/osm_cityB/40/34_13_632.npz \
    --solution outputs/my_solution.sol \
    --log      outputs/my_validate.txt
```

---

## 7. Checks trong validate_solution.py

| # | Check | Mô tả |
|---|-------|--------|
| 1 | Coverage | Mỗi arc required xuất hiện đúng 1 lần |
| 2 | Capacity | Mỗi route: `sum(demand) ≤ 1.0` (normalized) |
| 3 | Depot sentinels | Route bắt đầu và kết thúc bằng `0` |
| 4 | Class ordering | Variant P: class sequence non-decreasing trong mỗi route |
| 5 | Deadhead adj | `adj[i,j] == FW(head_i → tail_j)` cho tất cả pairs |
| 6 | Service times | `service_time[arc] == req[arc-1, 4]` |
| 7A | Timing (ls routes) | Re-tính T1/T2/T3 thủ công từ saved routes |
| 7B | Timing (Scheduler) | Scheduler re-partition → manual re-check phải khớp saved T |

---

## 8. Các vấn đề đã giải quyết

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| Capacity violation 15/20 routes | `routes[idx] = paths[idx]` dùng index vào costs làm index vào routes | `chosen = idxs[idx]; routes[chosen] = paths[chosen]` |
| `get_Ts` missing | Chưa implement | Thêm vào `cal_reward.py` |
| `run_parallel2` missing | Chưa implement | Thêm vào `ops.py` (sequential map) |
| `KeyError: 'demand'` in Scheduler | `vars` dict thiếu `'nv'` và td thiếu `'demand'` | Thêm cả hai |
| `np.flip` trong intra-swap | Flip arc = đảo hướng, sai cho directed graph | Thay bằng swap vị trí |
