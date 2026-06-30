# OSM Pre-Generation Strategy for HDCARP Training

## 0. Key architectural decisions

**Do NOT generate per-M.** `build_instance` only uses M to set `C = Σq/M + 0.5`. Raw per-arc demand `q` is stored in `req[:,2]` (pre-normalization), and `import_instance(es, M=...)` recomputes the fleet at load. Generate arcs **once per (city, bucket)** and **sample M ∈ {2,3,5,7,10} in the dataloader**, recomputing `C = q.sum()/M + 0.5` per access. **Net effect: 1x storage instead of 5x.**

**Two density tiers, not one.** Real urban drive networks have out-degree ranging from ~1.6 (organic/medieval) to ~2.5+ (grid). A single d=2.0 target rejects nearly all organic-city subgraphs. Two tiers cover the real distribution:

| Tier | Density d | Cities | Buckets |
|------|-----------|--------|---------|
| d=1.5 | 1.5 | organic (out-degree < 1.9) | 20_30, 30_45, 40_60, 50_75 |
| d=2.0 | 2.0 | grid + mixed (out-degree ≥ 1.9) | 20_40, 30_60, 40_80, 50_100 |
| d=3.0 | 3.0 | grid-only (out-degree ≥ 2.5) | 40_120 |

**BUCKETS definition** (±5% arc tolerance, `max_req = 3*(max_arc//4)`):

```python
BUCKETS = {
    # d = 1.5 — organic cities
    "20_30":  dict(num_loc=20, min_arc=28, max_arc=32),   # |A_r|=21
    "30_45":  dict(num_loc=30, min_arc=43, max_arc=47),   # |A_r|=33
    "40_60":  dict(num_loc=40, min_arc=57, max_arc=63),   # |A_r|=45
    "50_75":  dict(num_loc=50, min_arc=71, max_arc=79),   # |A_r|=54
    # d = 2.0 — grid/mixed cities
    "20_40":  dict(num_loc=20, min_arc=38, max_arc=42),   # |A_r|=30
    "30_60":  dict(num_loc=30, min_arc=57, max_arc=63),   # |A_r|=45
    "40_80":  dict(num_loc=40, min_arc=76, max_arc=84),   # |A_r|=60
    "50_100": dict(num_loc=50, min_arc=95, max_arc=105),  # |A_r|=75
    # d = 3.0 — grid-only
    "40_120": dict(num_loc=40, min_arc=114, max_arc=126), # |A_r|=90
}
```

**Curriculum phases** defined by |A_r| (the sequence length the policy must handle):

```python
CURRICULUM = {
    "small":  ["20_30", "20_40"],                              # |A_r| ≤ 30
    "medium": ["30_45", "30_60", "40_60", "50_75", "40_80"],  # |A_r| ∈ (30, 60]
    "large":  ["50_100", "40_120"],                            # |A_r| > 60
}
```

**Generator function signature** (all other params derive from city and bucket):

```python
def gen_city_bucket(
    city: str,        # → bbox lookup from CITIES dict
    bucket: str,      # → (num_loc, min_arc, max_arc) from BUCKETS dict
    per_bucket: int,  # number of accepted instances to generate
    seed: int,        # per-shard seed: base ^ hash(city, bucket)
    out_dir: str = "data/osm_train",
)
# Derived automatically: bbox, num_loc, min_arc, max_arc, max_req, m_nominal=3,
# output path = f"{out_dir}/{city}/{bucket}.npz"
```

---

## 1. City list (28 cities, ~42 bboxes)

Bounding boxes are central dense ~2km×2km districts. **Preflight each bbox** with one `load_osm_graph` call and measure `G.number_of_edges() / G.number_of_nodes()` (out-degree) to confirm which density tier it serves. Minimum 1,500 nodes required.

| # | City | District | N | S | E | W | Topology | Continent |
|---|------|----------|------|------|------|------|----------|-----------|
| 1 | New York | Midtown Manhattan | 40.7600 | 40.7420 | -73.9710 | -73.9930 | grid | N. America |
| 2 | Chicago | The Loop | 41.8900 | 41.8720 | -87.6190 | -87.6420 | grid | N. America |
| 3 | Boston | Back Bay/North End | 42.3680 | 42.3500 | -71.0520 | -71.0760 | organic | N. America |
| 4 | Toronto | Downtown Core | 43.6610 | 43.6430 | -79.3730 | -79.3960 | mixed | N. America |
| 5 | Mexico City | Centro/Roma | 19.4350 | 19.4170 | -99.1500 | -99.1730 | mixed | N. America |
| 6 | San Francisco | SoMa/FiDi | 37.7950 | 37.7770 | -122.3950 | -122.4180 | mixed | N. America |
| 7 | Buenos Aires | Microcentro | -34.5950 | -34.6130 | -58.3680 | -58.3910 | grid | S. America |
| 8 | São Paulo | Centro/Sé | -23.5400 | -23.5580 | -46.6280 | -46.6510 | organic | S. America |
| 9 | Bogotá | La Candelaria | 4.6080 | 4.5900 | -74.0650 | -74.0870 | mixed | S. America |
| 10 | London | City/Soho | 51.5200 | 51.5050 | -0.0900 | -0.1230 | organic | Europe |
| 11 | Paris | 1er–4e arr. | 48.8650 | 48.8500 | 2.3640 | 2.3380 | organic | Europe |
| 12 | Rome | Centro Storico | 41.9050 | 41.8900 | 12.4900 | 12.4660 | organic | Europe |
| 13 | Barcelona | Eixample | 41.3960 | 41.3810 | 2.1750 | 2.1530 | grid | Europe |
| 14 | Amsterdam | Centrum | 52.3760 | 52.3620 | 4.9100 | 4.8830 | organic | Europe |
| 15 | Berlin | Mitte | 52.5260 | 52.5110 | 13.4090 | 13.3830 | mixed | Europe |
| 16 | Prague | Staré Město | 50.0920 | 50.0780 | 14.4320 | 14.4090 | organic | Europe |
| 17 | Cairo | Downtown/Tahrir | 30.0560 | 30.0410 | 31.2470 | 31.2300 | organic | Africa |
| 18 | Lagos | Lagos Island | 6.4570 | 6.4420 | 3.4030 | 3.3870 | mixed | Africa |
| 19 | Marrakesh | Medina | 31.6350 | 31.6210 | -7.9810 | -7.9970 | organic | Africa |
| 20 | Nairobi | CBD | -1.2790 | -1.2930 | 36.8290 | 36.8130 | mixed | Africa |
| 21 | Tokyo | Shinjuku | 35.7000 | 35.6850 | 139.7110 | 139.6920 | mixed | Asia |
| 22 | Hanoi | Old Quarter | 21.0400 | 21.0260 | 105.8580 | 105.8410 | organic | Asia |
| 23 | Mumbai | Fort/Colaba | 18.9300 | 18.9150 | 72.8390 | 72.8230 | organic | Asia |
| 24 | Singapore | CBD/Chinatown | 1.2890 | 1.2750 | 103.8530 | 103.8380 | mixed | Asia |
| 25 | Shanghai | Huangpu | 31.2350 | 31.2200 | 121.4880 | 121.4710 | mixed | Asia |
| 26 | Chandigarh | Sector 17 | 30.7460 | 30.7320 | 76.7900 | 76.7720 | grid | Asia |
| 27 | Sydney | CBD | -33.8630 | -33.8780 | 151.2150 | 151.1990 | mixed | Oceania |
| 28 | Melbourne | CBD (Hoddle grid) | -37.8080 | -37.8230 | 144.9740 | 144.9560 | grid | Oceania |

**City → bucket routing** (determined by measured out-degree at preflight):

| Out-degree | Density tier | Buckets served |
|------------|-------------|----------------|
| < 1.9 | d=1.5 | 20_30, 30_45, 40_60, 50_75 |
| 1.9 – 2.4 | d=2.0 | 20_40, 30_60, 40_80, 50_100 |
| ≥ 2.5 | d=2.0 + d=3.0 | all five d=2.0 + 40_120 |

**Second bboxes:** add a 2nd neighborhood for the 14 largest cities → ~42 distinct road canvases.

---

## 2. Volume & distribution

**Target: ~150k accepted instances** across 9 buckets. Counts weighted by curriculum-phase exposure.

| Bucket | \|A_r\| | Phase | Exposure | Accepted | Reject factor | Attempts |
|--------|---------|-------|----------|----------|---------------|----------|
| 20_30  | 21 | small  | 3 | 18,000 | 1.10 | 19,800 |
| 20_40  | 30 | small  | 3 | 22,000 | 1.15 | 25,300 |
| 30_45  | 33 | medium | 2 | 16,000 | 1.15 | 18,400 |
| 30_60  | 45 | medium | 2 | 16,000 | 1.15 | 18,400 |
| 40_60  | 45 | medium | 2 | 16,000 | 1.18 | 18,900 |
| 50_75  | 54 | medium | 2 | 14,000 | 1.18 | 16,500 |
| 40_80  | 60 | medium | 2 | 16,000 | 1.18 | 18,900 |
| 50_100 | 75 | large  | 1 | 16,000 | 1.20 | 19,200 |
| 40_120 | 90 | large  | 1 | 16,000 | **2.50** | 40,000 |
| **Total** | | | | **150,000** | | **~195,000** |

Rationale:
- small buckets seen in all 3 phases → largest pools; 20_30 slightly smaller since d=1.5 instances are faster to generate.
- medium buckets split evenly; organic + grid/mixed cities both contribute here.
- large buckets equal at 16k; (40,120) dominated by grid cities with 2.5x rejection.

---

## 3. Diversity strategy

- **~3,600 instances per city on average** (150k / 42 canvases). Cap any single (city, bucket) cell at **1,200 instances**.
- **Topology balance per bucket:**
  - d=1.5 buckets: organic cities only (Paris, London, Rome, Amsterdam, Prague, Cairo, Marrakesh, Boston, São Paulo, Hanoi, Mumbai, …).
  - d=2.0 buckets: grid + mixed cities (Chicago, Tokyo, Singapore, Shanghai, Sydney, Melbourne, Toronto, …).
  - 40_120 (d=3.0): **grid-only** (New York, Chicago, Buenos Aires, Chandigarh, Melbourne) — needs out-degree ≥ 2.5.
- **M handling at load:** stored M-agnostic (m_nominal=3 aligns with paper benchmark). At load, sample M uniform over {2,3,5,7,10}. Weight toward {2,3} if A/B shows under-training on tight-fleet regimes.

---

## 4. Practical generation plan

**Preflight (one-time per bbox):**
- Call `load_osm_graph` for each bbox; record `G.number_of_nodes()` and out-degree.
- Reject/resize any canvas under 1,500 nodes.
- Assign city to density tier based on measured out-degree (see §1 routing table).

**Download:** 42 bboxes × (~30–60s) ≈ **35–55 min** (osmnx caches under `data/cache`).

**Generation (8 CPU workers, ~500 inst/min aggregate):**
- 195k attempts / 500 ≈ **~6.5 hours total**.
- Run (40,120) first/overnight — it alone is ~20% of wall-time.

**Parallelism:** one process per (city, bucket) shard. Load `G_proj` once per city worker, generate all its eligible buckets from that in-memory graph.

**Directory layout:**
```
data/
├── cache/                        # OSMnx HTTP cache (one-time download)
├── osm_train/
│   ├── <city>/
│   │   ├── 20_30.npz             # all instances for this (city, bucket) bundled
│   │   ├── 20_40.npz             # only buckets the city's density tier serves
│   │   └── ...
│   └── ...                       # 28 cities × eligible buckets ≈ 200 files
├── osm_test/
│   └── <city>/
│       └── <bucket>.npz          # held-out OSM test instances, generated separately
└── benchmark/
    └── <|V|>_<|A|>/
        └── *.npz                 # Ha et al. (2024) benchmark instances, reproduced
                                  # via gen_test_instances.py using paper's formulas
```
One bundled `.npz` per `(city, bucket)` — bucket name is self-describing, no subdirectory needed. Load all cities for one bucket: `glob("data/osm_train/*/20_40.npz")`. Train/test/benchmark are top-level siblings so the dataloader cannot accidentally mix them.

**Storage:** ~17 GB raw `.npz`. No pre-convert to `.data` (see §5).

---

## 5. Dataloader integration

**Recommendation: keep `.npz` on disk, add `OSMMultiSizeDataset`.**

**Adapter logic** (mirror `generate()` exactly):
1. Load `.npz`; read `req` columns `[tail, head, q, clss, service, traversal]` and `nonreq`.
2. **Sample M ∈ {2,3,5,7,10}**, recompute `C = req[:,2].sum()/M + 0.5`.
3. Build `adj = dist_edges_from_file(es)` → `(|A_r|+1, |A_r|+1)`.
4. Prepend depot row 0; `demands = [0, q]/C`; `capacity=1`; `num_vehicle=M`.
5. Emit `td.unsqueeze(0)`.

**Zero trainer changes:** `OSMMultiSizeDataset` exposes `self.bucket_ranges` and `collate_fn = torch.cat`, same interface as `MultiSizeCARPGenerator`. Pair with existing `SizeBucketBatchSampler`.

**Recommended rollout:**
- `curriculum_small`: 50/50 synthetic + OSM on small buckets.
- `curriculum_medium`: 50/50 synthetic + OSM on small + medium buckets.
- `curriculum_large`: 80/20 OSM/synthetic on all buckets.
- A/B against synthetic-only on held-out `osm_test/` cities before committing the full 150k run.

**Large-scale generalization** is zero-shot: apply the model trained on the largest bucket (|A_r|=90) directly to benchmark instances with larger |A_r| without retraining. No extra training buckets needed — this mirrors DaAM's approach (train Task100, evaluate Task200–600). The attention mechanism is size-agnostic by design.

**Baseline comparison** uses `src/solvers/lp.py` (MILP-P/U) and `src/solvers/meta.py` (ILS, EA, ACO) — all already implemented. Test set is `data/benchmark/` reproduced from Ha et al. (2024) using the same instance generator.

---

## 6. Code changes required before generation

1. **`src/utils/data_utils.py`** — `gen_city_bucket` + `CITIES` + `BUCKETS` + `CURRICULUM` already implemented. All instance physics (`build_instance`, `load_osm_graph`, graph sampling) are self-contained; no dependency on `scripts/`.
2. **`scripts/preflight_cities.sh`**: run `load_osm_graph` for all 28 bboxes, print node count and out-degree, flag any below 1,500 nodes or whose out-degree contradicts the assigned tier.
3. **`src/env/osm_dataset.py`**: `OSMMultiSizeDataset` that loads `.npz` shards and samples M at access time (see §5).

Relevant files:
- `src/utils/data_utils.py` — `gen_city_bucket`, `CITIES`, `BUCKETS`, `CURRICULUM`
- `src/utils/ops.py` — `import_instance`, `dist_edges_from_file`
- `src/env/generator.py` — `generate`, `MultiSizeCARPGenerator`, `SizeBucketBatchSampler`
