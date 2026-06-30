# OSM Training Data — info

## Overview

- **Total instances**: 125,393
- **Files**: 220 `.npz` shards (`data/osm_train/<city>/<bucket>.npz`)
- **Cities**: 28 (see list below)
- **Buckets**: 10 (see table below)
- **Generator**: `src/utils/data_utils.py` — `gen_city_bucket()`

---

## Instance schema

Each `.npz` file contains a batch of instances for one `(city, bucket)` pair.

| Key | Shape | Description |
|-----|-------|-------------|
| `req` | `(N, n_req, 6)` | Required arcs |
| `nonreq` | `(N, *, 6)` | Non-required arcs (ragged) |
| `C` | `(N,)` | Vehicle capacity (generated with M=3) |
| `M` | `(N,)` | Fleet size = 3 nominal |
| `n_req` | `(N,)` | Number of required arcs |
| `tau` | `(N,)` | Load factor = Σq/(M×C) |

**Arc feature columns** (same layout for `req` and `nonreq`):

| Col | Feature | Formula | Range |
|-----|---------|---------|-------|
| 0 | `tail` | node index | int |
| 1 | `head` | node index | int |
| 2 | `demand` q | `0.5d + 0.5` | [0.5, 1.0] |
| 3 | `class` | uniform over {1,2,3} | {1,2,3} (0 for nonreq) |
| 4 | `service` s | `2d` | [0, 2.0] |
| 5 | `traversal` d | `d'/d'_max` (Euclidean, normalized) | (0, 1] |

`nonreq` columns 2–4 are always 0 (no demand, no class, no service).

**Capacity formula** (paper F5): `C = Σq / M + 0.5`
M-agnostic: raw demand `q = req[:,2]` is stored, so C can be recomputed at load time for any M ≥ 1.

---

## Buckets

| Bucket | num_loc | |A| range | |A_r| nominal | Curriculum | Density tier | Cities |
|--------|---------|-----------|-------------|------------|-------------|--------|
| `20_30` | 20 | 28–32 | 21 | small | d=1.5 | 28 |
| `20_40` | 20 | 38–42 | 30 | small | d=2.0 | 16 |
| `30_45` | 30 | 43–47 | 33 | medium | d=1.5 | 28 |
| `30_60` | 30 | 57–63 | 45 | medium | d=2.0 | 16 |
| `40_60` | 40 | 57–63 | 45 | medium | d=1.5 | 28 |
| `50_75` | 50 | 71–79 | 54 | medium | d=1.5 | 28 |
| `40_80` | 40 | 76–84 | 60 | medium | d=2.0 | 16 |
| `50_100` | 50 | 95–105 | 75 | large | d=2.0 | 15* |
| `80_120` | 80 | 114–126 | 90 | large | d=1.5 | 28 |
| `80_133` | 80 | 126–140 | 99 | large | d=2.0 | 16* |

*`buenos_aires/50_100` skipped (infeasible — bbox out-degree too low for d=2.0 at 50 nodes).

**|A_r| formula**: `|A_r| = 3 × ⌊|A| / 4⌋`

---

## Instance counts per bucket

| Bucket | Cities | Instances |
|--------|--------|-----------|
| 20_30 | 28 | 18,004 |
| 20_40 | 16 | 12,576 |
| 30_45 | 28 | 15,988 |
| 30_60 | 16 | 9,136 |
| 40_60 | 28 | 15,988 |
| 50_75 | 28 | 14,000 |
| 40_80 | 16 | 9,136 |
| 50_100 | 15 | 8,565 |
| 80_120 | 28 | 14,000 |
| 80_133 | 16 | 8,000 |
| **Total** | | **125,393** |

---

## Density tier routing

| Tier | out-degree | Buckets | Cities |
|------|------------|---------|--------|
| d=1.5 | < 1.9 | 20_30, 30_45, 40_60, 50_75, 80_120 | boston, sao_paulo, london, paris, rome, amsterdam, prague, cairo, marrakesh, hanoi, mumbai, barcelona (12) |
| d=2.0 | 1.9–2.4 | all 10 | new_york, chicago, toronto, mexico_city, san_francisco, buenos_aires, bogota, berlin, lagos, nairobi, tokyo, singapore, shanghai, chandigarh, sydney, melbourne (16) |

---

## City list (28 cities)

| City | Bbox (N, S, E, W) | Topology | Continent |
|------|------------------|----------|-----------|
| new_york | 40.76, 40.742, -73.971, -73.993 | grid | N. America |
| chicago | 41.89, 41.872, -87.619, -87.642 | grid | N. America |
| boston | 42.368, 42.35, -71.052, -71.076 | organic | N. America |
| toronto | 43.661, 43.643, -79.373, -79.396 | mixed | N. America |
| mexico_city | 19.435, 19.417, -99.15, -99.173 | mixed | N. America |
| san_francisco | 37.795, 37.777, -122.395, -122.418 | mixed | N. America |
| buenos_aires | -34.595, -34.613, -58.368, -58.391 | grid | S. America |
| sao_paulo | -23.54, -23.558, -46.628, -46.651 | organic | S. America |
| bogota | 4.608, 4.59, -74.065, -74.087 | mixed | S. America |
| london | 51.52, 51.505, -0.09, -0.123 | organic | Europe |
| paris | 48.865, 48.85, 2.364, 2.338 | organic | Europe |
| rome | 41.905, 41.89, 12.49, 12.466 | organic | Europe |
| barcelona | 41.396, 41.381, 2.175, 2.153 | organic | Europe |
| amsterdam | 52.376, 52.362, 4.91, 4.883 | organic | Europe |
| berlin | 52.526, 52.511, 13.409, 13.383 | mixed | Europe |
| prague | 50.092, 50.078, 14.432, 14.409 | organic | Europe |
| cairo | 30.056, 30.041, 31.247, 31.23 | organic | Africa |
| lagos | 6.457, 6.442, 3.403, 3.387 | mixed | Africa |
| marrakesh | 31.635, 31.621, -7.981, -7.997 | organic | Africa |
| nairobi | -1.279, -1.293, 36.829, 36.813 | mixed | Africa |
| tokyo | 35.7, 35.685, 139.711, 139.692 | mixed | Asia |
| hanoi | 21.04, 21.026, 105.858, 105.841 | organic | Asia |
| mumbai | 18.93, 18.915, 72.839, 72.823 | organic | Asia |
| singapore | 1.289, 1.275, 103.853, 103.838 | mixed | Asia |
| shanghai | 31.235, 31.22, 121.488, 121.471 | mixed | Asia |
| chandigarh | 30.746, 30.732, 76.79, 76.772 | grid | Asia |
| sydney | -33.863, -33.878, 151.215, 151.199 | mixed | Oceania |
| melbourne | -37.808, -37.823, 144.974, 144.956 | grid | Oceania |

---

## Regeneration

```bash
# Full production (28 cities, all buckets, 8 workers)
bash scripts/gen_data.sh --all --mode all --workers 8

# Single city
bash scripts/gen_data.sh --city paris --mode all

# Test run (5 instances/bucket)
bash scripts/gen_data.sh --all --mode all --workers 4 --per-bucket 5
```
