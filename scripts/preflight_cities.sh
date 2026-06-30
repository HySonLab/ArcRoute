#!/usr/bin/env bash
# Download and cache OSM road graphs for all 28 training cities.
# Usage: bash scripts/preflight_cities.sh
# Requirements: uv add osmnx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache"

uv run python -c "import osmnx" 2>/dev/null \
  || { echo "osmnx not installed. Run: uv add osmnx"; exit 1; }

mkdir -p "$CACHE_DIR"

# columns: name  N        S        E          W
CITIES=(
  "new_york       40.7600  40.7420  -73.9710  -73.9930"
  "chicago        41.8900  41.8720  -87.6190  -87.6420"
  "boston         42.3680  42.3500  -71.0520  -71.0760"
  "toronto        43.6610  43.6430  -79.3730  -79.3960"
  "mexico_city    19.4350  19.4170  -99.1500  -99.1730"
  "san_francisco  37.7950  37.7770 -122.3950 -122.4180"
  "buenos_aires  -34.5950 -34.6130  -58.3680  -58.3910"
  "sao_paulo     -23.5400 -23.5580  -46.6280  -46.6510"
  "bogota          4.6080   4.5900  -74.0650  -74.0870"
  "london         51.5200  51.5050   -0.0900   -0.1230"
  "paris          48.8650  48.8500    2.3640    2.3380"
  "rome           41.9050  41.8900   12.4900   12.4660"
  "barcelona      41.3960  41.3810    2.1750    2.1530"
  "amsterdam      52.3760  52.3620    4.9100    4.8830"
  "berlin         52.5260  52.5110   13.4090   13.3830"
  "prague         50.0920  50.0780   14.4320   14.4090"
  "cairo          30.0560  30.0410   31.2470   31.2300"
  "addis_ababa     9.0300   9.0000   38.7700   38.7400"
  "casablanca     33.6100  33.5700   -7.5900   -7.6500"
  "nairobi        -1.2790  -1.2930   36.8290   36.8130"
  "tokyo          35.7000  35.6850  139.7110  139.6920"
  "hanoi          21.0400  21.0260  105.8580  105.8410"
  "mumbai         18.9300  18.9150   72.8390   72.8230"
  "singapore       1.2890   1.2750  103.8530  103.8380"
  "shanghai       31.2350  31.2200  121.4880  121.4710"
  "chandigarh     30.7460  30.7320   76.7900   76.7720"
  "sydney        -33.8630 -33.8780  151.2150  151.1990"
  "melbourne     -37.8080 -37.8230  144.9740  144.9560"
)

TOTAL=${#CITIES[@]}
PASSED=(); FAILED=()

echo "Cache dir : $CACHE_DIR"
echo "Cities    : $TOTAL"
echo

for i in "${!CITIES[@]}"; do
  read -r name N S E W <<< "${CITIES[$i]}"
  printf "[%2d/%d] %s ...\n" $((i+1)) $TOTAL "$name"

  if uv run python -c "
import sys, osmnx as ox
cache_dir, name, N, S, E, W = sys.argv[1:]
N, S, E, W = float(N), float(S), float(E), float(W)
ox.settings.cache_folder = cache_dir
try:
    try:
        G = ox.graph_from_bbox(north=N, south=S, east=E, west=W, network_type='drive')
    except TypeError:
        G = ox.graph_from_bbox(bbox=(W, S, E, N), network_type='drive')
    n, e = G.number_of_nodes(), G.number_of_edges()
    status = 'OK' if n >= 250 else 'LOW'
    print(f'  [{status}] {name:<20} nodes={n}  edges={e}')
    sys.exit(0 if status == 'OK' else 1)
except Exception as ex:
    print(f'  [ERR] {name:<20} {ex}')
    sys.exit(2)
" "$CACHE_DIR" "$name" "$N" "$S" "$E" "$W"; then
    PASSED+=("$name")
  else
    FAILED+=("$name")
  fi
done

echo
echo "=================================================="
echo "PASS : ${#PASSED[@]}/$TOTAL"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAIL : ${#FAILED[@]}"
  for name in "${FAILED[@]}"; do echo "  - $name"; done
  echo
  echo "Adjust bboxes in this script for failed cities."
else
  echo "All cities OK."
fi

cache_files=$(find "$CACHE_DIR" -type f | wc -l)
cache_mb=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
echo
echo "Cache     : $cache_files files, ${cache_mb} MB"
echo "Upload to server: rsync -av scripts/cache/ server:ArcRoute/scripts/cache/"
