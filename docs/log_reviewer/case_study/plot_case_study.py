"""Case-study figure: 2 rows (small melbourne / large bogota) x 4 cols
(ILS, EA, ACO, HARMONY). Each panel: full road network in light gray,
serviced arcs colored by priority class (Okabe-Ito, fixed order), vehicles
distinguished by linestyle. Shared legend.

Draws the EVALUATED vehicle partitions (case/<alg>_<tier>_vehicles.json,
produced and T-verified by analyze_case.py) -- for the metaheuristics the
.sol 'route' lines are only the raw arc order, not the scored assignment.
Verifies each partition serves every required arc exactly once.

Run from anywhere: paths are absolute. Requires analyze_case.py run first.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CASE = "/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm/case"
OUT = "/home/quangnm/projects/HDCARP_with_ML/figures/png/case_study.png"

CLASS_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}  # Okabe-Ito, fixed order
VEH_STYLES = ["solid", (0, (5, 2)), (0, (1, 1.6)), (0, (3, 1, 1, 1))]
VEH_NAMES = ["vehicle 1 (solid)", "vehicle 2 (dashed)", "vehicle 3 (dotted)"]
BG = "#c9c9c9"

ALGS = [("ils", "ILS"), ("ea", "EA"), ("aco", "ACO"), ("harmony", "HARMONY")]
TIERS = [
    ("small", "Small tier – melbourne\n$|V|=18$, $|A|=31$, 21 required arcs"),
    ("large", "Large tier – bogota\n$|V|=81$, $|A|=127$, 93 required arcs"),
]


def load_solution(tier, key):
    """T vector + per-vehicle arc lists from the T-verified partition file."""
    with open(f"{CASE}/{key}_{tier}_vehicles.json") as fh:
        d = json.load(fh)
    routes = [[a for trip in veh for a in trip] for veh in d["vehicles"]]
    return tuple(d["T"]), [r for r in routes if r]


def load_tier(tier):
    inst = np.load(f"{CASE}/{tier}_case.npz", allow_pickle=True)
    geo = np.load(f"{CASE}/{tier}_case_geo.npz", allow_pickle=True)
    return inst["req"], inst["nonreq"], geo["coords"], geo["edges"]


def draw_panel(ax, req, nonreq, coords, routes, lw):
    # background: every arc of the instance graph
    for e in np.vstack([req[:, :2], nonreq[:, :2]]).astype(int):
        p, q = coords[e[0]], coords[e[1]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BG, lw=0.9, zorder=1,
                solid_capstyle="round")
    # perpendicular-right offset so antiparallel serviced arcs don't overdraw
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
    off = 0.007 * span
    for vid, arcs in enumerate(routes):
        for a in arcs:
            t, h, cls = int(req[a - 1, 0]), int(req[a - 1, 1]), int(req[a - 1, 3])
            p, q = coords[t], coords[h]
            v = q - p
            n = np.hypot(*v)
            if n == 0:
                continue
            perp = np.array([v[1], -v[0]]) / n * off
            p2, q2 = p + perp, q + perp
            ax.plot([p2[0], q2[0]], [p2[1], q2[1]], color=CLASS_COLORS[cls],
                    lw=lw, ls=VEH_STYLES[vid], zorder=2 + (4 - cls),
                    solid_capstyle="round", dash_capstyle="round", alpha=0.95)
    ax.plot(*coords[0], marker="*", color="black", ms=13, zorder=10,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    m = 0.04 * span
    ax.set_xlim(coords[:, 0].min() - m, coords[:, 0].max() + m)
    ax.set_ylim(coords[:, 1].min() - m, coords[:, 1].max() + m)


fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.6))
summary = {}
for r, (tier, row_label) in enumerate(TIERS):
    req, nonreq, coords, edges = load_tier(tier)
    n_req = len(req)
    for c, (key, name) in enumerate(ALGS):
        Ts, routes = load_solution(tier, key)
        served = [a for arcs in routes for a in arcs]
        assert sorted(served) == list(range(1, n_req + 1)), \
            f"{key}/{tier}: routes do not serve each required arc exactly once"
        assert len(routes) <= 3, f"{key}/{tier}: more than 3 vehicles"
        ax = axes[r, c]
        draw_panel(ax, req, nonreq, coords, routes, lw=2.4 if tier == "small" else 1.7)
        ax.set_title(f"{name}\n$(T_1,T_2,T_3)=({Ts[0]:.2f},\\,{Ts[1]:.2f},\\,{Ts[2]:.2f})$",
                     fontsize=11, pad=6)
        if c == 0:
            ax.set_ylabel(row_label, fontsize=10.5, labelpad=10)
        # class composition per vehicle, for the prose
        comp = [[sum(1 for a in arcs if int(req[a - 1, 3]) == k) for k in (1, 2, 3)]
                for arcs in routes]
        summary[(tier, key)] = dict(T=Ts, n_routes=len(routes), comp=comp)

handles = (
    [Line2D([], [], color=CLASS_COLORS[k], lw=3, label=f"class {k}") for k in (1, 2, 3)]
    + [Line2D([], [], color="#444444", lw=2, ls=VEH_STYLES[i], label=VEH_NAMES[i])
       for i in range(3)]
    + [Line2D([], [], color="black", marker="*", ls="none", ms=12,
              markeredgecolor="white", markeredgewidth=0.6, label="depot"),
       Line2D([], [], color=BG, lw=2, label="non-serviced road")]
)
fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.005), handlelength=2.6,
           columnspacing=1.4)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight")
print(f"Saved -> {OUT}")

for k, v in summary.items():
    print(k, "T=", tuple(round(t, 3) for t in v["T"]), "routes=", v["n_routes"],
          "class-composition per vehicle [c1,c2,c3]:", v["comp"])
