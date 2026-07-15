import re, json, sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

BASE = "/tmp/claude-1002/-home-quangnm-projects-ArcRoute/37a29464-0515-4e22-912c-b17235383766/scratchpad/paper_bm"

def parse(path):
    T, times = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'\[\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\]\]\s*:::\s*([\d.eE+-]+)', line)
            if m:
                T.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                times.append(float(m.group(4)))
    return np.array(T), np.array(times)

LOGS = {
    ('small','P'): {'ILS': 'logs/bm_ils_P_small_20260706_111028.out','EA':'logs/bm_ea_P_small_20260706_112415.out','ACO':'logs/bm_aco_P_small_20260707_235156.out'},
    ('small','U'): {'ILS': 'logs/bm_ils_U_small_20260706_111035.out','EA':'logs/bm_ea_U_small_20260706_112847.out','ACO':'logs/bm_aco_U_small_20260707_235538.out'},
    ('medium','P'): {'ILS': 'logs/bm_ils_P_medium_20260707_002906.out','EA':'logs/bm_ea_P_medium_20260707_003934.out','ACO':'logs/bm_aco_P_medium_20260708_000355.out'},
    ('medium','U'): {'ILS': 'logs/bm_ils_U_medium_20260707_002957.out','EA':'logs/bm_ea_U_medium_20260707_004425.out','ACO':'logs/bm_aco_U_medium_20260708_001056.out'},
    ('large','P'): {'ILS': 'logs/bm_ils_P_large_20260707_012513.out','EA':'logs/bm_ea_P_large_20260707_012618.out','ACO':'logs/bm_aco_P_large_20260708_002712.out'},
    ('large','U'): {'ILS': 'logs/bm_ils_U_large_20260707_012529.out','EA':'logs/bm_ea_U_large_20260707_013308.out','ACO':'logs/bm_aco_U_large_20260708_004343.out'},
}

COLORS = {"HARMONY": "#0072B2", "ILS": "#E69F00", "EA": "#009E73", "ACO": "#D55E00"}
MARKERS = {"ILS": "s", "EA": "^", "ACO": "D"}

TIERS = ["small", "medium", "large"]
VARIANTS = ["P", "U"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

for vi, variant in enumerate(VARIANTS):
    for ti, tier in enumerate(TIERS):
        ax = axes[vi, ti]

        # HARMONY R-sweep only covers the first N_SWEEP instances (tractability);
        # restrict the baselines to that SAME subset so the comparison is apples-to-apples.
        d = json.load(open(f"{BASE}/rsweep_{tier}_{variant}.json"))
        R_GRID = d["R_GRID"]
        N_SWEEP = d["N"]
        h_t1 = [np.array(d["results"][str(R)]["T"])[:, 0].mean() for R in R_GRID]
        h_time = [np.array(d["results"][str(R)]["time"]).mean() for R in R_GRID]

        # reference solvers at their own fixed operating point (absolute T1 + time),
        # sliced to the same first-N_SWEEP instances as the HARMONY R-sweep
        ref = {}
        for name, path in LOGS[(tier, variant)].items():
            T, t = parse(f"/home/quangnm/projects/ArcRoute/{path}")
            ref[name] = (float(T[:N_SWEEP, 0].mean()), float(t[:N_SWEEP].mean()))

        # best-found reference across ILS/EA/ACO/HARMONY(all R) at THIS instance set,
        # for gap% -- consistent definition with tab:planned (min across compared configs)
        best_t1 = min([ref[n][0] for n in ref] + h_t1)

        def gap(x):
            return (x - best_t1) / best_t1 * 100

        ax.plot(h_time, [gap(x) for x in h_t1], "-o", color=COLORS["HARMONY"],
                linewidth=2, markersize=5, label="HARMONY ($R$ swept)", zorder=3)
        for R, tm, t1 in zip(R_GRID, h_time, h_t1):
            if R in (1, 256):
                ax.annotate(f"$R$={R}", (tm, gap(t1)), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color=COLORS["HARMONY"])

        for name in ["ILS", "EA", "ACO"]:
            t1, tm = ref[name]
            ax.scatter([tm], [gap(t1)], color=COLORS[name], marker=MARKERS[name],
                       s=55, zorder=4, label=name, edgecolor="white", linewidth=0.5)

        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{tier} / HDCARP-{variant}", fontsize=10)
        if ti == 0:
            ax.set_ylabel("$T_1$ gap vs. best found (%)")
        if vi == 1:
            ax.set_xlabel("Time / instance (s, log scale)")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=-2)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Time--quality trade-off: HARMONY's anytime curve vs. single-operating-point baselines", fontsize=11, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{BASE}/anytime_curve_v2.png", dpi=200, bbox_inches="tight")
print("saved v2")
