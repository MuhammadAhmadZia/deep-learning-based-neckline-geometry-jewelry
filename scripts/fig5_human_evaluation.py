"""Figure 4 - human evaluation: blind control study and paired comparison."""
import sys
sys.path.insert(0, "/home/claude")
from figlib import *
import pandas as pd

h = pd.read_csv("/mnt/user-data/uploads/human_eval_v2_final.csv")
t = pd.read_csv("/mnt/user-data/uploads/type_vs_geometry_final.csv")

real = h[~h["is_control"]]; ctrl = h[h["is_control"]]
a_vals = [(real["majority"] == "Y").mean() * 100, (ctrl["majority"] == "Y").mean() * 100]
a_hit = [int((real["majority"] == "Y").sum()), int((ctrl["majority"] == "Y").sum())]
a_n = [len(real), len(ctrl)]

p = t.pivot(index="pair_id", columns="shown_condition", values="majority")
b_vals = [(p["geometry"] == "Y").mean() * 100, (p["type_only"] == "Y").mean() * 100]
b_hit = [int((p["geometry"] == "Y").sum()), int((p["type_only"] == "Y").sum())]
b_n = [len(p), len(p)]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85))
fig.subplots_adjust(left=0.142, right=0.988, top=0.715, bottom=0.255, wspace=0.78)

PANELS = [
    dict(ax=axes[0], title="(a)  Blind control study",
         sub="150 images, 3 independent raters, majority vote",
         labels=["System\nrecommendation", "Swapped\ncontrol"],
         vals=a_vals, hit=a_hit, n=a_n, cols=[TEAL, RED],
         stat="Fisher exact  $p$ < 1$\\times$10$^{-6}$ ;  Fleiss $\\kappa$ = 0.932"),
    dict(ax=axes[1], title="(b)  Paired comparison",
         sub="30 disagreement pairs, same raters and images",
         labels=["Per-image\ngeometry", "Neckline\nlabel only"],
         vals=b_vals, hit=b_hit, n=b_n, cols=[TEAL, VIOLET],
         stat="McNemar exact  $p$ = 2.98$\\times$10$^{-8}$ ;  26 : 0 split"),
]

for P in PANELS:
    ax = P["ax"]
    y = np.arange(2)[::-1]
    for yy, v, c, hit, n in zip(y, P["vals"], P["cols"], P["hit"], P["n"]):
        ax.barh(yy, v, height=0.44, color=tint(c, 0.22), edgecolor=c, linewidth=1.2,
                zorder=3)
        txt = f"{v:.1f}%  ({hit}/{n})"
        if v >= 50:
            ax.text(v - 2.5, yy, txt, fontsize=7.2, va="center", ha="right",
                    color="white", fontweight="bold", zorder=5)
        else:
            ax.text(max(v, 0) + 2.5, yy, txt, fontsize=7.2, va="center", ha="left",
                    color=INK, fontweight="bold", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(P["labels"], fontsize=7.2, linespacing=1.35)
    ax.set_xlim(0, 100); ax.set_ylim(-0.62, 1.62)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Rated appropriate by majority (%)", fontsize=7.3, color=INK_2,
                  labelpad=3)
    ax.tick_params(axis="x", labelsize=7.0, colors=MUTED, length=3)
    ax.tick_params(axis="y", length=0, colors=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.xaxis.grid(True, color="#e1e0d9", lw=0.7)
    ax.set_axisbelow(True)
    ax.text(0, 1.30, P["title"], transform=ax.transAxes, fontsize=8.1,
            fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0, 1.135, P["sub"], transform=ax.transAxes, fontsize=6.6,
            color=MUTED, ha="left", va="center")
    ax.text(0, -0.44, P["stat"], transform=ax.transAxes, fontsize=6.5,
            color=INK_2, ha="left", va="center")

fig.savefig("/mnt/user-data/outputs/Fig4_human_evaluation.png", dpi=400)
print("saved Fig4")
