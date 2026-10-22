"""Figure 3 - openness ratio distribution by neckline type."""
import sys
sys.path.insert(0, "/home/claude")
from figlib import *
import pandas as pd
from matplotlib.patches import Patch

df = pd.read_csv("/mnt/user-data/uploads/recommendations_v4_final__1_.csv")
unseen = df[~df["in_train"]].copy()

COLLAR = ["Mandarin Collar", "Band Collar", "Shirt Collar"]
counts = unseen["neckline_type"].value_counts()
keep = [t for t in counts.index if counts[t] >= 20]
sub = unseen[unseen["neckline_type"].isin(keep)]
order = sub.groupby("neckline_type")["openness_ratio"].median().sort_values().index.tolist()

q30, q70 = 1.773, 2.082
XLO, XHI = 0.7, 3.5

fig, ax = plt.subplots(figsize=(7.16, 4.15))
fig.subplots_adjust(left=0.235, right=0.975, top=0.915, bottom=0.115)

ax.set_xlim(XLO, XHI)
ax.set_ylim(-0.75, len(order) - 0.05)

ax.axvspan(XLO, q30, color=tint(VIOLET, 0.955), zorder=0)
ax.axvspan(q30, q70, color=tint(TEAL, 0.945), zorder=0)
ax.axvspan(q70, XHI, color=tint(GOLD, 0.940), zorder=0)

data = [unseen.loc[unseen["neckline_type"] == t, "openness_ratio"].values for t in order]
pos = np.arange(len(order))
bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.60,
                patch_artist=True, showfliers=False, zorder=3,
                medianprops=dict(color=INK, lw=1.7),
                whiskerprops=dict(color=MUTED, lw=1.0),
                capprops=dict(color=MUTED, lw=1.0))
for t, patch in zip(order, bp["boxes"]):
    c = ROSE if t in COLLAR else BLUE
    patch.set(facecolor=tint(c, 0.58), edgecolor=c, linewidth=1.25)

ytop = len(order) - 0.05
for x, col in ((q30, VIOLET), (q70, GOLD)):
    ax.axvline(x, color=col, lw=1.3, ls=(0, (4, 2)), zorder=4)
ax.text(q30 - 0.035, -0.62, "$q_{30}$ = 1.773", fontsize=7.0, color=VIOLET,
        fontweight="bold", ha="right", va="center", zorder=6)
ax.text(q70 + 0.035, -0.62, "$q_{70}$ = 2.082", fontsize=7.0, color=GOLD,
        fontweight="bold", ha="left", va="center", zorder=6)

for xm, lab, col in (((XLO + q30) / 2, "Narrow / deep", VIOLET),
                     ((q30 + q70) / 2, "Balanced", TEAL),
                     ((q70 + XHI) / 2, "Wide / shallow", GOLD)):
    ax.text(xm, ytop - 0.18, lab, fontsize=7.3, color=col, fontweight="bold",
            ha="center", va="top", zorder=6)

ax.set_yticks(pos)
ax.set_yticklabels([f"{t}   (n = {counts[t]:,})" for t in order], fontsize=7.4)
ax.set_xlabel("Openness ratio   (neckline box width / box height)",
              fontsize=8.2, color=INK_2, labelpad=4)
ax.tick_params(axis="x", labelsize=7.4, colors=MUTED, length=3)
ax.tick_params(axis="y", length=0, colors=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(HAIRLINE)
ax.xaxis.grid(True, color="#e1e0d9", lw=0.7)
ax.set_axisbelow(True)

ax.legend(handles=[
    Patch(facecolor=tint(BLUE, .58), edgecolor=BLUE, label="Open neckline, routed by geometry"),
    Patch(facecolor=tint(ROSE, .58), edgecolor=ROSE, label="Closed collar, routed by type"),
], loc="lower right", bbox_to_anchor=(1.0, 0.015), fontsize=7.0, frameon=False,
   handlelength=1.3, handleheight=0.9)

fig.savefig("/mnt/user-data/outputs/Fig3_openness_by_type.png", dpi=400)
print("saved Fig3 |", len(order), "types | excluded:",
      [t for t in counts.index if counts[t] < 20])
