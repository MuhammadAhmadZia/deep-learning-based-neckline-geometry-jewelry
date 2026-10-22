"""Figure 1 - system overview (Introduction). Sized for IJACSA full page width."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figlib import *
import numpy as np
from PIL import Image

_g = Image.open("../figures/Fig4_qualitative_detections.png").convert("RGB")
GW, GH = _g.size
CW, CH = GW // 5, GH // 4

def real_photo(ax, x, y, s, color, rc=(0, 0), keep_top=0.52, max_w=12.0):
    r, c = rc
    im = _g.crop((c * CW, r * CH + int(0.13 * CH), (c + 1) * CW, (r + 1) * CH))
    a = np.asarray(im).astype(int)
    nw = (a.sum(axis=2) < 720)
    rows, cols = np.where(nw.any(axis=1))[0], np.where(nw.any(axis=0))[0]
    im = im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    w0, h0 = im.size
    im = im.crop((0, 0, w0, int(h0 * keep_top)))
    w0, h0 = im.size
    sc = min(max_w / w0, s * 1.02 / h0)
    w, h = w0 * sc, h0 * sc
    ax.imshow(np.asarray(im), extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
              zorder=5, aspect="auto", interpolation="lanczos")
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor="none", edgecolor=color, linewidth=1.1,
                                zorder=6))

XMAX, YLO, YHI = 100, 2.6, 33.0
W_IN = 7.16
fig, ax = plt.subplots(figsize=(W_IN, W_IN * (YHI - YLO) / XMAX))
ax.set_xlim(0, XMAX); ax.set_ylim(YLO, YHI); ax.set_aspect("equal"); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

BW, BH, BY = 17.0, 19.5, 19.0
xs = [10.0, 30.0, 50.0, 70.0, 90.0]

stages = [
    dict(c=BLUE,   icon=ic_image,    t="Catalog\nImage",
         s="One product photo,\nno manual tagging",        chip=None),
    dict(c=VIOLET, icon=ic_detect,   t="Neckline\nDetection",
         s="YOLOv8n locates the\nneckline region",         chip=("mAP50  0.813", VIOLET)),
    dict(c=TEAL,   icon=ic_measure,  t="Geometry\nMeasurement",
         s="Openness ratio from\nbox width / height",      chip=("17,949 images", TEAL)),
    dict(c=ROSE,   icon=ic_branch,   t="Rule\nRouting",
         s="Collars split off, rest\ncut at quantiles",    chip=("4 classes", ROSE)),
    dict(c=GOLD,   icon=ic_necklace, t="Jewelry\nSuggestion",
         s="Necklace and earring\npairing returned",       chip=("99.2% approved", GOLD)),
]

for i, (x, st) in enumerate(zip(xs, stages)):
    lw = 2.1 if i == len(xs) - 1 else 1.4
    rbox(ax, x, BY, BW, BH, st["c"], fill_amt=0.90, lw=lw)
    st["icon"](ax, x, BY + 5.4, 6.6, st["c"])
    label(ax, x, BY - 1.4, st["t"], size=8.0, weight="bold", color=INK, lh=1.3)
    label(ax, x, BY - 6.2, st["s"], size=6.4, color=INK_2, lh=1.45)
    if st["chip"]:
        chip(ax, x, 5.6, st["chip"][0], st["chip"][1], size=6.2)

for a, b in zip(xs, xs[1:]):
    arrow(ax, (a + BW / 2 + 0.5, BY), (b - BW / 2 - 0.5, BY),
          color=HAIRLINE, lw=1.6, ms=9)

label(ax, (xs[0] + xs[2]) / 2, 31.6, "P E R C E P T I O N", size=6.6,
      weight="bold", color=MUTED)
label(ax, (xs[3] + xs[4]) / 2, 31.6, "D E C I S I O N", size=6.6,
      weight="bold", color=MUTED)
ax.plot([xs[0] - BW / 2, xs[2] + BW / 2], [30.3, 30.3], color=HAIRLINE, lw=0.9)
ax.plot([xs[3] - BW / 2, xs[4] + BW / 2], [30.3, 30.3], color=HAIRLINE, lw=0.9)

fig.savefig("../figures/Fig1_system_overview.png", dpi=400)
print("saved Fig1")
