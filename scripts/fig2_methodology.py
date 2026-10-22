"""Figure 2 - methodology pipeline with worked examples from the dataset."""
import sys
sys.path.insert(0, "/home/claude")
from figlib import *
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- photos
GRID = "/mnt/user-data/uploads/openness_extremes_check__1_.png"
_g = Image.open(GRID).convert("RGB")
GW, GH = _g.size
CW, CH = GW // 5, GH // 4

def cell_photo(r, c, keep_top=0.55):
    """Crop one grid cell, trim to the photo, then keep the upper body so the
    detection box is large enough to read at print size."""
    im = _g.crop((c * CW, r * CH + int(0.13 * CH), (c + 1) * CW, (r + 1) * CH))
    a = np.asarray(im).astype(int)
    nonwhite = (a.sum(axis=2) < 720)
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    if len(rows) and len(cols):
        im = im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    w, h = im.size
    return im.crop((0, 0, w, int(h * keep_top)))

def place_photo(ax, img, x, y, max_w, max_h, edge):
    w_px, h_px = img.size
    scale = min(max_w / w_px, max_h / h_px)
    w, h = w_px * scale, h_px * scale
    ax.imshow(np.asarray(img), extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
              zorder=5, aspect="auto", interpolation="lanczos")
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor="none", edgecolor=edge, linewidth=1.1,
                                zorder=6))
    return w, h

# ---------------------------------------------------------------- canvas
XMAX, YLO, YHI = 100, 1.0, 126.0
W_IN = 7.16
fig, ax = plt.subplots(figsize=(W_IN, W_IN * (YHI - YLO) / XMAX))
ax.set_xlim(0, XMAX); ax.set_ylim(YLO, YHI); ax.set_aspect("equal"); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

COLS = [27.0, 56.0, 85.0]
BW, BH = 26.0, 13.0
RAIL_X, RAIL_W = 7.0, 9.0
PLANE_L, PLANE_R = RAIL_X - RAIL_W / 2, 98.5
TS, BS, RS = 8.3, 7.1, 6.2          # title / body / rail font sizes

def plane(y, h, c):
    panel(ax, (PLANE_L + PLANE_R) / 2, y, PLANE_R - PLANE_L, h, c, fill_amt=0.955)

def rail(y, h, c, icon, text):
    rbox(ax, RAIL_X, y, RAIL_W, h - 1.8, c, fill_amt=0.80, lw=1.2, z=3)
    icon(ax, RAIL_X, y + 4.0, 5.6, c)
    ax.text(RAIL_X, y - 3.4, text, fontsize=RS, fontweight="bold", color=c,
            ha="center", va="center", zorder=6)

def card(x, y, w, h, c, title, body, ty=3.5, by=-2.3):
    rbox(ax, x, y, w, h, c, fill_amt=0.985, lw=1.25, z=4)
    label(ax, x, y + ty, title, size=TS, weight="bold", colour=INK, z=6)
    label(ax, x, y + by, body, size=BS, colour=INK_2, lh=1.58, z=6)

def chain(y, c):
    for a_, b_ in zip(COLS, COLS[1:]):
        arrow(ax, (a_ + BW / 2 + 0.4, y), (b_ - BW / 2 - 0.4, y),
              colour=c, lw=1.4, ms=8.5, z=6)

# ---------------------------------------------------------------- A  DATA
yA, hA = 116.5, 17.0
plane(yA, hA, BLUE); rail(yA, hA, BLUE, ic_stack, "DATA")
SX, SW, SH = 41.0, 54.0, 6.0
card(SX, yA + 3.6, SW, SH, BLUE, "Fashionpedia",
     "46,781 images, 34,055 with a labelled neckline",
     ty=1.45, by=-1.55)
card(SX, yA - 3.6, SW, SH, BLUE, "Kouture Kurta Set",
     "20,000 kurta photos, 1,996 boxed (Kaggle, MIT)",
     ty=1.45, by=-1.55)
card(COLS[2], yA, BW, BH, BLUE, "Detector Corpus",
     "36,051 images merged\n80 / 10 / 10 split by product ID")
for dy, rad in ((3.6, -0.26), (-3.6, 0.26)):
    arrow(ax, (SX + SW / 2 + 0.4, yA + dy), (COLS[2] - BW / 2 - 0.4, yA),
          colour=BLUE, lw=1.4, ms=8.5, z=6, rad=rad)

# ---------------------------------------------------------------- B  DETECTOR
yB, hB = 97.0, 17.0
plane(yB, hB, VIOLET); rail(yB, hB, VIOLET, ic_net, "DETECTOR")
for x, (t, b) in zip(COLS, [
    ("Training", "YOLOv8n at 640 px\n30 epochs, seeds 0 to 2"),
    ("Held-Out Test", "mAP50  0.813 $\\pm$ 0.002\nFashionpedia 0.800, Kouture 0.995"),
    ("Leakage Audit", "Perceptual hashing across\nthe split: 0 duplicate pairs"),
]):
    card(x, yB, BW, BH, VIOLET, t, b)
chain(yB, VIOLET)

# ---------------------------------------------------------------- C  GEOMETRY
yC, hC = 77.5, 17.0
plane(yC, hC, TEAL); rail(yC, hC, TEAL, ic_measure, "GEOMETRY")
for x, (t, b) in zip(COLS, [
    ("Inference", "20,000 kurta images scored\n19,934 detections (99.7%)"),
    ("Openness Ratio", "box width / box height,\nmeasured in pixel space"),
    ("Admissibility", "Box-area filter $\\rightarrow$ 19,534\nUnseen subset 17,949"),
]):
    card(x, yC, BW, BH, TEAL, t, b)
chain(yC, TEAL)

# ---------------------------------------------------------------- EXAMPLES
yX, hX = 56.0, 21.0
plane(yX, hX, TEAL)
rbox(ax, RAIL_X, yX, RAIL_W, hX - 1.8, TEAL, fill_amt=0.80, lw=1.2, z=3)
ic_garment(ax, RAIL_X, yX + 4.0, 6.2, TEAL)
ax.text(RAIL_X, yX - 3.4, "EXAMPLES", fontsize=RS, fontweight="bold", color=TEAL,
        ha="center", va="center", zorder=6)

EX = [((0, 0), "Off-Shoulder", 11.63), ((0, 4), "Off-Shoulder", 7.68),
      ((2, 0), "V-Neck", 0.61), ((3, 4), "Mandarin Collar", 0.81)]
ex_xs = [27.0, 46.5, 66.0, 85.5]
for (rc, name, val), x in zip(EX, ex_xs):
    place_photo(ax, cell_photo(*rc), x, yX + 2.2, 18.0, 12.8, TEAL)
    label(ax, x, yX - 6.0, f"{name}\nopenness = {val:.2f}", size=6.9,
          colour=INK_2, lh=1.45, z=7)

# ---------------------------------------------------------------- D  RULES
yD, hD = 34.5, 17.0
plane(yD, hD, GOLD); rail(yD, hD, GOLD, ic_branch, "RULES")
for x, (t, b) in zip(COLS, [
    ("Closed-Collar Branch", "Mandarin, Band and Shirt\nrouted by type: 4,830 (24.7%)"),
    ("Quantile Cut Points", "Fitted on open necklines only\n$q_{30}$ = 1.773,  $q_{70}$ = 2.082"),
    ("Four Output Classes", "Narrow 4,411 / Balanced 5,882\nWide 4,411 / Collar 4,830"),
]):
    card(x, yD, BW, BH, GOLD, t, b)
chain(yD, GOLD)

# ---------------------------------------------------------------- E  EVALUATION
yE, hE = 13.0, 21.0
plane(yE, hE, ROSE); rail(yE, hE, ROSE, ic_raters, "EVALUATION")
EB_Y, EB_H = 10.0, 13.0
for x, (t, b) in zip(COLS, [
    ("Separation Test", "Kruskal-Wallis H = 2876\n$\\varepsilon^{2}$ = 0.212"),
    ("Blind Control Study", "150 images, 30 swapped\n99.2% vs 0%, p < 1e-6"),
    ("Paired Comparison", "30 pairs, 29/30 vs 3/30\nMcNemar p = 2.98e-8"),
]):
    card(x, EB_Y, BW, EB_H, ROSE, t, b)
BUS_Y = EB_Y + EB_H / 2 + 4.6
ax.plot([COLS[0], COLS[2]], [BUS_Y, BUS_Y], color=ROSE, lw=1.4, zorder=6,
        solid_capstyle="round")
ax.add_patch(Circle((COLS[1], BUS_Y), 0.8, facecolor=ROSE, edgecolor="none", zorder=7))
for x in COLS:
    arrow(ax, (x, BUS_Y - 0.1), (x, EB_Y + EB_H / 2 + 0.4), colour=ROSE,
          lw=1.4, ms=8.5, z=6)
label(ax, PLANE_R - 1.5, BUS_Y + 2.0, "three independent arms", size=6.2,
      colour=MUTED, ha="right", style="italic")

# ---------------------------------------------------------------- band links
LINKS = [((yA, hA), (yB, hB)), ((yB, hB), (yC, hC)), ((yC, hC), (yX, hX)),
         ((yX, hX), (yD, hD))]
for (yt, ht), (yb, hb) in LINKS:
    arrow(ax, (COLS[1], yt - ht / 2 - 0.15), (COLS[1], yb + hb / 2 + 0.15),
          colour=HAIRLINE, lw=2.0, ms=11, z=6)
arrow(ax, (COLS[1], yD - hD / 2 - 0.15), (COLS[1], BUS_Y + 1.0),
      colour=HAIRLINE, lw=2.0, ms=11, z=6)

fig.savefig("/mnt/user-data/outputs/Fig2_methodology.png", dpi=400)
print("saved Fig2  |  figure height in:", round(W_IN * (YHI - YLO) / XMAX, 2))
