"""Shared drawing helpers for the neckline-jewellery paper figures.

Palette is the validated categorical set, with slots chosen so the domain
reads correctly: gold for jewellery, teal for measurement, blue for data.
All colours are print-safe and pass CVD adjacent-pair checks.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle, Polygon,
                                Rectangle, Arc, Wedge, Ellipse, PathPatch)
from matplotlib.path import Path
import numpy as np

# ---------------------------------------------------------------- palette
INK        = "#0b0b0b"
INK_2      = "#52514e"
MUTED      = "#898781"
SURFACE    = "#fcfcfb"
HAIRLINE   = "#c3c2b7"

BLUE       = "#2a78d6"   # data
VIOLET     = "#6250d6"   # model / training
TEAL       = "#1baf7a"   # geometry / measurement
GOLD       = "#b07d00"   # jewellery  (darkened from #eda100 for print contrast)
GOLD_BR    = "#eda100"   # jewellery bright, for icon fills
ROSE       = "#d55181"   # human evaluation
RED        = "#d03b3b"   # excluded / filtered

def tint(hex_colour, amount=0.90):
    """Blend a hue toward the chart surface to get a soft fill."""
    c = np.array(matplotlib.colors.to_rgb(hex_colour))
    s = np.array(matplotlib.colors.to_rgb(SURFACE))
    return tuple(c + (s - c) * amount)

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})

# ---------------------------------------------------------------- shapes
def rbox(ax, x, y, w, h, colour, fill_amt=0.90, lw=1.6, z=2, ls="solid", pad=0.0):
    """Rounded box centred at (x, y)."""
    b = FancyBboxPatch((x - w / 2 + pad, y - h / 2 + pad), w - 2 * pad, h - 2 * pad,
                       boxstyle="round,pad=0,rounding_size=1.1",
                       linewidth=lw, edgecolor=colour, facecolor=tint(colour, fill_amt),
                       linestyle=ls, zorder=z)
    ax.add_patch(b)
    return b

def panel(ax, x, y, w, h, colour, z=1, fill_amt=0.955):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0,rounding_size=1.4",
                       linewidth=0, facecolor=tint(colour, fill_amt), zorder=z)
    ax.add_patch(b)
    return b

def arrow(ax, p0, p1, colour=MUTED, lw=1.7, z=3, style="-|>", ms=9, rad=0.0, ls="solid"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=colour, zorder=z,
                        shrinkA=0, shrinkB=0, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a

def label(ax, x, y, s, size=9, weight="normal", colour=INK, ha="center", va="center",
          z=5, style="normal", lh=1.25):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=colour,
                   ha=ha, va=va, zorder=z, linespacing=lh, fontstyle=style)

def chip(ax, x, y, s, colour, size=7.6, padx=1.0, pady=0.55, z=5):
    """Small pill-shaped metric tag."""
    t = ax.text(x, y, s, fontsize=size, fontweight="bold", color="white",
                ha="center", va="center", zorder=z + 1,
                bbox=dict(boxstyle=f"round,pad=0.38", facecolor=colour,
                          edgecolor="none"))
    return t

# ---------------------------------------------------------------- icons
def ic_garment(ax, x, y, s, colour):
    """Kurta / dress silhouette with a visible V neckline."""
    body = Polygon([(x - .42 * s, y - .55 * s), (x - .34 * s, y + .22 * s),
                    (x - .52 * s, y + .30 * s), (x - .40 * s, y + .50 * s),
                    (x - .16 * s, y + .56 * s), (x, y + .22 * s),
                    (x + .16 * s, y + .56 * s), (x + .40 * s, y + .50 * s),
                    (x + .52 * s, y + .30 * s), (x + .34 * s, y + .22 * s),
                    (x + .42 * s, y - .55 * s)],
                   closed=True, facecolor=tint(colour, .62), edgecolor=colour,
                   linewidth=1.4, zorder=4, joinstyle="round")
    ax.add_patch(body)

def ic_stack(ax, x, y, s, colour):
    """Stacked data layers."""
    for i, dy in enumerate([-.34, 0.0, .34]):
        e = Ellipse((x, y + dy * s), .86 * s, .30 * s,
                    facecolor=tint(colour, .55 - i * .12), edgecolor=colour,
                    linewidth=1.3, zorder=4 + i)
        ax.add_patch(e)

def ic_detect(ax, x, y, s, colour):
    """Dashed detection box with corner ticks over a neckline curve."""
    ax.add_patch(Rectangle((x - .46 * s, y - .30 * s), .92 * s, .56 * s,
                           facecolor="none", edgecolor=colour, linewidth=1.5,
                           linestyle=(0, (3, 2)), zorder=4))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([x + sx * .46 * s, x + sx * .30 * s],
                    [y + sy * .30 * s - (.02 * s if sy < 0 else 0) - (.0 if sy > 0 else 0),
                     y + sy * .30 * s - (.02 * s if sy < 0 else 0)],
                    color=colour, lw=2.4, solid_capstyle="round", zorder=5)
            ax.plot([x + sx * .46 * s, x + sx * .46 * s],
                    [y + sy * .30 * s, y + sy * .14 * s],
                    color=colour, lw=2.4, solid_capstyle="round", zorder=5)
    v = Polygon([(x - .26 * s, y + .22 * s), (x, y - .20 * s), (x + .26 * s, y + .22 * s)],
                closed=False, facecolor="none", edgecolor=colour, linewidth=2.0,
                zorder=6, joinstyle="round")
    ax.add_patch(v)

def ic_measure(ax, x, y, s, colour):
    """Width / height calipers around a neckline opening."""
    v = Polygon([(x - .28 * s, y + .30 * s), (x, y - .22 * s), (x + .28 * s, y + .30 * s)],
                closed=False, facecolor="none", edgecolor=tint(colour, .35),
                linewidth=2.0, zorder=4, joinstyle="round")
    ax.add_patch(v)
    # width bar
    ax.annotate("", xy=(x + .30 * s, y + .44 * s), xytext=(x - .30 * s, y + .44 * s),
                arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=1.5,
                                mutation_scale=7, shrinkA=0, shrinkB=0), zorder=5)
    # height bar
    ax.annotate("", xy=(x + .44 * s, y - .24 * s), xytext=(x + .44 * s, y + .32 * s),
                arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=1.5,
                                mutation_scale=7, shrinkA=0, shrinkB=0), zorder=5)
    ax.text(x, y + .60 * s, "w", fontsize=7.5, color=colour, ha="center",
            va="center", fontweight="bold", zorder=6)
    ax.text(x + .62 * s, y + .04 * s, "h", fontsize=7.5, color=colour, ha="center",
            va="center", fontweight="bold", zorder=6)

def ic_branch(ax, x, y, s, colour):
    """A decision split into three routes."""
    ax.add_patch(Circle((x - .40 * s, y), .11 * s, facecolor=colour,
                        edgecolor="none", zorder=5))
    for dy in (.34, 0.0, -.34):
        ax.add_patch(FancyArrowPatch((x - .30 * s, y), (x + .34 * s, y + dy * s),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.5, color=colour, zorder=4,
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={-dy*0.45}"))
        ax.add_patch(Circle((x + .42 * s, y + dy * s), .075 * s,
                            facecolor=tint(colour, .4), edgecolor=colour,
                            linewidth=1.1, zorder=5))

def ic_necklace(ax, x, y, s, colour, bright=None):
    """Chain arc with a pendant - the jewellery glyph."""
    bright = bright or colour
    ax.add_patch(Arc((x, y + .30 * s), .86 * s, .86 * s, theta1=196, theta2=344,
                     edgecolor=colour, linewidth=2.1, zorder=4))
    ax.add_patch(Circle((x, y - .17 * s), .135 * s, facecolor=tint(bright, .25),
                        edgecolor=colour, linewidth=1.4, zorder=5))
    for sx in (-1, 1):
        ax.add_patch(Circle((x + sx * .415 * s, y + .295 * s), .055 * s,
                            facecolor=colour, edgecolor="none", zorder=5))

def ic_raters(ax, x, y, s, colour):
    """Three independent assessors."""
    for i, dx in enumerate((-.34, 0.0, .34)):
        ax.add_patch(Circle((x + dx * s, y + .20 * s), .115 * s,
                            facecolor=tint(colour, .35), edgecolor=colour,
                            linewidth=1.3, zorder=5))
        w = Wedge((x + dx * s, y - .04 * s), .20 * s, 20, 160,
                  facecolor=tint(colour, .55), edgecolor=colour, linewidth=1.3,
                  zorder=4)
        ax.add_patch(w)

def ic_stats(ax, x, y, s, colour):
    """Small bar trio for the statistical arm."""
    for i, (dx, h) in enumerate(zip((-.28, 0.0, .28), (.34, .58, .44))):
        ax.add_patch(FancyBboxPatch((x + dx * s - .085 * s, y - .32 * s),
                                    .17 * s, h * s,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=tint(colour, .45 - i * .08),
                                    edgecolor=colour, linewidth=1.2, zorder=4))
    ax.plot([x - .44 * s, x + .44 * s], [y - .33 * s, y - .33 * s],
            color=colour, lw=1.6, solid_capstyle="round", zorder=5)

def ic_funnel(ax, x, y, s, colour):
    """Quality filter."""
    f = Polygon([(x - .44 * s, y + .38 * s), (x + .44 * s, y + .38 * s),
                 (x + .12 * s, y - .02 * s), (x + .12 * s, y - .40 * s),
                 (x - .12 * s, y - .26 * s), (x - .12 * s, y - .02 * s)],
                closed=True, facecolor=tint(colour, .55), edgecolor=colour,
                linewidth=1.4, zorder=4, joinstyle="round")
    ax.add_patch(f)

def ic_net(ax, x, y, s, colour):
    """Compact detector-network glyph."""
    cols = [(-.36, (.26, 0.0, -.26)), (0.0, (.34, 0.0, -.34)), (.36, (.17, -.17))]
    pts = {}
    for cx, ys in cols:
        for cy in ys:
            pts.setdefault(cx, []).append((x + cx * s, y + cy * s))
    keys = sorted(pts)
    for a, b in zip(keys, keys[1:]):
        for p in pts[a]:
            for q in pts[b]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=tint(colour, .55),
                        lw=0.8, zorder=3)
    for k in keys:
        for p in pts[k]:
            ax.add_patch(Circle(p, .072 * s, facecolor=colour, edgecolor="none",
                                zorder=5))

def ic_image(ax, x, y, s, colour):
    """Catalogue photo."""
    ax.add_patch(FancyBboxPatch((x - .42 * s, y - .38 * s), .84 * s, .76 * s,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=tint(colour, .70), edgecolor=colour,
                                linewidth=1.4, zorder=4))
    ax.add_patch(Circle((x - .18 * s, y + .16 * s), .085 * s, facecolor=colour,
                        edgecolor="none", zorder=5))
    m = Polygon([(x - .36 * s, y - .18 * s), (x - .08 * s, y + .06 * s),
                 (x + .10 * s, y - .10 * s), (x + .36 * s, y + .18 * s),
                 (x + .36 * s, y - .32 * s), (x - .36 * s, y - .32 * s)],
                closed=True, facecolor=colour, edgecolor="none", zorder=5)
    ax.add_patch(m)

def new_canvas(w_in, h_in, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax
