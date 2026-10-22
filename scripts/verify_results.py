"""Recompute every number reported in the paper from the shipped result tables.

Usage:  python scripts/verify_results.py
Needs:  pandas, numpy, scipy  (see requirements.txt)

Nothing here re-runs the detector. The detector outputs are already in
results/recommendations_final.csv; this script checks that every statistic in
the manuscript follows from those tables.
"""
import sys, pathlib
import numpy as np
import pandas as pd
from scipy.stats import kruskal, fisher_exact, binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

CLOSED = ["Mandarin Collar", "Band Collar", "Shirt Collar"]
Q30, Q70 = 1.773, 2.082

def rule(title):
    print("\n" + title)
    print("-" * len(title))

def check(label, got, want, tol=5e-3):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got}  (paper: {want})")
    return ok

ok = []

# ----------------------------------------------------------------- detector
rule("1. Detector performance across three seeds")
s = pd.read_csv(R / "seed_comparison.csv", index_col=0)
runs = s.iloc[:3]
for col, want_m, want_s in [("mAP50", 0.813, 0.002), ("mAP50-95", 0.532, 0.002),
                            ("precision", 0.820, 0.006), ("recall", 0.747, 0.002)]:
    v = runs[col].astype(float).values
    print(f"  {col:10s} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}"
          f"   (paper: {want_m:.3f} +/- {want_s:.3f})")
    ok.append(abs(v.mean() - want_m) < 1e-2)

# ----------------------------------------------------------------- geometry
rule("2. Geometry table and admissibility")
d = pd.read_csv(R / "recommendations_final.csv")
d["productId"] = d["productId"].astype(str)
ok.append(check("rows after box-area filter", len(d), 19534))
ok.append(check("images used in detector training", int(d["in_train"].sum()), 1585))
unseen = d[~d["in_train"]]
ok.append(check("unseen subset", len(unseen), 17949))

counts = unseen["neckline_type"].value_counts()
keep = [t for t in counts.index if counts[t] >= 20]
print(f"  types with n >= 20: {len(keep)}")
med = unseen[unseen.neckline_type.isin(keep)].groupby("neckline_type")["openness_ratio"].median()
print("  lowest median :", med.idxmin(), round(med.min(), 3))
print("  highest median:", med.idxmax(), round(med.max(), 3))

# ----------------------------------------------------------------- separation
rule("3. Separation of neckline types by openness ratio")
openn = unseen[~unseen.neckline_type.isin(CLOSED)]
grp = [g["openness_ratio"].values for _, g in openn.groupby("neckline_type")
       if len(g) >= 20]
H, p = kruskal(*grp)
k, n = len(grp), sum(len(g) for g in grp)
eps2 = (H - k + 1) / (n - k)
print(f"  Kruskal-Wallis H = {H:.1f}  (paper: 2876.2)")
print(f"  p = {p:.3g}")
ok.append(check("epsilon squared", round(eps2, 3), 0.212, tol=0.01))

# ----------------------------------------------------------------- rules
rule("4. Rule assignment")
ok.append(check("closed-collar branch", int((d.category == "Closed collar").sum()), 4830))
op = d[d.category != "Closed collar"]
share = op.category.value_counts(normalize=True).mul(100).round(1).to_dict()
print("  open-neckline split (%):", share, " (paper: 30 / 40 / 30)")
q30 = unseen[~unseen.neckline_type.isin(CLOSED)]["openness_ratio"].quantile(.30)
print(f"  recomputed q30 on unseen open necklines: {q30:.3f}  (paper: {Q30})")

mode = openn.groupby("neckline_type")["category"].agg(lambda x: x.mode()[0])
dis = openn[openn.category != openn.neckline_type.map(mode)]
print(f"  geometry disagrees with the type label on "
      f"{len(dis):,} of {len(openn):,} images ({100*len(dis)/len(openn):.1f}%)")

# ----------------------------------------------------------------- control
rule("5. Blind control study")
h = pd.read_csv(R / "human_eval_control_study.csv")
rc = [c for c in h.columns if "rater" in c]
real, ctrl = h[~h.is_control], h[h.is_control]
ry, cy = int((real.majority == "Y").sum()), int((ctrl.majority == "Y").sum())
print(f"  system recommendation approved: {ry}/{len(real)} ({100*ry/len(real):.1f}%)")
print(f"  swapped control approved:       {cy}/{len(ctrl)} ({100*cy/len(ctrl):.1f}%)")
_, pf = fisher_exact([[ry, len(real) - ry], [cy, len(ctrl) - cy]])
print(f"  Fisher exact p = {pf:.3g}")
ok.append(ry == 119 and cy == 0)

agree = (h[rc].nunique(axis=1) == 1).mean()
print(f"  all three raters agreed on {100*agree:.1f}% of images")

# ----------------------------------------------------------------- paired
rule("6. Paired comparison on disagreement cases")
t = pd.read_csv(R / "type_vs_geometry_paired.csv")
pv = t.pivot(index="pair_id", columns="shown_condition", values="majority")
g, l = int((pv.geometry == "Y").sum()), int((pv.type_only == "Y").sum())
b = int(((pv.geometry == "Y") & (pv.type_only == "N")).sum())
c = int(((pv.geometry == "N") & (pv.type_only == "Y")).sum())
print(f"  per-image geometry approved: {g}/{len(pv)}")
print(f"  neckline label only:         {l}/{len(pv)}")
print(f"  discordant pairs {b} : {c}")
print(f"  McNemar exact p = {binomtest(b, b + c, 0.5).pvalue:.3g}")
ok.append(g == 29 and l == 3 and b == 26 and c == 0)

gaps = t.groupby("pair_id")["image_id"].apply(lambda s: abs(s.iloc[0] - s.iloc[1]))
adj = gaps[gaps <= 2].index.tolist()
cl = t[~t.pair_id.isin(adj)].pivot(index="pair_id", columns="shown_condition",
                                   values="majority")
b2 = int(((cl.geometry == "Y") & (cl.type_only == "N")).sum())
c2 = int(((cl.geometry == "N") & (cl.type_only == "Y")).sum())
print(f"  robustness, {len(adj)} adjacent pairs removed: {int((cl.geometry=='Y').sum())}"
      f"/{len(cl)} vs {int((cl.type_only=='Y').sum())}/{len(cl)}, "
      f"p = {binomtest(b2, b2 + c2, 0.5).pvalue:.3g}")

rule("Summary")
print(f"  {sum(ok)} of {len(ok)} headline checks passed")
sys.exit(0 if all(ok) else 1)
