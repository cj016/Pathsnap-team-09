"""Twin-arc figure for student_test points. Same definitions as scripts/verify_results.py:
denominator = test points with at least one candidate; ambiguous = top-2 distances differ by < 1 cm.
Saves reports/figures/twin_arc_analysis.png and results/twin_arc_analysis.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
c = pd.read_parquet(ROOT / "processed" / "candidates_test.parquet")
keys = ["trip_id", "point_idx"]
n_pts = c.groupby(keys).ngroups
top2 = c[c.dist_rank <= 2].pivot_table(index=keys, columns="dist_rank", values="dist_to_arc")
ambiguous = ((top2[2] - top2[1]).abs() < 0.01).sum()
hit1 = c[(c.label == 1) & (c.dist_rank == 1)].groupby(keys).ngroups
hit2 = c[(c.label == 1) & (c.dist_rank <= 2)].groupby(keys).ngroups

res = pd.DataFrame({
    "metric": ["true arc is rank 1", "true arc in top 2", "ambiguous (top-2 distances equal)"],
    "points": [hit1, hit2, ambiguous],
})
res["pct"] = 100 * res["points"] / n_pts
res["denominator_points"] = n_pts
res.to_csv(ROOT / "results" / "twin_arc_analysis.csv", index=False)
print(res.to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 3.8))
bars = ax.bar(res["metric"], res["pct"], color=["#4c72b0", "#55a868", "#c44e52"])
for b, p, n in zip(bars, res["pct"], res["points"]):
    ax.text(b.get_x() + b.get_width() / 2, p + 1.5, f"{p:.1f}%\n({n:,})", ha="center", va="bottom", fontsize=9)
ax.set(ylim=(0, 110), ylabel=f"% of test points (n = {n_pts:,})",
       title="Twin arcs: nearest arc is often a same-geometry duplicate (student_test)")
ax.tick_params(axis="x", labelsize=9)
fig.savefig(ROOT / "reports" / "figures" / "twin_arc_analysis.png", dpi=130, bbox_inches="tight")
print("saved reports/figures/twin_arc_analysis.png")
