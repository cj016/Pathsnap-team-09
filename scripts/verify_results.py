import pandas as pd
from pathlib import Path

R, P = Path("results"), Path("processed")

# Data quality numbers
dq = pd.read_csv(R / "data_quality.csv", dtype={"trip_id": str})
tr, te = dq[dq.partition == "train"], dq[dq.partition == "student_test"]
print("trips train/test:", len(tr), len(te))
print("NaNs:", dq["nan_values"].sum(), "| duplicate timestamps:", dq["duplicate_timestamps"].sum())
print("duplicate points train/test:", tr["duplicate_points"].sum(), te["duplicate_points"].sum())
print("trips with gap>10s train/test:", (tr.n_gaps_gt_10s > 0).sum(), (te.n_gaps_gt_10s > 0).sum())
print("largest gap:", dq.loc[dq.max_gap_s.idxmax(), ["trip_id", "max_gap_s"]].tolist())
print("route breaks train:", tr.route_breaks.sum(), "in", (tr.route_breaks > 0).sum(), "trips | test:", te.route_breaks.sum())

# Twin-arc and top-2 numbers (test)
c = pd.read_parquet(P / "candidates_test.parquet")
keys = ["trip_id", "point_idx"]
n_pts = c.groupby(keys).ngroups
top2 = c[c.dist_rank <= 2].pivot_table(index=keys, columns="dist_rank", values="dist_to_arc")
ambiguous = ((top2[2] - top2[1]).abs() < 0.01).sum()
print(f"ambiguous (top-2 distances equal): {ambiguous / n_pts:.1%}")
hit1 = c[(c.label == 1) & (c.dist_rank == 1)].groupby(keys).ngroups
hit2 = c[(c.label == 1) & (c.dist_rank <= 2)].groupby(keys).ngroups
print(f"true arc is rank 1: {hit1 / n_pts:.1%} | in top 2: {hit2 / n_pts:.1%}")