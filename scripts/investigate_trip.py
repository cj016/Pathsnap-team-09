"""Why are a trip's labels uncertain? Pointer lag vs GPS genuinely off the reference route.

  python scripts/investigate_trip.py train/00000054
Saves results/trip_<id>_investigation.csv (one row of evidence).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import DATA, arc_lines, label_points, load_trip, project_trip  # noqa: E402


def nearest_dist(points, geoms):
    """Distance from each point to the nearest geometry (no pointer, no window)."""
    _, d = shapely.STRtree(geoms).query_nearest(points, return_distance=True, all_matches=False)
    return d


def investigate(trip_dir):
    trip = project_trip(load_trip(trip_dir))
    lines = arc_lines(trip)
    trip = label_points(trip, lines)
    t, route = trip["track"], trip["route"]
    pts = shapely.points(t[["x", "y"]].to_numpy())
    unc = t["label_uncertain"].to_numpy()

    d_route = nearest_dist(pts, lines[route])  # to ANY reference-route arc
    d_net = nearest_dist(pts, lines)           # to ANY arc of the road network
    d_end = nearest_dist(pts, lines[route[-5:]])
    route_end_idx = int(np.argmin(d_end))      # GPS point where the car reaches the end of the route
    step = np.hypot(np.diff(t["x"]), np.diff(t["y"]))
    dt = np.diff(t["timestamp"])
    after = slice(route_end_idx, None)
    return {
        "trip_id": trip_dir.name,
        "n_points": len(t),
        "n_uncertain": int(unc.sum()),
        "uncertain_frac": unc.mean(),
        # pointer lag = uncertain point that IS within 50 m of some route arc
        "uncertain_within_50m_of_any_route_arc": int((d_route[unc] <= 50).sum()),
        "uncertain_gt_50m_from_every_route_arc": int((d_route[unc] > 50).sum()),
        "median_dist_uncertain_to_route_m": float(np.median(d_route[unc])),
        "median_dist_uncertain_to_network_m": float(np.median(d_net[unc])),
        "all_points_within_50m_of_network_frac": float((d_net <= 50).mean()),
        "route_km": shapely.length(lines[route]).sum() / 1000,
        "track_km": step.sum() / 1000,
        "pointer_final_route_index": int(np.flatnonzero(route == t["label"].iloc[-1])[0]),
        "n_route_arcs": len(route),
        "route_end_reached_at_point": route_end_idx,
        "route_end_min_dist_m": float(d_end.min()),
        "hours_before_route_end": dt[:route_end_idx].sum() / 3600,
        "hours_after_route_end": dt[after].sum() / 3600,
        "km_after_route_end": step[after].sum() / 1000,
        "uncertain_frac_after_route_end": unc[route_end_idx:].mean(),
        "uncertain_frac_by_track_decile": [round(x, 2) for x in pd.Series(unc).groupby(np.arange(len(unc)) * 10 // len(unc)).mean()],
    }


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else "train/00000054"
    res = investigate(DATA / spec)
    out = ROOT / "results" / f"trip_{res['trip_id']}_investigation.csv"
    pd.DataFrame([res]).to_csv(out, index=False)
    print(pd.Series(res).to_string())
    print("saved", out.relative_to(ROOT))
