"""Person 4 prediction adapter: candidate generation without reading ground truth."""
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from src.data_prep import project_trip, arc_lines
from features.feature_engineering import build_trip_features

RADIUS_METRES = 75.0

def build_candidates(trip_directory):
    directory = Path(trip_directory)
    tid = directory.name
    def read(ext, names):
        return pd.read_csv(directory / f"{tid}.{ext}", sep=r"\s+", header=None, names=names)
    trip = {"trip_id": tid,
            "track": read("track", ["lon", "lat", "timestamp"]),
            "nodes": read("nodes", ["lon", "lat"]),
            "arcs": read("arcs", ["from_node", "to_node"])}
    project_trip(trip)
    lines = arc_lines(trip)
    t = trip["track"]
    pts = shapely.points(t[["x", "y"]].to_numpy())
    x0,y0,x1,y1 = t["x"].min(),t["y"].min(),t["x"].max(),t["y"].max()
    pad = max(150.0, RADIUS_METRES)
    keep = np.flatnonzero(shapely.intersects(lines, shapely.box(x0-pad,y0-pad,x1+pad,y1+pad)))
    if not len(keep):
        return pd.DataFrame()
    pi, ki = shapely.STRtree(lines[keep]).query(pts, predicate="dwithin", distance=RADIUS_METRES)
    ai = keep[ki]
    c = pd.DataFrame({
        "trip_id": tid, "partition": "inference", "point_idx": pi,
        "timestamp":t["timestamp"].to_numpy()[pi],
        "x":t["x"].to_numpy()[pi], "y":t["y"].to_numpy()[pi],
        "arc_id":ai,"dist_to_arc":shapely.distance(pts[pi],lines[ai]),
    })
    c = c.sort_values(["point_idx","dist_to_arc"],ignore_index=True)
    c["dist_rank"] = c.groupby("point_idx").cumcount()+1
    # These placeholders are solely needed by the common feature function's
    # schema check. Never read or infer reference labels during prediction.
    c["label"] = -1
    c["label_uncertain"] = True
    features = build_trip_features(c, trip, lines)
    return features.drop(columns=["label","label_uncertain"])
