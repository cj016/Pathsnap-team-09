"""PathSnap data preparation: load, validate, project to UTM, label points, generate candidates."""
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PARTITIONS = ["train", "student_test"]
RADII = [25, 50, 75, 100]


# ---------- Step 2: loading and validation ----------
def list_trips(partition):
    """All trip folders of one partition, sorted by id."""
    return sorted(p for p in (DATA / partition).iterdir() if p.is_dir())


def load_trip(trip_dir):
    """Read the 4 files of one trip into DataFrames (row index = node_id / arc_id)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    read = lambda ext, cols: pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, names=cols)
    return {
        "trip_id": tid,
        "track": read("track", ["lon", "lat", "timestamp"]),
        "nodes": read("nodes", ["lon", "lat"]),
        "arcs": read("arcs", ["from_node", "to_node"]),
        "route": read("route", ["arc_id"])["arc_id"].to_numpy(),
    }


def validate_trip(trip, partition):
    """One row of data-quality checks for a trip."""
    t, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    dt = t["timestamp"].diff().dropna()
    n_nodes, n_arcs = len(nodes), len(arcs)
    route_ok = (route >= 0) & (route < n_arcs)
    # connectivity break = consecutive route arcs that don't share a node (to_node of a != from_node of b)
    r = route[route_ok]
    breaks = int((arcs["to_node"].to_numpy()[r[:-1]] != arcs["from_node"].to_numpy()[r[1:]]).sum()) if len(r) > 1 else 0
    return {
        "trip_id": trip["trip_id"],
        "partition": partition,
        "n_points": len(t),
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route_arcs": len(route),
        "nan_values": int(t.isna().sum().sum() + nodes.isna().sum().sum() + arcs.isna().sum().sum()),
        "timestamps_sorted": bool(t["timestamp"].is_monotonic_increasing),
        "duplicate_timestamps": int(t["timestamp"].duplicated().sum()),
        "duplicate_points": int(t.duplicated(["lon", "lat"]).sum()),
        "median_dt_s": float(dt.median()),
        "max_gap_s": float(dt.max()),
        "n_gaps_gt_10s": int((dt > 10).sum()),
        "lon_min": t["lon"].min(), "lon_max": t["lon"].max(),
        "lat_min": t["lat"].min(), "lat_max": t["lat"].max(),
        "lonlat_in_range": bool(t["lon"].between(-180, 180).all() and t["lat"].between(-90, 90).all()),
        "arc_nodes_in_range": bool(arcs.isin(range(n_nodes)).all().all()) if n_nodes else False,
        "route_ids_in_range": bool(route_ok.all()),
        "route_breaks": breaks,
        "route_repeated_arcs": int(len(route) - len(np.unique(route))),
    }


# ---------- Step 3: UTM projection ----------
def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat): 326xx north, 327xx south."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_trip(trip):
    """Add x, y (metres) to track and nodes using the trip's own UTM zone."""
    t = trip["track"]
    epsg = utm_epsg(t["lon"].mean(), t["lat"].mean())
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    for key in ["track", "nodes"]:
        df = trip[key]
        df["x"], df["y"] = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    trip["epsg"] = epsg
    return trip


def arc_lines(trip):
    """One shapely LineString per arc (index = arc_id), in UTM metres."""
    xy = trip["nodes"][["x", "y"]].to_numpy()
    a = trip["arcs"]
    coords = np.stack([xy[a["from_node"]], xy[a["to_node"]]], axis=1)  # (n_arcs, 2, 2)
    return shapely.linestrings(coords)


# ---------- Step 4: per-point labelling ----------
def label_points(trip, lines, window=5, max_dist=50.0):
    """Forward-only pointer: for each point look at route[ptr : ptr+window+1], take the nearest arc,
    move the pointer there (never backwards). Returns label arc, label distance, uncertain flag."""
    pts = shapely.points(trip["track"][["x", "y"]].to_numpy())
    route = trip["route"]
    ptr = 0
    labels, dists = [], []
    for p in pts:
        idx = np.arange(ptr, min(ptr + window + 1, len(route)))
        d = shapely.distance(p, lines[route[idx]])
        best = int(np.argmin(d))
        ptr = idx[best]
        labels.append(route[ptr])
        dists.append(d[best])
    t = trip["track"]
    t["label"] = labels
    t["label_dist"] = dists
    t["label_uncertain"] = t["label_dist"] > max_dist
    return trip


# ---------- Step 5: candidate generation ----------
def candidates(trip, lines, partition, radius):
    """All arcs within `radius` metres of each GPS point (shapely STRtree), one row per (point, arc)."""
    t = trip["track"]
    pts = shapely.points(t[["x", "y"]].to_numpy())
    # keep only arcs inside the track's bounding box + 150 m (networks can have ~300k arcs, most far away)
    x0, y0, x1, y1 = t["x"].min(), t["y"].min(), t["x"].max(), t["y"].max()
    pad = max(150.0, radius)
    keep = np.flatnonzero(shapely.intersects(lines, shapely.box(x0 - pad, y0 - pad, x1 + pad, y1 + pad)))
    tree = shapely.STRtree(lines[keep])
    pi, ki = tree.query(pts, predicate="dwithin", distance=radius)
    ai = keep[ki]  # map back to the original arc_id
    c = pd.DataFrame({
        "trip_id": trip["trip_id"],
        "partition": partition,
        "point_idx": pi,
        "timestamp": t["timestamp"].to_numpy()[pi],
        "x": t["x"].to_numpy()[pi],
        "y": t["y"].to_numpy()[pi],
        "arc_id": ai,
        "dist_to_arc": shapely.distance(pts[pi], lines[ai]),
    })
    c = c.sort_values(["point_idx", "dist_to_arc"], ignore_index=True)
    c["dist_rank"] = c.groupby("point_idx").cumcount() + 1  # 1 = nearest arc
    c["label"] = (c["arc_id"].to_numpy() == t["label"].to_numpy()[c["point_idx"]]).astype(int)
    c["label_uncertain"] = t["label_uncertain"].to_numpy()[c["point_idx"]]
    return c


def sweep_stats(trip, cands, radii):
    """Per radius: points, misses (true arc not in candidates), total candidates."""
    rows = []
    n = len(trip["track"])
    certain = ~trip["track"]["label_uncertain"].to_numpy()
    for r in radii:
        sub = cands[cands["dist_to_arc"] <= r]
        hit = np.zeros(n, bool)
        hit[sub.loc[sub["label"] == 1, "point_idx"]] = True
        rows.append({"radius_m": r, "n_points": n, "misses": int((~hit).sum()), "n_candidates": len(sub),
                     "n_certain": int(certain.sum()), "misses_certain": int((~hit & certain).sum())})
    return rows


def process_trip(trip_dir, partition, radii=RADII):
    """Full per-trip pipeline. Candidates are built at the largest radius so any smaller R
    can be selected later by filtering dist_to_arc. Returns (quality_row, candidates, sweep_rows, track)."""
    trip = load_trip(trip_dir)
    q = validate_trip(trip, partition)
    trip = project_trip(trip)
    lines = arc_lines(trip)
    trip = label_points(trip, lines)
    q["epsg"] = trip["epsg"]
    q["label_dist_median_m"] = float(trip["track"]["label_dist"].median())
    q["label_uncertain_frac"] = float(trip["track"]["label_uncertain"].mean())
    c = candidates(trip, lines, partition, max(radii))
    return q, c, sweep_stats(trip, c, radii), trip["track"]


def summarise_sweep(rows):
    """Aggregate per-trip sweep rows into one table: miss rate and mean candidates per point."""
    df = pd.DataFrame(rows).groupby("radius_m").sum(numeric_only=True).reset_index()
    df["miss_rate"] = df["misses"] / df["n_points"]
    df["miss_rate_certain"] = df["misses_certain"] / df["n_certain"]
    df["avg_candidates_per_point"] = df["n_candidates"] / df["n_points"]
    return df


def pick_radius(sweep, tol=0.01):
    """Smallest radius whose miss rate is within `tol` (1 percentage point) of the largest radius."""
    best = sweep["miss_rate"].min()
    return int(sweep.loc[sweep["miss_rate"] <= best + tol, "radius_m"].min())


# ---------- Step 8: Logistic Regression baseline ----------
FEATURES = ["dist_to_arc", "dist_rank"]


def top1_accuracy(df, proba):
    """Per GPS point take the candidate with the highest probability; correct if it is the label arc.
    Points whose true arc is not among the candidates count as wrong."""
    best = df.assign(p=proba).sort_values("p", ascending=False).drop_duplicates(["trip_id", "point_idx"])
    return best["label"].mean()


def logreg_baseline(train, test, seed=42):
    """LR on (dist_to_arc, dist_rank). Scaler is fitted on train only (inside the pipeline, so also
    per CV fold). Uncertain-label points are left out of training. Returns one row of metrics."""
    import time
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, log_loss
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = train[~train["label_uncertain"]]
    X, y, groups = tr[FEATURES], tr["label"], tr["trip_id"]
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed))
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_f1 = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro")

    t0 = time.time(); model.fit(X, y); train_s = time.time() - t0
    t0 = time.time(); proba = model.predict_proba(test[FEATURES])[:, 1]; infer_s = time.time() - t0
    return {
        "cv_macro_f1_mean": cv_f1.mean(), "cv_macro_f1_std": cv_f1.std(),
        "test_top1_accuracy": top1_accuracy(test, proba),
        "test_top1_accuracy_certain_points": top1_accuracy(test[~test["label_uncertain"]], proba[~test["label_uncertain"].to_numpy()]),
        "test_macro_f1": f1_score(test["label"], proba >= 0.5, average="macro"),
        "test_log_loss": log_loss(test["label"], proba),
        "nearest_arc_top1_accuracy": test.loc[test["dist_rank"] == 1, "label"].sum() / test.groupby(["trip_id", "point_idx"]).ngroups,
        "train_time_s": train_s, "inference_time_s": infer_s,
        "n_train_rows": len(tr), "n_test_rows": len(test),
        "coef_dist_to_arc": model[-1].coef_[0][0], "coef_dist_rank": model[-1].coef_[0][1],
    }
