"""Join the team's road and GPS features and cache one pair of tables per trip."""

import argparse
import json
import math
from pathlib import Path

from features.data_preparation import point_inventory
from features.feature_engineering import FEATURE_COLUMNS, build_trip_features
from src.data_prep import arc_lines, candidates, label_points, load_trip, project_trip

KEYS = ["trip_id", "point_idx"]


def build_trip(directory, partition, radius):
    trip = project_trip(load_trip(directory))
    lines = arc_lines(trip)
    label_points(trip, lines)
    track = trip["track"]
    rows = candidates(trip, lines, partition, radius)
    points = point_inventory(track.assign(trip_id=directory.name))
    features = build_trip_features(rows, trip, lines)
    return features[
        KEYS + ["arc_id", "label", "label_uncertain"] + FEATURE_COLUMNS
    ], points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", type=Path, help="Folder containing train and student_test"
    )
    parser.add_argument(
        "cache", type=Path, help="Use a fresh folder for each radius or input revision"
    )
    parser.add_argument("--radius", type=float, default=75)
    parser.add_argument("--train-limit", type=int)
    args = parser.parse_args()
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("Radius must be positive and finite")
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("Train limit must be at least one")
    metadata = args.cache / "cache_config.json"
    settings = {"candidate_radius_metres": args.radius, "train_limit": args.train_limit}
    if metadata.exists() and json.loads(metadata.read_text()) != settings:
        parser.error("Use a fresh cache when changing the radius or trip limit")
    for partition in ("train", "student_test"):
        target = args.cache / partition
        target.mkdir(parents=True, exist_ok=True)
        trips = sorted(
            d
            for d in (args.data / partition).iterdir()
            if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
        )
        if not trips:
            parser.error(f"No trip folders in {args.data / partition}")
        if partition == "train" and args.train_limit is not None:
            trips = trips[: args.train_limit]
        old_ids = {p.name.split(".")[0] for p in target.glob("*.parquet")}
        if old_ids - {trip.name for trip in trips}:
            parser.error(f"Use a fresh cache. Extra trips found in {target}")
        for trip in trips:
            rows, points = build_trip(trip, partition, args.radius)
            rows.to_parquet(target / f"{trip.name}.candidates.parquet", index=False)
            points.to_parquet(target / f"{trip.name}.points.parquet", index=False)
            print(
                f"{partition}/{trip.name}: {len(points)} points, {len(rows)} candidates",
                flush=True,
            )
    metadata.write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
