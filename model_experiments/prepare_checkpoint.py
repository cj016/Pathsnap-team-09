"""Join the team's road and GPS features and cache one pair of tables per trip."""
import argparse
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
    return features[KEYS + ["arc_id", "label", "label_uncertain"] + FEATURE_COLUMNS], points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Folder containing train and student_test")
    parser.add_argument("cache", type=Path, help="Use a fresh folder for each radius or input revision")
    parser.add_argument("--radius", type=float, default=75)
    parser.add_argument("--train-limit", type=int)
    args = parser.parse_args()
    for partition in ("train", "student_test"):
        target = args.cache / partition
        target.mkdir(parents=True, exist_ok=True)
        trips = sorted(d for d in (args.data / partition).iterdir() if d.is_dir())
        if partition == "train" and args.train_limit is not None:
            trips = trips[:args.train_limit]
        for trip in trips:
            rows, points = build_trip(trip, partition, args.radius)
            rows.to_parquet(target / f"{trip.name}.candidates.parquet", index=False)
            points.to_parquet(target / f"{trip.name}.points.parquet", index=False)
            print(f"{partition}/{trip.name}: {len(points)} points, {len(rows)} candidates", flush=True)


if __name__ == "__main__":
    main()
