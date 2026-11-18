"""Export student-test point labels for scoring on the API laptop."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted((args.cache / "student_test").glob("*.points.parquet"))
    if not paths:
        parser.error("Missing student_test point inventories in the prepared cache")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        points = pd.read_parquet(path, columns=["point_idx", "reference_arc_id"])
        points = points.rename(columns={"reference_arc_id": "arc_id"})
        points.sort_values("point_idx").to_csv(
            args.output / f"{path.name.split('.')[0]}.csv", index=False
        )
    print(f"Exported student-test references for {len(paths)} trips")


if __name__ == "__main__":
    main()
