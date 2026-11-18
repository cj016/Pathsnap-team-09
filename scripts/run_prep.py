"""Run the PathSnap prep pipeline.

  python scripts/run_prep.py --limit 5   -> 5 train trips  -> processed/candidates_sample.parquet
  python scripts/run_prep.py             -> all trips      -> processed/candidates_{train,test}.parquet
  python scripts/run_prep.py --logreg    -> Logistic Regression baseline on saved candidates -> results/logreg_results.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import PARTITIONS, RADII, list_trips, logreg_baseline, pick_radius, plot_confusion_matrix, process_trip, summarise_sweep  # noqa: E402

PROCESSED, RESULTS = ROOT / "processed", ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N train trips (sample run)")
    ap.add_argument("--logreg", action="store_true", help="only run the Logistic Regression baseline")
    args = ap.parse_args()
    if args.logreg:
        res = logreg_baseline(pd.read_parquet(PROCESSED / "candidates_train.parquet"),
                              pd.read_parquet(PROCESSED / "candidates_test.parquet"))
        pd.DataFrame([res]).to_csv(RESULTS / "logreg_results.csv", index=False)
        plot_confusion_matrix(res, ROOT / "reports" / "figures" / "logreg_confusion_matrix.png")
        print(pd.Series(res).to_string())
        return
    PROCESSED.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    parts = ["train"] if args.limit else PARTITIONS
    quality, cands, sweep, tracks = [], {p: [] for p in parts}, {p: [] for p in parts}, []
    t0 = time.time()
    part_time = {}
    for part in parts:
        tp = time.time()
        trips = list_trips(part)[: args.limit] if args.limit else list_trips(part)
        res = Parallel(n_jobs=-1)(delayed(process_trip)(d, part) for d in trips)  # trips in parallel
        for d, (q, c, s, tr) in zip(trips, res):
            quality.append(q)
            cands[part].append(c)
            sweep[part].extend(s)
            tracks.append(tr.assign(trip_id=d.name, partition=part))
        part_time[part] = time.time() - tp
        print(f"[{part}] {len(trips)} trips processed in {part_time[part]:.0f}s")

    # radius sweep on TRAIN only, then pick R
    sw = summarise_sweep(sweep["train"])
    R = pick_radius(sw)
    sw["chosen"] = sw["radius_m"] == R
    suffix = "_sample" if args.limit else ""
    sw.to_csv(RESULTS / f"radius_sweep{suffix}.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS / f"data_quality{suffix}.csv", index=False)
    pd.concat(tracks).to_parquet(PROCESSED / f"points_labelled{suffix}.parquet", index=False)
    print("\nRadius sweep (train):\n", sw.to_string(index=False))
    print(f"\nChosen R = {R} m: smallest radius whose miss rate is within 1 point of the R={max(RADII)} m miss rate, "
          f"so a bigger radius only adds candidates without finding many more true arcs.")

    out = {"train": "candidates_train", "student_test": "candidates_test"}
    for part in parts:
        c = pd.concat(cands[part], ignore_index=True)
        c = c[c["dist_to_arc"] <= R].reset_index(drop=True)
        name = "candidates_sample" if args.limit else out[part]
        c.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"saved processed/{name}.parquet: {len(c):,} rows")
    for part, sec in part_time.items():
        print(f"time {part}: {sec:.0f}s")
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
