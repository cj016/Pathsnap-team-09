"""Exercise each published trip through the API and the evaluator form."""

import argparse
import csv
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api", help="URL of the API on laptop 1")
    parser.add_argument("--ui", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path("logs/student_demo.csv"))
    args = parser.parse_args()
    api = args.api.rstrip("/")
    health = requests.get(f"{api}/health", timeout=10)
    health.raise_for_status()
    if health.json()["status"] != "ready":
        parser.error(f"API not ready: {health.json()}")
    listing = requests.get(f"{api}/trips", timeout=10)
    listing.raise_for_status()
    trips = listing.json()["trip_ids"]
    if len(trips) != 10 or len(set(trips)) != 10:
        parser.error(f"Expected 10 distinct trips, found {trips}")
    home = requests.get(args.ui, timeout=20)
    home.raise_for_status()
    records = []
    for trip in trips:
        if f'value="{trip}"' not in home.text:
            raise RuntimeError(f"Missing trip in selector: {trip}")
        response = requests.get(f"{api}/predict/{trip}", timeout=(5, 300))
        response.raise_for_status()
        result = response.json()
        points = result["observation_count"]
        if (
            result["trip_id"] != trip
            or not points
            or len(result["observations"]) != points
        ):
            raise RuntimeError(f"Incomplete prediction for {trip}")
        page = requests.post(args.ui, data={"trip": trip}, timeout=(10, 330))
        page.raise_for_status()
        if (
            f"Predicted route for trip {trip}" not in page.text
            or 'role="alert"' in page.text
        ):
            raise RuntimeError(f"Map was not returned for {trip}")
        m = result["metrics"]
        clf, hmm = m.get("classifier", {}), m.get("hmm", {})
        continuity = result["route_continuity"]
        records.append(
            dict(
                trip_id=trip,
                points=points,
                candidates=result["candidate_count"],
                classifier_correct=clf.get("correct_points"),
                hmm_correct=hmm.get("correct_points"),
                classifier_top1=clf.get("top1_accuracy"),
                hmm_top1=hmm.get("top1_accuracy"),
                classifier_continuity=continuity["classifier"]["rate"],
                hmm_continuity=continuity["hmm"]["rate"],
                hmm_helped=m.get("hmm_helped"),
                hmm_hurt=m.get("hmm_hurt"),
                hmm_correct_pair_fraction=hmm.get("correct_transition_pair_fraction"),
                classifier_correct_pair_fraction=clf.get(
                    "correct_transition_pair_fraction"
                ),
                seconds=result["timing_seconds"]["total"],
            )
        )
        print(
            f"{trip}: {points} points, map returned, scores {m['status']}", flush=True
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
