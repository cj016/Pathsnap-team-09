"""Logistic Regression (and optionally Gaussian NB) through the team's evaluation harness.

Uses Person 2's / model_experiments' code unchanged: same per-trip cache, same shuffled
GroupKFold splits (seed 42), same training-row filter and same metric function
(collect_metrics -> scopes "all" and "certain"), so results are comparable with LightGBM/MLP.

  python scripts/run_logreg_team.py --features basic   # dist_to_arc, dist_rank
  python scripts/run_logreg_team.py --features full    # Person 2's six features (+ --gnb)
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import score_candidate_classifier  # noqa: E402
from experiments.cross_validation import assert_no_trip_leakage, make_group_splits  # noqa: E402
from features.data_preparation import prepare_xy  # noqa: E402
from features.feature_engineering import FEATURE_COLUMNS  # noqa: E402
from model_experiments.run_checkpoint import collect_metrics, read_partition  # noqa: E402

FEATURE_SETS = {"basic": ["dist_to_arc", "dist_rank"], "full": FEATURE_COLUMNS}


def make_logreg(seed=42):
    # imputer (median, fitted on the training fold) only matters for NaN heading/speed in the full set
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(random_state=seed, max_iter=1000))])


def make_gnb(seed=42):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", GaussianNB())])


def summarise(metrics):
    """Same columns as model_experiments' comparison.csv."""
    m = pd.DataFrame(metrics)
    cv = m[m.fold.str.startswith("cv_")].groupby(["model", "scope"]).macro_f1.agg(["mean", "std"])
    test = m[m.fold.eq("student_test")].set_index(["model", "scope"])
    out = pd.DataFrame({"cv_mean_macro_f1": cv["mean"], "cv_std_macro_f1": cv["std"],
                        "student_top1": test.top1, "student_macro_f1": test.macro_f1,
                        "student_log_loss": test.log_loss, "student_correct_points": test.correct,
                        "student_total_points": test.gps_points, "fit_seconds": test.fit_seconds,
                        "inference_seconds": test.inference_seconds})
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, default="basic")
    ap.add_argument("--gnb", action="store_true", help="also run Gaussian Naive Bayes")
    ap.add_argument("--cache", type=Path, default=ROOT / "processed" / "eval1-seed42")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    feats = FEATURE_SETS[args.features]
    models = {"LogReg": make_logreg} | ({"GaussianNB": make_gnb} if args.gnb else {})

    train, train_points = read_partition(args.cache, "train")
    test, test_points = read_partition(args.cache, "student_test")
    splits = make_group_splits(train, n_splits=5, random_state=args.seed)
    assert_no_trip_leakage(train, splits)
    cases = [(f"cv_{i}", tr, va) for i, (tr, va) in enumerate(splits, 1)] + [("student_test", None, None)]

    metrics = []
    for fold, tr, va in cases:
        final = tr is None
        fit_rows = train if final else train.iloc[tr]
        val = test if final else train.iloc[va]
        points = test_points if final else train_points[train_points.trip_id.isin(val.trip_id.unique())]
        X, y = prepare_xy(fit_rows, feats)
        for name, factory in models.items():
            model = factory(args.seed)
            t0 = perf_counter(); model.fit(X, y); fit_s = perf_counter() - t0
            measured, scored = score_candidate_classifier(model, val, feats)
            metrics.extend(collect_metrics(name, fold, scored, points, measured["inference_seconds"], fit_s))
            print(f"{fold} {name}: fit {fit_s:.1f}s on {len(X):,} rows", flush=True)

    tag = "" if args.features == "full" else "_basic"
    pd.DataFrame(metrics).to_csv(ROOT / "results" / f"logreg_team_metrics{tag}.csv", index=False)
    summary = summarise(metrics).assign(features=",".join(feats))
    out = ROOT / "results" / ("logreg_full_features.csv" if args.features == "full" else "logreg_team_protocol_basic.csv")
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
