"""Train and compare two candidate classifiers using the shared trip-grouped harness."""
import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

from evaluation.baseline import nearest_candidate_top1
from evaluation.metrics import score_candidate_classifier, top1_accuracy
from experiments.cross_validation import assert_no_trip_leakage, make_group_splits
from features.data_preparation import prepare_xy
from features.feature_engineering import FEATURE_COLUMNS
from .comparison import write_comparison
from .models import make_boosting, make_mlp

KEYS = ["trip_id", "point_idx"]
MODELS = {"LightGBM": make_boosting, "MLP": make_mlp}


def read_partition(cache, partition):
    folder = cache / partition
    files = sorted(folder.glob("*.points.parquet"))
    if not files:
        raise ValueError(f"No point inventories in {folder}")
    def read_tables(kind):
        return pd.concat([pd.read_parquet(f.with_suffix("").with_suffix(f".{kind}.parquet"))
                          for f in files], ignore_index=True)
    rows, points = read_tables("candidates"), read_tables("points")
    if points.duplicated(KEYS).any() or rows.duplicated(KEYS + ["arc_id"]).any():
        raise ValueError("Duplicate GPS point or candidate ID")
    return rows, points


def collect_metrics(name, fold, scored, points, seconds, fit_seconds):
    # The shared top1 helper counts empty candidate sets as misses. The challenge
    # also counts nonempty sets with no reference arc, so calculate coverage here.
    hits = scored.loc[scored.label.eq(1), KEYS].drop_duplicates()
    marked = points.merge(hits.assign(hit=True), on=KEYS, how="left", validate="one_to_one")
    for scope in ["all", "certain"]:
        inventory = marked if scope == "all" else marked.loc[~marked.label_uncertain]
        rows = scored if scope == "all" else scored.loc[~scored.label_uncertain]
        top = top1_accuracy(scored, inventory, include_uncertain=True)
        result = dict(model=name, fold=fold, scope=scope, gps_points=len(inventory),
                      candidate_rows=len(rows), correct=top["correct_points"],
                      top1=top["top1_accuracy"], candidate_miss_rate=inventory.hit.isna().mean(),
                      macro_f1=None, log_loss=None, fit_seconds=fit_seconds, inference_seconds=seconds)
        if name != "Nearest" and len(rows):
            result.update(macro_f1=f1_score(rows.label, rows.predicted_label, labels=[0, 1],
                                            average="macro", zero_division=0),
                          log_loss=log_loss(rows.label, rows.probability_label_1, labels=[0, 1]))
        yield result


def save_example(scored_by_model, output):
    rows = scored_by_model["LightGBM"].copy()
    rows["boosting_probability"] = rows["probability_label_1"]
    mlp = scored_by_model["MLP"][KEYS + ["arc_id", "probability_label_1"]].rename(
        columns={"probability_label_1": "mlp_probability"})
    rows = rows.merge(mlp, on=KEYS + ["arc_id"], validate="one_to_one")
    # Select on geometry and label availability only, independent of model outcome.
    near = rows.loc[(rows.dist_rank <= 2) & ~rows.label_uncertain]
    examples = near.groupby(KEYS).agg(n=("arc_id", "size"), gap=("dist_to_arc", lambda v: v.max()-v.min()),
                                      heading_span=("heading_difference", lambda v: v.max()-v.min()),
                                      positives=("label", "sum"))
    possible = examples.query("n == 2 and gap < 1 and heading_span > 90 and positives == 1")
    if possible.empty:
        possible = examples.query("positives == 1")
    if possible.empty:
        return
    trip, point = possible.index[0]
    cols = KEYS + ["arc_id"] + FEATURE_COLUMNS + ["label", "boosting_probability", "mlp_probability"]
    rows.loc[rows.trip_id.eq(trip) & rows.point_idx.eq(point), cols].sort_values("dist_rank").to_csv(
        output / "example_point.csv", index=False)


def run(cache, output, seed, folds):
    train, train_points = read_partition(cache, "train")
    test, test_points = read_partition(cache, "student_test")
    if set(train_points.trip_id) & set(test_points.trip_id):
        raise ValueError("Training and student-test trips overlap")
    # Use shared candidate-row split assignments for direct comparison with teammates.
    if set(train.trip_id) != set(train_points.trip_id):
        raise ValueError("An entire training trip lacks candidates; review fold assignment")
    splits = make_group_splits(train, n_splits=folds, random_state=seed)
    assert_no_trip_leakage(train, splits)
    cases = [(f"cv_{i}", tr, va) for i, (tr, va) in enumerate(splits, 1)]
    cases.append(("student_test", None, None))
    metrics, histories, split_log = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for fold, tr, va in cases:
        final = tr is None
        fit_rows = train if final else train.iloc[tr]
        val = test if final else train.iloc[va]
        points = test_points if final else train_points.loc[train_points.trip_id.isin(val.trip_id.unique())]
        X, y = prepare_xy(fit_rows, FEATURE_COLUMNS)
        split_log.append(dict(fold=fold, training_trips=sorted(fit_rows.trip_id.unique()),
                              scored_trips=sorted(points.trip_id.unique()), fit_rows=len(X)))
        nearest = val.assign(probability_label_1=-val.dist_rank)
        baseline = list(collect_metrics("Nearest", fold, nearest, points, None, None))
        assert baseline[0]["correct"] == nearest_candidate_top1(val, points, include_uncertain=True)["correct_points"]
        metrics.extend(baseline)
        saved = {}
        for name, factory in MODELS.items():
            model = factory(seed)
            start = perf_counter()
            model.fit(X, y)
            fit_seconds = perf_counter() - start
            measured, scored = score_candidate_classifier(model, val, FEATURE_COLUMNS)
            metrics.extend(collect_metrics(name, fold, scored, points, measured["inference_seconds"], fit_seconds))
            if name == "MLP":
                histories.extend(dict(fold=fold, epoch=i, loss=v)
                                 for i, v in enumerate(model[-1].loss_curve_, 1))
            if final:
                saved[name] = scored
            print(f"{fold} {name}: {fit_seconds:.1f}s, {len(X)} training rows", flush=True)
        if final:
            save_example(saved, output)
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(histories).to_csv(output / "mlp_loss.csv", index=False)
    config = dict(seed=seed, features=FEATURE_COLUMNS, splits=split_log, python=platform.python_version(),
                  versions={p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm",
                                                     "shapely", "pyproj", "pyarrow", "scipy"]},
                  models={n: f(seed)[-1].get_params() for n, f in MODELS.items()})
    (output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    history = pd.DataFrame(histories).query("fold == 'student_test'")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history.epoch, history.loss, color="#165c75", linewidth=2)
    ax.set(xlabel="Training epoch", ylabel="Training loss", title="MLP loss on the final training fit")
    fig.tight_layout()
    fig.savefig(output / "mlp_training_loss.png", dpi=150)
    plt.close(fig)
    write_comparison(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.cache, args.output, args.seed, args.folds)
