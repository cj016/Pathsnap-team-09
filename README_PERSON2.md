# Person 2 — feature engineering and evaluation

This branch is based on Person 1's `data-preprocessing` branch. It uses the same
trip objects and candidate table produced by `src.data_prep`.

## Agreed model feature order

All model implementations must use this exact feature order:

1. `dist_to_arc`
2. `heading_difference`
3. `arc_length`
4. `dist_rank`
5. `speed`
6. `position_along_arc`

Use `features.feature_engineering.get_feature_matrix(table)` to select this
ordered matrix.

## Building the feature batch from one trip

Person 1's `src.data_prep` exposes `load_trip`, `project_trip`, `arc_lines`
and `candidates`. Raw `Data/` files are expected locally and are gitignored.

```python
from src.data_prep import DATA, load_trip, project_trip, arc_lines, candidates
from features.feature_engineering import build_trip_features, FEATURE_COLUMNS, get_feature_matrix

trip_id = "00000001"
trip = project_trip(load_trip(DATA / "train" / trip_id))
lines = arc_lines(trip)
candidate_rows = candidates(trip, lines, "train", radius=75)

features = build_trip_features(candidate_rows, trip, lines)
X = get_feature_matrix(features)
```

The exact feature definitions:
- `dist_to_arc`: Person 1's Shapely distance from GPS point to candidate line.
- `heading_difference`: smallest angular difference between consecutive-point GPS
  movement bearing and directed arc bearing, degrees in [0, 180].
- `arc_length`: directed arc endpoint segment length in metres.
- `dist_rank`: Person 1's nearest-first candidate rank.
- `speed`: GPS displacement divided by elapsed seconds, metres/second.
  Stationary samples carry the previous valid movement heading; zero/invalid
  time intervals and gaps above 30 seconds do not yield a speed.
- `position_along_arc`: clamped scalar projection onto the directed arc, from
  0 (start node) to 1 (end node).

For this dataset's `src.data_prep.arc_lines`, each road arc is a straight
node-to-node directed segment, so endpoint length and line bearing are
consistent with the candidate geometry.

## Training rows and candidate misses

`features.data_preparation.training_rows` removes uncertain points and removes
all candidate rows for GPS points whose correct arc is absent from the candidate
set (no positive row). Candidate misses are not usable as candidate-classifier
training examples.

For evaluation, create the full point inventory from
`processed/points_labelled.parquet` using
`features.data_preparation.point_inventory`. This includes GPS points with
zero candidates. The candidate Parquet alone cannot provide the correct
denominator for Top-1 accuracy or candidate miss rate.

## sklearn Pipeline integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features.feature_engineering import FEATURE_COLUMNS, get_feature_matrix
from features.data_preparation import prepare_xy

train_rows = ... # features from training trips
X_train, y_train = prepare_xy(train_rows, FEATURE_COLUMNS)
model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42)),
])
model.fit(X_train, y_train)

X_eval = get_feature_matrix(eval_feature_table)
probabilities = model.predict_proba(X_eval)
```

Scaling/imputation must be fitted on training data only and inside the
cross-validation pipeline. The feature table can contain NaNs for initial GPS
heading or invalid/gapped speed; use a train-fitted imputer in the sklearn
pipeline (for example `SimpleImputer(strategy="median")`) before scaling.

## Metrics and baseline

- `evaluation.metrics.score_candidate_classifier`: candidate-row accuracy,
  macro F1, log loss, inference duration and candidate probabilities.
- `evaluation.metrics.top1_accuracy`: highest-probability arc per point;
  counts candidate misses as wrong using the full point inventory.
- `evaluation.baseline.nearest_candidate_top1`: always choose the
  `dist_rank=1` candidate; same denominator and uncertainty filtering.
- Report Top-1 both excluding uncertain points and including all points.

## Trip-grouped CV

```python
from experiments.cross_validation import make_group_splits, assert_no_trip_leakage
splits = make_group_splits(train_rows, n_splits=5)
assert_no_trip_leakage(train_rows, splits)
```

Keep all points from a trip together in a fold. Fit preprocessing pipelines
inside each fold. Do not include `label`, `label_uncertain`, `arc_id`,
`trip_id`, `point_idx` or reference metadata in model features.

## Inputs and reproducibility

Run Person 1's preprocessing pipeline first to create
`processed/points_labelled.parquet` and candidate Parquet files. Raw `Data/`
is not committed to Git. Do not report metrics until the actual code has been
run on the permitted dataset. The sample Logistic Regression row in
`results/logreg_results.csv` is Person 1's preprocessing-branch output, not
a result produced by this Person 2 framework.
