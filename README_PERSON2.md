# Person 2 — Feature engineering and evaluation framework

This branch adds the shared candidate feature interface and reusable evaluation
utilities for the PathSnap map-matching project.

## Agreed model feature order

All model code should use this exact order:

1. `dist_to_arc`
2. `heading_difference`
3. `arc_length`
4. `dist_rank`
5. `speed`
6. `position_along_arc`

Use `features.feature_engineering.get_feature_matrix(table)` to select them.

## Candidate table contract

Each row is one (GPS point, candidate directed arc) pair. Required base
columns are `trip_id, point_idx, timestamp, x, y, arc_id, dist_to_arc,
dist_rank`. The sample supplied by Person 1 also contains `label` and
`label_uncertain`.

The current sample alone does **not** contain enough information to calculate
all six features. The geometry-dependent features require directed arc endpoint
coordinates and bearings joined from `.nodes/.arcs`, while GPS heading/speed
require ordered points from `.track`. The feature code raises explicit
errors rather than inserting made-up values.

For an initial geometry join, add these columns to each candidate row:

- `arc_start_x, arc_start_y, arc_end_x, arc_end_y` (UTM metres)
- `gps_heading, arc_heading` (compass degrees clockwise from north)

The arc length implementation currently uses endpoint chord length. For
polyline arcs, use the sum of segment lengths once full geometry is available.

## Training exclusions

`features.data_preparation.training_rows` removes uncertain points and removes
all candidate rows belonging to a GPS point that has no positive candidate.
Candidate-miss points remain part of evaluation through a separate
point_inventory table.

## Evaluation

`evaluation.metrics.score_candidate_classifier` calculates candidate-level
accuracy, macro F1, log loss and inference time and appends P(label=1).
`top1_accuracy` chooses the highest-probability arc per point. It requires a
one-row-per-GPS-point inventory, including points with zero candidate rows.
Provide `reference_arc_id` in the inventory when possible. Otherwise positive
candidate labels are used to recover the reference arc for points with a
candidate hit.

`evaluation.baseline.nearest_candidate_top1` evaluates the always-pick-
`dist_rank=1` baseline using the same point inventory. A zero-candidate point
is counted wrong.

## sklearn pipeline integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from features.feature_engineering import FEATURE_COLUMNS, get_feature_matrix
from features.data_preparation import prepare_xy

train_rows = ...  # candidate rows from training trips
X_train, y_train = prepare_xy(train_rows, FEATURE_COLUMNS)

mlp = Pipeline([
    ("scale", StandardScaler()),
    ("model", MLPClassifier(random_state=42, max_iter=300)),
])
mlp.fit(X_train, y_train)

# Use the same column order at inference:
X_eval = get_feature_matrix(eval_feature_table)
probabilities = mlp.predict_proba(X_eval)
```

All scaling/normalisation must be fitted on training data only. If using
GroupKFold, fit the entire pipeline separately within each fold.

## Trip-based cross-validation

```python
from experiments.cross_validation import make_group_splits, assert_no_trip_leakage

splits = make_group_splits(train_rows, n_splits=5)
assert_no_trip_leakage(train_rows, splits)
```

GroupKFold splits by `trip_id`, preventing points from one trip appearing in
both train and validation for the same fold.

## Important input still needed

To complete and validate the six-feature batch, integrate the real
`.track`, `.nodes`, and `.arcs` data. The sample rows in the team message
are sufficient to establish the CSV schema and labels, but not the geometry.
No model scores or performance claims are included here because the full data
has not yet been run.
