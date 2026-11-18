# PathSnap evaluation: five candidate classifiers

Run from the repository root with Python 3.12. The workflow uses the shared six-feature framework, candidate labels, and trip-grouped splits. It evaluates Logistic Regression, Random Forest, KNN, LightGBM and MLP, alongside a nearest-candidate baseline.

```powershell
& ./.venv/Scripts/python.exe -m pip install -r model_experiments/requirements.txt
$Data = Read-Host 'Path to the extracted folder containing train and student_test'
& ./.venv/Scripts/python.exe -m model_experiments.checkpoint prepare "$Data" processed/five-model-seed42 --radius 75
& ./.venv/Scripts/python.exe -m model_experiments.checkpoint train processed/five-model-seed42 model_experiments/results/five-model-seed42 --folds 5 --seed 42
```

Use a fresh cache and output folder when changing the radius, source data, feature code or seed. For a smoke test, add `--train-limit 5` to the prepare command and use separate smoke-test folders. This limits training trips only; all student-test trips are still prepared.

## Models

All models receive the same six approved inputs: distance to arc, heading difference, arc length, distance rank, speed, and position along arc. Uncertain labels and points without a positive candidate are excluded from model fitting. Imputation and scaling are fitted inside each training fold where required.

- Logistic Regression: median imputation, standardisation, balanced class weights.
- Random Forest: 150 trees, depth capped at 20, minimum leaf size 5.
- KNN: 15 neighbours, distance weighting, median imputation and standardisation. To keep memory and prediction time practical on this multi-million-row dataset, KNN uses a deterministic class-stratified training sample capped at 100,000 rows. Validation and student-test rows remain complete. The sample cap is recorded in `run.json` and fit-row counts in `metrics.csv`.
- LightGBM: 80 trees, 15 leaves, learning rate 0.08.
- MLP: two hidden layers (24, 12), Adam, median imputation, standardisation, up to 35 epochs.

## Evaluation and outputs

All five models use the same trip-grouped cross-validation folds and student-test candidate rows. The runner records candidate-level macro F1 and log loss, point-level Top-1 accuracy (including points with no candidate rows), fit time and inference time. It also records certain-only Top-1 results. The nearest-candidate baseline is included for context, not counted as one of the five learned models.

- `comparison.csv`: one summary row per baseline/model.
- `metrics.csv`: per-fold and student-test metrics, including fit and inference time.
- `run.json`: seed, feature list, trip assignments, package versions, model parameters and KNN sample cap.
- `mlp_loss.csv` and `mlp_training_loss.png`: MLP training loss.
- `example_point.csv`: a held-out example comparing LightGBM and MLP probabilities, when a suitable point exists.

Published results span multiple runs. The RF/KNN quick run used two folds. Use a fresh output directory for a controlled five-model comparison. The live API check and its model are documented in the [project README](../README.md).
