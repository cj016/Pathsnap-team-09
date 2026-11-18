# PathSnap evaluation 1: LightGBM and MLP

Run the commands from the repository root with Python 3.12. The preprocessing code and shared feature framework are both in main at commit 6ff751a.

```powershell
git fetch origin
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r model_experiments/requirements.txt
$Data = Read-Host 'Path to the extracted folder containing train and student_test'
& ./.venv/Scripts/python.exe -m model_experiments.checkpoint prepare "$Data" processed/eval1-seed42 --radius 75
& ./.venv/Scripts/python.exe -m model_experiments.checkpoint train processed/eval1-seed42 model_experiments/results/eval1-seed42 --folds 5 --seed 42
```

The point inventory and candidate files in processed/ are local caches. Use fresh cache and output directories for a different experiment. To rehearse with five training trips, add `--train-limit 5` to prepare, and use separate cache and result directories in both commands.

## Models and evaluation

A row represents one GPS observation and one possible road arc. The six inputs are distance to the arc (metres), heading difference (degrees), arc length (metres), distance rank, speed (m/s), and fractional position along the arc. The shared code derives motion from timestamps in seconds. Reference labels and IDs are excluded from the feature matrix.

LightGBM learns interactions between features with 80 trees, 15 leaves per tree and a learning rate of 0.08. The MLP uses two ReLU hidden layers (24 and 12 units), Adam, fold-fitted median imputation and standardisation. Both models give a positive-class probability for each candidate. The MLP monitors training loss and has a 35-epoch limit.

Keep whole trips in the same fold. The seed defaults to 42 and is passed to the team's shuffled GroupKFold. Evaluate the nearest-road baseline and each fitted model on the same GPS observations. Report candidate-level macro F1 and log loss and point-level Top-1. Training excludes uncertain points and points without a positive candidate. All-points Top-1 counts every reference miss as incorrect, including points with zero candidate rows. Results are also recorded on the certain-only subset. Labels are derived from the team's route alignment.

## What to show

- `comparison.csv`: five-fold mean macro F1, student-test Top-1, macro F1, log loss, and timing.
- `example_point.csv`: real held-out candidates with distances, headings, labels and predicted probabilities.
- `mlp_training_loss.png` and `mlp_loss.csv`: the actual training loss history.
- `metrics.csv` and `run.json`: split-specific results, trip assignments, model settings and software versions.

AI assistance: Codex helped implement these experiment scripts and check compatibility with the team modules. Review the code, run it locally and use the outputs of your own run when presenting.
