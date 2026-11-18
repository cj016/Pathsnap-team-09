"""Create a comparison table from the recorded fold metrics."""
import argparse
from pathlib import Path
import pandas as pd

MODEL_ORDER = ["Nearest", "Logistic Regression", "Random Forest", "KNN", "LightGBM", "MLP"]


def write_comparison(output):
    metrics = pd.read_csv(output / "metrics.csv")
    all_points = metrics.loc[metrics.scope.eq("all")]
    cv_f1 = all_points.loc[all_points.fold.str.startswith("cv_")].groupby("model").macro_f1.mean()
    test = all_points.loc[all_points.fold.eq("student_test")].set_index("model")
    certain = metrics.loc[metrics.fold.eq("student_test") & metrics.scope.eq("certain")].set_index("model")
    if test.empty:
        raise ValueError("The student-test run must finish before making the comparison.")
    summary = pd.DataFrame({
        "cv_mean_macro_f1": cv_f1,
        "student_top1_all": test.top1,
        "student_top1_certain": certain.top1,
        "student_macro_f1_all": test.macro_f1,
        "student_log_loss_all": test.log_loss,
        "student_correct_points": test.correct,
        "student_total_points": test.gps_points,
        "fit_seconds": test.fit_seconds,
        "inference_seconds": test.inference_seconds,
    }).reindex([name for name in MODEL_ORDER if name in test.index])
    summary.to_csv(output / "comparison.csv", index_label="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_comparison(parser.parse_args().output)
