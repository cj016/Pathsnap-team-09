"""Candidate classifiers for the six columns selected by get_feature_matrix."""
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_boosting(seed=42):
    """Shallow boosted trees for nonlinear interactions between road features."""
    return Pipeline([
        ("model", LGBMClassifier(
            objective="binary", n_estimators=80, learning_rate=0.08,
            num_leaves=15, min_child_samples=30, random_state=seed,
            n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
        )),
    ])


def make_mlp(seed=42):
    """Small dense network. Fit preprocessing separately within each training fold."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(24, 12), activation="relu", solver="adam",
            batch_size=512, max_iter=35, alpha=0.001, random_state=seed,
            early_stopping=False, n_iter_no_change=5, tol=0.0005,
        )),
    ])
