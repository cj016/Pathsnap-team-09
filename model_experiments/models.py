"""Candidate classifiers for the six approved map-matching features."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(seed=42):
    """Regularised linear baseline with fold-fitted imputation and scaling."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=seed,
            solver="lbfgs",
        )),
    ])


def make_random_forest(seed=42):
    """Bagged decision trees for nonlinear feature interactions."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed,
            n_jobs=2,
        )),
    ])


def make_knn(seed=42):
    """Distance-weighted KNN; run_checkpoint caps its fit sample for practicality."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=2)),
    ])


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
    """Small dense network with fold-fitted median imputation and scaling."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(24, 12), activation="relu", solver="adam",
            batch_size=512, max_iter=35, alpha=0.001, random_state=seed,
            early_stopping=False, n_iter_no_change=5, tol=0.0005,
        )),
    ])
