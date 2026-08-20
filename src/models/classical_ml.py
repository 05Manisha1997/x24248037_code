from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def _is_regression(y: np.ndarray) -> bool:
    return len(np.unique(y)) > 10 or y.dtype in (np.float32, np.float64)


def build_classical_model(name: str, task: str = "classification") -> Any:
    name = name.lower()
    if task == "regression":
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
            "svm": Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0))]),
            "xgboost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=1),
        }
    else:
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight="balanced"),
            "svm": CalibratedClassifierCV(
                LinearSVC(max_iter=3000, class_weight="balanced", random_state=42),
                cv=3,
            ),
            "xgboost": XGBClassifier(
                n_estimators=100, random_state=42, n_jobs=1,
                eval_metric="logloss",
            ),
        }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models)}")
    return models[name]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if _is_regression(y_true):
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics["roc_auc"] = 0.0
    return metrics


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, float]:
    scoring = "r2" if _is_regression(y) else "f1_weighted"
    n_samples = len(y)
    if n_samples > 40_000:
        return {"cv_mean": float("nan"), "cv_std": float("nan"), "scoring": scoring, "cv_skipped": True}
    cv_folds = 3 if n_samples > 15_000 else cv
    X = np.asarray(X, dtype=np.float32)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
    return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std()), "scoring": scoring}


def train_and_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    task = "regression" if _is_regression(y_train) else "classification"
    model = build_classical_model(model_name, task=task)
    cv_results = cross_validate_model(model, X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = None
    if task == "classification" and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif task == "classification" and hasattr(model, "named_steps"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

    test_metrics = evaluate_predictions(y_test, y_pred, y_prob)
    return {"model": model_name, "task": task, "cv": cv_results, "test": test_metrics, "estimator": model}
