"""Shared training utilities for economics models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.economics.ml_artifacts import ARTIFACT_DIR
from agents.economics.training.dataset_builder import build_supervised_dataset


MODEL_CONFIGS = {
    "yield": {
        "horizon_days": 28,
        "sample_freq": "W-FRI",
        "max_features": 80,
        "select_k": 35,
        "estimators": [
            "ridge_selected",
            "elasticnet_selected",
            "hgb",
            "xgboost",
        ],
        "target_mode": "delta_to_current",
    },
    "inflation": {
        "horizon_days": 35,
        "sample_freq": "W-FRI",
        "max_features": 120,
        "select_k": 50,
        "estimators": [
            "hgb",
            "xgboost",
        ],
        "target_mode": "delta_to_current",
        "elasticnet_preselect": True,
    },
    "growth": {
        "horizon_days": 91,
        "sample_freq": "W-FRI",
        "max_features": 120,
        "select_k": 50,
        "estimators": [
            "hgb",
            "xgboost",
        ],
        "target_mode": "delta_to_current",
        "elasticnet_preselect": True,
    },
    "policy": {
        "horizon_days": 35,
        "sample_freq": "W-FRI",
        "max_features": 80,
        "select_k": 35,
        "estimators": [
            "hgb",
            "xgboost",
        ],
        "target_mode": "policy_delta",
        "elasticnet_preselect": True,
    },
}


def train_model_type(
    *,
    model_type: str,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one small ML artifact and save it."""
    config = MODEL_CONFIGS[model_type]
    dataset = build_supervised_dataset(
        data_dir=data_dir,
        model_type=model_type,
        horizon_days=int(config["horizon_days"]),
        sample_freq=str(config["sample_freq"]),
        max_features=int(config["max_features"]),
    )
    X = dataset.X
    y = dataset.y
    if len(X) < 80:
        raise ValueError(f"Not enough rows to train {model_type}: {len(X)}")

    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    input_feature_names = list(X.columns)

    preselection: dict[str, Any] = {
        "method": "none",
        "selected_features": input_feature_names,
        "n_input_features": len(input_feature_names),
        "n_selected_features": len(input_feature_names),
    }
    if bool(config.get("elasticnet_preselect")):
        selected_features, preselection = _elasticnet_preselect_features(
            X_train=X_train,
            y_train=y_train,
            feature_names=input_feature_names,
        )
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    model, model_name, candidate_metrics = _fit_best_model(
        estimator_names=list(config["estimators"]),
        select_k=int(config["select_k"]),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    baseline_train = np.zeros(len(y_train))
    baseline_test = np.zeros(len(y_test))
    residual_sigma = float(np.nanstd(y_train.to_numpy() - pred_train))
    metrics = {
        "train_rmse": _rmse(y_train, pred_train),
        "test_rmse": _rmse(y_test, pred_test),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_r2": float(r2_score(y_test, pred_test)) if len(y_test) > 1 else 0.0,
        "baseline_test_rmse": _rmse(y_test, baseline_test),
        "baseline_test_mae": float(mean_absolute_error(y_test, baseline_test)),
        "baseline_test_r2": float(r2_score(y_test, baseline_test)) if len(y_test) > 1 else 0.0,
        "rmse_improvement_vs_baseline": _relative_improvement(
            baseline_rmse=_rmse(y_test, baseline_test),
            model_rmse=_rmse(y_test, pred_test),
        ),
        "best_estimator": model_name,
        "candidate_metrics": candidate_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    artifact = {
        "version": 1,
        "model_type": model_type,
        "estimator": model_name,
        "model": model,
        "input_feature_names": input_feature_names,
        "feature_names": list(X_train.columns),
        "selected_features": list(X_train.columns),
        "selected_feature_names": _selected_feature_names(model, list(X_train.columns)),
        "feature_preselection": preselection,
        "target_tickers": dataset.target_tickers,
        "target_mode": config["target_mode"],
        "residual_sigma": residual_sigma,
        "metrics": metrics,
        "dataset_diagnostics": dataset.diagnostics,
        "notes": "Small JPMaQS point-in-time model. Keep heavily blended with Kalshi prior until Brier backtests prove edge.",
    }

    out_dir = Path(output_dir) if output_dir else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_type}_model.joblib"
    joblib.dump(artifact, path)
    return {"artifact_path": str(path), "metrics": metrics, "dataset": dataset.diagnostics}


def _build_estimator(name: str, select_k: int | None = None) -> Pipeline:
    use_selector = name.endswith("_selected")
    base_name = name.replace("_selected", "")
    if base_name == "ridge":
        estimator = Ridge(alpha=10.0)
    elif base_name == "elasticnet":
        estimator = ElasticNet(alpha=0.01, l1_ratio=0.20, max_iter=10000)
    elif base_name == "hgb":
        estimator = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=17,
        )
    elif base_name == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as exc:
            raise ValueError(f"xgboost unavailable: {exc}") from exc

        estimator = XGBRegressor(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=8.0,
            reg_alpha=0.25,
            objective="reg:squarederror",
            random_state=17,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown estimator: {name}")

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_selector:
        steps.append(("selector", SelectKBest(score_func=f_regression, k=select_k or 50)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _fit_best_model(
    *,
    estimator_names: list[str],
    select_k: int,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> tuple[Pipeline, str, dict[str, dict[str, float | str]]]:
    """Train candidate estimators and pick the best holdout RMSE."""
    best_model: Pipeline | None = None
    best_name = ""
    best_rmse = float("inf")
    metrics: dict[str, dict[str, float | str]] = {}

    for name in estimator_names:
        try:
            candidate = _build_estimator(name, select_k=min(select_k, X_train.shape[1]))
            candidate.fit(X_train, y_train)
            pred = candidate.predict(X_test)
            rmse = _rmse(y_test, pred)
            mae = float(mean_absolute_error(y_test, pred))
            r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0
            metrics[name] = {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}
        except Exception as exc:
            metrics[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = candidate

    if best_model is None:
        raise ValueError(f"No estimator could be trained. Metrics: {metrics}")
    return best_model, best_name, metrics


def _elasticnet_preselect_features(
    *,
    X_train: Any,
    y_train: Any,
    feature_names: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Select nonzero ElasticNetCV features using only the training split."""
    selector = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNetCV(
                    cv=5,
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    max_iter=5000,
                    alphas=np.logspace(-3, 0, 60),
                    random_state=17,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        selector.fit(X_train, y_train)
    elasticnet = selector.named_steps["elasticnet"]
    coefs = np.asarray(elasticnet.coef_, dtype=float)
    selected_features = [
        name for name, coef in zip(feature_names, coefs, strict=False) if float(coef) != 0.0
    ]
    n_input = len(feature_names)
    n_selected = len(selected_features)
    if n_selected >= 0.5 * n_input:
        warnings.warn(
            (
                "ElasticNetCV preselection kept too many features: "
                f"{n_selected}/{n_input}. The training assertion will fail."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    assert n_selected < 0.5 * n_input, (
        "ElasticNetCV preselection must keep fewer than half of input features; "
        f"kept {n_selected}/{n_input}"
    )
    if not selected_features:
        raise ValueError("ElasticNetCV preselection selected zero features")
    fold_counts = _elasticnet_fold_nonzero_counts(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        alpha=float(elasticnet.alpha_),
        l1_ratio=float(elasticnet.l1_ratio_),
    )
    policy_rate_fold_counts = {
        name: count for name, count in fold_counts.items() if "_policy_rate_level__" in name
    }
    return selected_features, {
        "method": "ElasticNetCV",
        "cv": 5,
        "l1_ratio": [0.1, 0.5, 0.9, 1.0],
        "max_iter": 5000,
        "n_input_features": n_input,
        "n_selected_features": n_selected,
        "selected_features": selected_features,
        "alpha": float(elasticnet.alpha_),
        "l1_ratio_selected": float(elasticnet.l1_ratio_),
        "fold_nonzero_counts": fold_counts,
        "policy_rate_fold_nonzero_counts": policy_rate_fold_counts,
        "policy_rate_features_nonzero_in_at_least_3_folds": [
            name for name, count in policy_rate_fold_counts.items() if count >= 3
        ],
    }


def _elasticnet_fold_nonzero_counts(
    *,
    X_train: Any,
    y_train: Any,
    feature_names: list[str],
    alpha: float,
    l1_ratio: float,
) -> dict[str, int]:
    counts = {name: 0 for name in feature_names}
    splitter = KFold(n_splits=5, shuffle=False)
    for train_idx, _ in splitter.split(X_train):
        candidate = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=5000,
                        random_state=17,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            candidate.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        coefs = np.asarray(candidate.named_steps["elasticnet"].coef_, dtype=float)
        for name, coef in zip(feature_names, coefs, strict=False):
            if float(coef) != 0.0:
                counts[name] += 1
    return counts


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _relative_improvement(*, baseline_rmse: float, model_rmse: float) -> float:
    if baseline_rmse <= 0:
        return 0.0
    return float((baseline_rmse - model_rmse) / baseline_rmse)


def _selected_feature_names(model: Pipeline, feature_names: list[str]) -> list[str]:
    selector = model.named_steps.get("selector")
    if selector is None or not hasattr(selector, "get_support"):
        return feature_names
    mask = selector.get_support()
    return [name for name, keep in zip(feature_names, mask, strict=False) if bool(keep)]
