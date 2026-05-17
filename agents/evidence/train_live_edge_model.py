"""Train a compact residual model from the live canonical feature matrix.

Target:

    residual = resolved_yes - kalshi_prior

The model is deliberately linear and capped. It is used only as a small edge
nudge around the market prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.evidence.edge_feature_schema import build_market_rule_features, default_llm_feature_row


META_COLUMNS = {
    "snapshot_time",
    "finish_time",
    "template",
    "ticker",
    "event_ticker",
    "title",
    "prior",
    "result",
    "resolved_yes",
    "residual_target",
    "label_status",
    "settlement_ts",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Labeled canonical matrix CSV.")
    parser.add_argument("--artifact", default="agents/evidence/artifacts/live_edge_residual.joblib")
    parser.add_argument("--report", default="reports/live_edge_residual_backtest.json")
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--max-delta", type=float, default=0.04)
    parser.add_argument("--min-nonzero-share", type=float, default=0.01)
    parser.add_argument("--max-features", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.matrix)
    training = build_training_frame(frame)
    feature_columns = _candidate_feature_columns(training, args.min_nonzero_share)
    if not feature_columns:
        raise SystemExit("No usable feature columns found.")

    training = training.sort_values(["finish_time", "ticker"], na_position="last").reset_index(drop=True)
    split = max(1, int(len(training) * (1.0 - args.test_frac)))
    train = training.iloc[:split].copy()
    test = training.iloc[split:].copy()
    if len(test) < 40:
        raise SystemExit(f"Need at least 40 test rows; found {len(test)}.")

    screened = _screen_features(train, feature_columns, max_delta=args.max_delta)
    selected_features = [
        row["feature"]
        for row in screened
        if row["kept"]
    ][: args.max_features]

    if selected_features:
        model = _fit_model(train, selected_features)
        raw_delta = model.predict(test[selected_features])
    else:
        model = None
        raw_delta = np.zeros(len(test), dtype=float)

    prior = test["prior"].to_numpy(dtype=float)
    outcome = test["resolved_yes"].to_numpy(dtype=float)
    delta = np.clip(raw_delta, -args.max_delta, args.max_delta)
    pred = np.clip(prior + delta, 0.01, 0.99)
    baseline = np.clip(prior, 0.01, 0.99)
    intercept_delta = float(train["residual_target"].mean())
    intercept_pred = np.clip(
        prior + np.clip(intercept_delta, -args.max_delta, args.max_delta),
        0.01,
        0.99,
    )
    report = _metrics(outcome, baseline, pred)
    intercept_report = _metrics(outcome, baseline, intercept_pred)
    report.update(
        {
            "n_rows": int(len(training)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_candidate_features": int(len(feature_columns)),
            "selected_features": selected_features,
            "dropped_features": [feature for feature in feature_columns if feature not in selected_features],
            "feature_screen": screened,
            "feature_correlations": _feature_correlations(training, feature_columns),
            "standardized_coefficients": _standardized_coefficients(model, selected_features),
            "max_delta": args.max_delta,
            "intercept_only_delta": intercept_delta,
            "intercept_only_brier": intercept_report["model_brier"],
            "feature_incremental_brier_improvement": intercept_report["model_brier"] - report["model_brier"],
            "deployable": bool(selected_features) and report["brier_improvement"] > 0.0,
            "artifact": args.artifact,
        }
    )
    report["template_metrics"] = _template_metrics(test, pred)

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": selected_features,
            "all_feature_columns": feature_columns,
            "max_delta": args.max_delta,
            "report": report,
        },
        artifact_path,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric residual-training frame with engineered edge features."""
    data = frame.copy()
    data["prior"] = pd.to_numeric(data["prior"], errors="coerce").clip(0.01, 0.99)
    data["resolved_yes"] = pd.to_numeric(data["resolved_yes"], errors="coerce")
    data = data.dropna(subset=["prior", "resolved_yes"]).copy()
    data["residual_target"] = data["resolved_yes"] - data["prior"]

    engineered_rows: list[dict[str, float]] = []
    for row in data.to_dict(orient="records"):
        event = {
            "title": row.get("title"),
            "spread": _first_present(row, "sports__market_spread", "generic__market_spread", "crypto__market_spread", "macro__market_spread"),
            "volume_fp": _first_present(row, "sports__market_volume_log", "generic__market_volume_log", "crypto__market_volume_log", "macro__market_volume_log"),
            "hours_to_finish": _first_present(row, "sports__hours_to_finish", "generic__hours_to_finish", "crypto__hours_to_finish", "macro__hours_to_finish"),
        }
        features = build_market_rule_features(event, float(row["prior"]))
        # Existing matrices store log volume already, so preserve it instead of
        # log1p(log_volume) when this path is used.
        existing_volume_log = _first_present(row, "sports__market_volume_log", "generic__market_volume_log", "crypto__market_volume_log", "macro__market_volume_log")
        if existing_volume_log is not None:
            features["edge__market_volume_log"] = _to_float(existing_volume_log)
        engineered_rows.append(features)

    engineered = pd.DataFrame(engineered_rows, index=data.index)
    for column in engineered.columns:
        data[column] = engineered[column].astype(float)

    for feature in default_llm_feature_row():
        if feature not in data.columns:
            data[feature] = 0.0

    for column in data.columns:
        if "__" in column:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    return data


def _candidate_feature_columns(frame: pd.DataFrame, min_nonzero_share: float) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS or "__" not in column:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if values.nunique(dropna=False) <= 1:
            continue
        if float((values.abs() > 1e-12).mean()) < min_nonzero_share:
            continue
        columns.append(column)
    return columns


def _screen_features(train: pd.DataFrame, feature_columns: list[str], max_delta: float) -> list[dict[str, Any]]:
    if len(train) < 60:
        return [
            {
                "feature": feature,
                "kept": False,
                "reason": "not_enough_train_rows",
                "inner_brier_improvement": 0.0,
            }
            for feature in feature_columns
        ]

    split = max(1, int(len(train) * 0.70))
    screen_train = train.iloc[:split]
    screen_test = train.iloc[split:]
    baseline = np.clip(screen_test["prior"].to_numpy(dtype=float), 0.01, 0.99)
    outcome = screen_test["resolved_yes"].to_numpy(dtype=float)
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    intercept_delta = float(screen_train["residual_target"].mean())
    intercept_pred = np.clip(
        baseline + np.clip(intercept_delta, -max_delta, max_delta),
        0.01,
        0.99,
    )
    intercept_brier = float(np.mean((intercept_pred - outcome) ** 2))
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = pd.to_numeric(screen_train[feature], errors="coerce").fillna(0.0)
        if values.nunique(dropna=False) <= 1:
            rows.append(
                {
                    "feature": feature,
                    "kept": False,
                    "reason": "constant",
                    "inner_brier_improvement": 0.0,
                    "nonzero_share_train": float((values.abs() > 1e-12).mean()),
                }
            )
            continue
        model = _fit_ridge(screen_train, [feature])
        raw_delta = model.predict(screen_test[[feature]])
        pred = np.clip(baseline + np.clip(raw_delta, -max_delta, max_delta), 0.01, 0.99)
        model_brier = float(np.mean((pred - outcome) ** 2))
        improvement = baseline_brier - model_brier
        incremental_improvement = intercept_brier - model_brier
        rows.append(
            {
                "feature": feature,
                "kept": bool(incremental_improvement > 0.0),
                "reason": "beats_intercept_calibration" if incremental_improvement > 0.0 else "no_incremental_feature_edge",
                "inner_baseline_brier": baseline_brier,
                "inner_intercept_brier": intercept_brier,
                "inner_model_brier": model_brier,
                "inner_brier_improvement": improvement,
                "inner_incremental_brier_improvement": incremental_improvement,
                "inner_relative_brier_improvement": improvement / baseline_brier if baseline_brier else 0.0,
                "nonzero_share_train": float((values.abs() > 1e-12).mean()),
            }
        )
    return sorted(rows, key=lambda row: float(row.get("inner_incremental_brier_improvement") or 0.0), reverse=True)


def _fit_ridge(train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 20))),
        ]
    )
    model.fit(train[feature_columns], train["residual_target"])
    return model


def _fit_model(train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    if len(feature_columns) >= 3 and len(train) >= 100:
        estimator = ElasticNetCV(
            cv=5,
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            max_iter=5000,
            random_state=7,
        )
    else:
        estimator = RidgeCV(alphas=np.logspace(-4, 3, 20))
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    model.fit(train[feature_columns], train["residual_target"])
    return model


def _metrics(outcome: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    model_brier = float(np.mean((pred - outcome) ** 2))
    improvement = baseline_brier - model_brier
    return {
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "brier_improvement": improvement,
        "relative_brier_improvement": improvement / baseline_brier if baseline_brier else 0.0,
    }


def _template_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    scored = test.copy()
    scored["_pred"] = pred
    for template, group in scored.groupby("template"):
        if len(group) < 5:
            continue
        outcome = group["resolved_yes"].to_numpy(dtype=float)
        baseline = group["prior"].to_numpy(dtype=float)
        values = group["_pred"].to_numpy(dtype=float)
        metric = _metrics(outcome, baseline, values)
        metric["n"] = float(len(group))
        rows[str(template)] = metric
    return rows


def _feature_correlations(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
        corr_residual = values.corr(frame["residual_target"])
        corr_outcome = values.corr(frame["resolved_yes"])
        rows.append(
            {
                "feature": feature,
                "n": int(len(frame)),
                "nonzero_share": float((values.abs() > 1e-12).mean()),
                "corr_with_residual_target": _safe_float(corr_residual),
                "corr_with_resolved_yes": _safe_float(corr_outcome),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["corr_with_residual_target"] or 0.0)), reverse=True)


def _standardized_coefficients(model: Pipeline | None, feature_columns: list[str]) -> list[dict[str, float | str]]:
    if model is None or not feature_columns:
        return []
    estimator = model.named_steps.get("model") or model.named_steps.get("ridge")
    coefs = getattr(estimator, "coef_", [])
    rows = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in zip(feature_columns, coefs)
    ]
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)


def _first_present(row: dict[str, Any], *columns: str) -> Any:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


if __name__ == "__main__":
    main()
