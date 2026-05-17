"""Train a simple linear residual model on LLM-extracted evidence features.

This is for unstructured evidence features only. It joins timestamped evidence
rows to Kalshi trade snapshots, then learns:

    target = resolved_yes - kalshi_prior_at_that_time

The model is intentionally small and linear. It should only be deployed if it
beats the Kalshi prior on held-out Brier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.order_flow.features import build_order_flow_dataset


META_COLUMNS = {
    "template",
    "event_ticker",
    "market_ticker",
    "timestamp",
    "as_of_time",
    "source_urls",
    "short_rationale",
    # This is a quality-control field, not an alpha feature.
    "llm_temporal_leakage_risk",
    "temporal_leakage_risk",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow_labeled")
    parser.add_argument("--evidence-features", default="data/evidence_feature_rows.csv")
    parser.add_argument("--artifact", default="agents/evidence/artifacts/evidence_residual.joblib")
    parser.add_argument("--report", default="reports/evidence_residual_backtest.json")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=0.03)
    parser.add_argument("--min-rows", type=int, default=80)
    parser.add_argument(
        "--select-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Screen evidence features on an inner time split before fitting the final model.",
    )
    parser.add_argument(
        "--min-feature-improvement",
        type=float,
        default=0.0,
        help="Minimum inner-split Brier improvement required for a feature to survive.",
    )
    parser.add_argument(
        "--max-selected-features",
        type=int,
        default=8,
        help="Hard cap on surviving evidence features.",
    )
    parser.add_argument(
        "--min-feature-nonzero-share",
        type=float,
        default=0.05,
        help="Drop sparse features before screening unless at least this share is nonzero in train.",
    )
    args = parser.parse_args()

    dataset = build_order_flow_dataset(args.data_dir)
    evidence = _load_evidence_features(Path(args.evidence_features))
    training = _join_to_order_flow(evidence, dataset)
    if len(training) < args.min_rows:
        raise SystemExit(
            f"Need at least {args.min_rows} joined evidence rows; found {len(training)}. "
            "Generate more timestamped LLM evidence rows first."
        )

    feature_columns = _numeric_feature_columns(training)
    if not feature_columns:
        raise SystemExit("No numeric evidence feature columns found.")

    training = training.sort_values("timestamp").reset_index(drop=True)
    split = max(1, int(len(training) * (1.0 - args.test_frac)))
    train = training.iloc[:split]
    test = training.iloc[split:]
    if len(test) < 20:
        raise SystemExit(f"Need at least 20 test rows; found {len(test)}.")

    selected_features = feature_columns
    feature_screen: list[dict[str, Any]] = []
    if args.select_features:
        selected_features, feature_screen = _select_features(
            train=train,
            feature_columns=feature_columns,
            max_delta=args.max_delta,
            min_feature_improvement=args.min_feature_improvement,
            max_selected_features=args.max_selected_features,
            min_feature_nonzero_share=args.min_feature_nonzero_share,
        )

    if selected_features:
        model = _fit_ridge(train, selected_features)
        raw_delta = model.predict(test[selected_features])
    else:
        model = None
        raw_delta = np.zeros(len(test), dtype=float)

    prior = test["prior"].to_numpy(dtype=float)
    outcome = test["resolved_yes"].to_numpy(dtype=float)
    delta = np.clip(raw_delta, -args.max_delta, args.max_delta)
    pred = np.clip(prior + delta, 0.01, 0.99)
    baseline = np.clip(prior, 0.01, 0.99)

    report = _metrics(outcome, baseline, pred)
    report.update(
        {
            "n_rows": int(len(training)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "feature_columns": feature_columns,
            "selected_features": selected_features,
            "dropped_features": [feature for feature in feature_columns if feature not in selected_features],
            "feature_screen": feature_screen,
            "max_delta": args.max_delta,
            "artifact": args.artifact,
            "evidence_features": args.evidence_features,
            "deployable": bool(selected_features) and report["brier_improvement"] > 0.0,
            "standardized_coefficients": _standardized_coefficients(model, selected_features),
        }
    )

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


def _load_evidence_features(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Evidence feature file not found or empty: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "timestamp" not in frame.columns and "as_of_time" in frame.columns:
        frame["timestamp"] = frame["as_of_time"]
    if "market_ticker" not in frame.columns:
        raise SystemExit("Evidence feature file must include market_ticker.")
    if "timestamp" not in frame.columns:
        raise SystemExit("Evidence feature file must include timestamp or as_of_time.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["market_ticker", "timestamp"])


def _join_to_order_flow(evidence: pd.DataFrame, dataset: Any) -> pd.DataFrame:
    base = dataset.X.reset_index(drop=True).copy()
    base["target"] = dataset.y.reset_index(drop=True)
    base["prior"] = dataset.prior.reset_index(drop=True)
    base["resolved_yes"] = dataset.resolved_yes.reset_index(drop=True)
    base = base.join(dataset.meta.reset_index(drop=True))
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base = base.dropna(subset=["market_ticker", "timestamp"])
    base = base.drop_duplicates(["market_ticker", "timestamp"], keep="last")

    evidence = evidence.copy()
    evidence["market_ticker"] = evidence["market_ticker"].astype(str)
    base["market_ticker"] = base["market_ticker"].astype(str)

    joined = evidence.merge(
        base[["market_ticker", "timestamp", "prior", "resolved_yes", "target"]],
        on=["market_ticker", "timestamp"],
        how="inner",
    )
    if not joined.empty:
        return joined

    # If exact timestamps do not match, use the nearest prior snapshot per market.
    rows: list[pd.DataFrame] = []
    for ticker, evidence_group in evidence.groupby("market_ticker"):
        base_group = base[base["market_ticker"] == ticker].sort_values("timestamp")
        if base_group.empty:
            continue
        merged = pd.merge_asof(
            evidence_group.sort_values("timestamp"),
            base_group[["timestamp", "prior", "resolved_yes", "target"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        rows.append(merged.dropna(subset=["prior", "resolved_yes", "target"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = set(META_COLUMNS) | {"prior", "resolved_yes", "target"}
    columns: list[str] = []
    for column in frame.columns:
        if column in blocked:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            frame[column] = numeric.fillna(0.0)
            columns.append(column)
    return columns


def _fit_ridge(train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 20))),
        ]
    )
    model.fit(train[feature_columns], train["target"])
    return model


def _select_features(
    train: pd.DataFrame,
    feature_columns: list[str],
    max_delta: float,
    min_feature_improvement: float,
    max_selected_features: int,
    min_feature_nonzero_share: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Keep only evidence features that beat the prior on an inner time split."""
    if len(train) < 40:
        return [], []

    inner_split = max(1, int(len(train) * 0.75))
    screen_train = train.iloc[:inner_split]
    screen_test = train.iloc[inner_split:]
    if len(screen_test) < 10:
        return [], []

    baseline = np.clip(screen_test["prior"].to_numpy(dtype=float), 0.01, 0.99)
    outcome = screen_test["resolved_yes"].to_numpy(dtype=float)
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        values = pd.to_numeric(screen_train[feature], errors="coerce").fillna(0.0)
        nonzero_share = float((values.abs() > 1e-12).mean())
        if values.nunique(dropna=False) <= 1:
            rows.append(
                {
                    "feature": feature,
                    "kept": False,
                    "reason": "constant_or_zero",
                    "inner_brier_improvement": 0.0,
                    "nonzero_share_train": nonzero_share,
                }
            )
            continue
        if nonzero_share < min_feature_nonzero_share:
            rows.append(
                {
                    "feature": feature,
                    "kept": False,
                    "reason": "too_sparse",
                    "inner_brier_improvement": 0.0,
                    "nonzero_share_train": nonzero_share,
                    "min_feature_nonzero_share": min_feature_nonzero_share,
                }
            )
            continue

        model = _fit_ridge(screen_train, [feature])
        raw_delta = model.predict(screen_test[[feature]])
        pred = np.clip(baseline + np.clip(raw_delta, -max_delta, max_delta), 0.01, 0.99)
        model_brier = float(np.mean((pred - outcome) ** 2))
        improvement = baseline_brier - model_brier
        coef = float(model.named_steps["ridge"].coef_[0])
        kept = improvement > min_feature_improvement
        rows.append(
            {
                "feature": feature,
                "kept": bool(kept),
                "reason": "improves_inner_brier" if kept else "no_inner_brier_edge",
                "inner_baseline_brier": baseline_brier,
                "inner_model_brier": model_brier,
                "inner_brier_improvement": improvement,
                "inner_relative_brier_improvement": improvement / baseline_brier if baseline_brier else 0.0,
                "standardized_coefficient": coef,
                "nonzero_share_train": nonzero_share,
            }
        )

    rows = sorted(rows, key=lambda row: float(row.get("inner_brier_improvement") or 0.0), reverse=True)
    selected = [
        str(row["feature"])
        for row in rows
        if row.get("kept")
    ][:max_selected_features]
    selected_set = set(selected)
    for row in rows:
        if row["feature"] not in selected_set and row.get("kept"):
            row["kept"] = False
            row["reason"] = "over_max_selected_features"
    return selected, rows


def _metrics(outcome: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    model_brier = float(np.mean((pred - outcome) ** 2))
    improvement = baseline_brier - model_brier
    return {
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "brier_improvement": improvement,
        "relative_brier_improvement": improvement / baseline_brier if baseline_brier else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(pred - baseline))),
    }


def _standardized_coefficients(model: Pipeline | None, feature_columns: list[str]) -> list[dict[str, float | str]]:
    """Return Ridge coefficients in standardized feature space."""
    if model is None:
        return []
    ridge = model.named_steps.get("ridge")
    coefs = getattr(ridge, "coef_", None)
    if coefs is None:
        return []
    rows = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in zip(feature_columns, coefs, strict=True)
    ]
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)


if __name__ == "__main__":
    main()
