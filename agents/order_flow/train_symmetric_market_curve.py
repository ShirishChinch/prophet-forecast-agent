"""Train a symmetric one-curve market-prior calibrator.

The model is intentionally small:

    forecast_probability - market_probability
        = coefficient * (p - 0.5) * abs(p - 0.5)

Training duplicates every YES observation as its NO complement. A 90c YES row
therefore also contributes a 10c NO row, forcing the runtime curve to satisfy
f(1 - p) == 1 - f(p).

The fitted residual uses market_probability - actual_probability as a simple
historical proxy for forecast-minus-actual over/underconfidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.market_calibration import DEFAULT_ARTIFACT_PATH


DEFAULT_INPUT_DIR = Path("data/sector_15d")
DEFAULT_REPORT = Path("reports/symmetric_market_curve_backtest.json")
DEFAULT_CURVE_REPORT = Path("reports/symmetric_market_curve.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--curve-report", default=str(DEFAULT_CURVE_REPORT))
    parser.add_argument("--min-probability", type=float, default=0.0001)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    observed = _load_observed(Path(args.input_dir))
    if len(observed) < 200:
        raise SystemExit(f"Need at least 200 observations; found {len(observed)}")

    train, test = _chronological_market_split(observed, args.test_frac)
    sym_train = _symmetrize(train)
    coefficient = _fit_coefficient(sym_train)

    sym_test = _symmetrize(test)
    raw = sym_test["market_probability"].to_numpy(dtype=float)
    actual = sym_test["actual_probability"].to_numpy(dtype=float)
    calibrated = _apply_curve(raw, coefficient, args.min_probability)
    report = _metrics(actual, raw, calibrated)
    report.update(
        {
            "artifact": args.artifact,
            "input_dir": str(args.input_dir),
            "n_observations": int(len(observed)),
            "n_markets": int(observed["market_ticker"].nunique()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_symmetric_train": int(len(sym_train)),
            "n_symmetric_test": int(len(sym_test)),
            "coefficient": float(coefficient),
            "min_probability": float(args.min_probability),
            "table_type": "symmetric_signed_parabola_residual",
            "split_method": "market_disjoint_chronological",
            "notes": "YES rows are mirrored as NO rows. Coefficient is fit to market-actual residuals; runtime curve is p + coefficient*(p-.5)*abs(p-.5).",
        }
    )

    artifact = {
        "version": 1,
        "table_type": "symmetric_signed_parabola_residual",
        "allow_runtime": True,
        "coefficient": float(coefficient),
        "min_probability": float(args.min_probability),
        "backtest": report,
    }
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    grid = pd.DataFrame({"market_probability": np.arange(1, 100, dtype=float) / 100.0})
    grid["calibrated_probability"] = _apply_curve(
        grid["market_probability"].to_numpy(dtype=float),
        coefficient,
        args.min_probability,
    )
    grid["delta_points"] = grid["calibrated_probability"] - grid["market_probability"]
    curve_path = Path(args.curve_report)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(curve_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_observed(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("*/*_15d_observed_snapshots.csv")):
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    observed = pd.concat(frames, ignore_index=True)
    needed = {"p_yes", "resolved_yes", "market_ticker", "observed_time"}
    missing = needed.difference(observed.columns)
    if missing:
        raise ValueError(f"Observed CSVs missing required columns: {sorted(missing)}")

    observed = observed.dropna(subset=["p_yes", "resolved_yes", "market_ticker", "observed_time"]).copy()
    observed["market_probability"] = observed["p_yes"].astype(float).clip(0.01, 0.99)
    observed["actual_probability"] = observed["resolved_yes"].astype(float).clip(0.0, 1.0)
    observed["observed_time"] = pd.to_datetime(observed["observed_time"], utc=True, errors="coerce", format="mixed")
    return observed.dropna(subset=["observed_time"])


def _chronological_market_split(
    frame: pd.DataFrame,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_times = frame.groupby("market_ticker")["observed_time"].max().sort_values()
    split_markets = max(1, int(len(market_times) * (1.0 - test_frac)))
    train_markets = set(market_times.iloc[:split_markets].index)
    train = frame[frame["market_ticker"].isin(train_markets)].copy()
    test = frame[~frame["market_ticker"].isin(train_markets)].copy()
    return train, test


def _symmetrize(frame: pd.DataFrame) -> pd.DataFrame:
    yes = frame[["market_probability", "actual_probability"]].copy()
    no = pd.DataFrame(
        {
            "market_probability": 1.0 - yes["market_probability"].to_numpy(dtype=float),
            "actual_probability": 1.0 - yes["actual_probability"].to_numpy(dtype=float),
        }
    )
    return pd.concat([yes, no], ignore_index=True)


def _fit_coefficient(frame: pd.DataFrame) -> float:
    probability = frame["market_probability"].to_numpy(dtype=float)
    actual = frame["actual_probability"].to_numpy(dtype=float)
    centered = probability - 0.5
    feature = centered * np.abs(centered)
    target = probability - actual
    denominator = float(np.dot(feature, feature))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(target, feature) / denominator)


def _apply_curve(probability: np.ndarray, coefficient: float, min_probability: float) -> np.ndarray:
    centered = probability - 0.5
    calibrated = probability + coefficient * centered * np.abs(centered)
    min_probability = max(0.0, min(0.01, float(min_probability)))
    return np.clip(calibrated, min_probability, 1.0 - min_probability)


def _metrics(actual: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, Any]:
    raw_brier = float(np.mean((raw - actual) ** 2))
    calibrated_brier = float(np.mean((calibrated - actual) ** 2))
    return {
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": raw_brier - calibrated_brier,
        "mean_raw_probability": float(np.mean(raw)),
        "mean_calibrated_probability": float(np.mean(calibrated)),
        "mean_actual_probability": float(np.mean(actual)),
    }


if __name__ == "__main__":
    main()
