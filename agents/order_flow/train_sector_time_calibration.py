"""Train sector x time-left x odds-cent lookup calibration tables.

This is intentionally not ML. It estimates:

    P(resolve YES | sector, time_to_close_bucket, market_odds_cent)

and falls back to broader lookup tables when exact cells are sparse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.market_calibration import DEFAULT_ARTIFACT_PATH
from agents.order_flow.features import classify_market_template


TIME_BUCKETS = ("<=1m", "<=5m", "<=15m", "<=1h", "<=4h", "<=24h", "<=7d", ">7d", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        action="append",
        default=[],
        help="Folder with markets.csv and trades.csv. Repeatable.",
    )
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--report", default="reports/sector_time_calibration_backtest.json")
    parser.add_argument("--table-report", default="reports/sector_time_calibration_table.csv")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--smoothing-strength", type=float, default=20.0)
    parser.add_argument("--max-delta", type=float, default=0.25)
    args = parser.parse_args()

    data_dirs = args.data_dir or ["data/kalshi_order_flow_labeled_no_llm"]
    frame = build_sector_time_frame(data_dirs)
    if len(frame) < 200:
        raise SystemExit(f"Need at least 200 labeled snapshots; found {len(frame)}")

    train, test = _chronological_market_split(frame, args.test_frac)
    tables = _fit_tables(train, args.smoothing_strength, args.max_delta)
    pred = np.array(
        [
            _lookup_probability(
                row=row,
                tables=tables,
                min_samples=args.min_samples,
            )
            for row in test.itertuples(index=False)
        ],
        dtype=float,
    )
    baseline = test["prior"].to_numpy(dtype=float)
    outcome = test["resolved_yes"].to_numpy(dtype=float)
    report = _metrics(outcome, baseline, pred)
    report.update(
        {
            "artifact": args.artifact,
            "data_dirs": data_dirs,
            "n_rows": int(len(frame)),
            "n_markets": int(frame["market_ticker"].nunique()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "min_samples": int(args.min_samples),
            "smoothing_strength": float(args.smoothing_strength),
            "max_delta": float(args.max_delta),
            "split_method": "market_disjoint_chronological",
            "table_type": "sector_time_odds",
            "notes": "No regression/LLM. Lookup fallback order: sector+time, sector+ALL, ALL+time, ALL+ALL.",
        }
    )

    artifact = {
        "version": 2,
        "table_type": "sector_time_odds",
        "allow_runtime": True,
        "time_buckets": TIME_BUCKETS,
        "min_samples": int(args.min_samples),
        "smoothing_strength": float(args.smoothing_strength),
        "max_delta": float(args.max_delta),
        "tables": tables,
        "backtest": report,
    }
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    rows = [{"lookup_key": key, **row} for key, table in tables.items() for row in table]
    table_path = Path(args.table_report)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_sector_time_frame(data_dirs: list[str]) -> pd.DataFrame:
    """Build timestamped odds snapshots joined to resolved market outcomes."""
    frames: list[pd.DataFrame] = []
    for data_dir in data_dirs:
        root = Path(data_dir)
        markets_path = root / "markets.csv"
        trades_path = root / "trades.csv"
        if not markets_path.exists() or not trades_path.exists():
            continue
        markets = _normalize_columns(pd.read_csv(markets_path))
        trades = _normalize_columns(pd.read_csv(trades_path))
        if markets.empty or trades.empty:
            continue

        markets = _market_labels(markets)
        trades = _trade_snapshots(trades)
        if markets.empty or trades.empty:
            continue

        joined = trades.merge(markets, on="market_ticker", how="inner")
        joined = joined.dropna(subset=["timestamp", "prior", "resolved_yes"])
        if joined.empty:
            continue
        joined["hours_to_close"] = joined.apply(_hours_to_close, axis=1)
        joined["time_bucket"] = joined["hours_to_close"].map(_time_bucket)
        joined["sector"] = joined.apply(
            lambda row: classify_market_template(
                str(row.get("title") or ""),
                str(row.get("market_ticker") or ""),
            ),
            axis=1,
        )
        joined["odds_cent"] = joined["prior"].map(_odds_cent)
        frames.append(
            joined[
                [
                    "market_ticker",
                    "timestamp",
                    "title",
                    "sector",
                    "time_bucket",
                    "hours_to_close",
                    "odds_cent",
                    "prior",
                    "resolved_yes",
                ]
            ].copy()
        )

    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(["market_ticker", "timestamp", "prior"])
    frame = frame.sort_values(["timestamp", "market_ticker"])
    return frame


def _fit_tables(frame: pd.DataFrame, smoothing_strength: float, max_delta: float) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    sector_values = sorted(str(value) for value in frame["sector"].dropna().unique())
    table_specs: list[tuple[str, pd.DataFrame]] = [("ALL|ALL", frame)]
    table_specs.extend((f"ALL|{bucket}", frame[frame["time_bucket"] == bucket]) for bucket in TIME_BUCKETS)
    for sector in sector_values:
        sector_frame = frame[frame["sector"] == sector]
        table_specs.append((f"{sector}|ALL", sector_frame))
        for bucket in TIME_BUCKETS:
            table_specs.append((f"{sector}|{bucket}", sector_frame[sector_frame["time_bucket"] == bucket]))

    for key, group in table_specs:
        tables[key] = _fit_odds_rows(group, smoothing_strength, max_delta)
    return tables


def _fit_odds_rows(frame: pd.DataFrame, smoothing_strength: float, max_delta: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for odds_cent in range(1, 100):
        group = frame[frame["odds_cent"] == odds_cent]
        n = int(len(group))
        prior = odds_cent / 100.0
        mean_prior = float(group["prior"].mean()) if n else prior
        empirical = float(group["resolved_yes"].mean()) if n else mean_prior
        smoothed = ((empirical * n) + (mean_prior * smoothing_strength)) / (n + smoothing_strength) if n else mean_prior
        calibrated = max(mean_prior - max_delta, min(mean_prior + max_delta, smoothed))
        rows.append(
            {
                "odds_cent": odds_cent,
                "n": n,
                "mean_prior": _clip(mean_prior),
                "empirical_yes_rate": _clip(empirical),
                "calibrated_probability": _clip(calibrated),
            }
        )
    return rows


def _lookup_probability(row: Any, tables: dict[str, list[dict[str, Any]]], min_samples: int) -> float:
    sector = str(row.sector)
    time_bucket = str(row.time_bucket)
    odds_cent = int(row.odds_cent)
    for key in (f"{sector}|{time_bucket}", f"{sector}|ALL", f"ALL|{time_bucket}", "ALL|ALL"):
        table = tables.get(key)
        if not table:
            continue
        cell = table[odds_cent - 1]
        if int(cell.get("n") or 0) >= min_samples:
            return float(cell["calibrated_probability"])
    return _clip(float(row.prior))


def _chronological_market_split(frame: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_times = frame.groupby("market_ticker")["timestamp"].max().sort_values()
    split_markets = max(1, int(len(market_times) * (1.0 - test_frac)))
    train_tickers = set(market_times.iloc[:split_markets].index)
    train = frame[frame["market_ticker"].isin(train_tickers)].copy()
    test = frame[~frame["market_ticker"].isin(train_tickers)].copy()
    return train, test


def _market_labels(markets: pd.DataFrame) -> pd.DataFrame:
    ticker_col = _first_existing(markets, ["ticker", "market_ticker"])
    if ticker_col is None:
        return pd.DataFrame()
    result_col = _first_existing(markets, ["result", "resolution", "outcome", "settlement_value"])
    if result_col is None:
        return pd.DataFrame()
    frame = markets.copy()
    frame["market_ticker"] = frame[ticker_col].astype(str)
    frame["resolved_yes"] = frame[result_col].map(_resolved_yes)
    deadline_col = _first_existing(frame, ["expected_expiration_time", "close_time", "expiration_time", "latest_expiration_time"])
    frame["deadline"] = pd.to_datetime(frame[deadline_col], utc=True, errors="coerce") if deadline_col else pd.NaT
    if "title" not in frame.columns:
        frame["title"] = frame["market_ticker"]
    return frame[["market_ticker", "title", "deadline", "resolved_yes"]].dropna(subset=["resolved_yes"])


def _trade_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    ticker_col = _first_existing(trades, ["ticker", "market_ticker"])
    time_col = _first_existing(trades, ["created_time", "timestamp", "time"])
    price_col = _first_existing(trades, ["yes_price_dollars", "yes_price", "price", "price_dollars"])
    if ticker_col is None or time_col is None or price_col is None:
        return pd.DataFrame()
    frame = trades.copy()
    frame["market_ticker"] = frame[ticker_col].astype(str)
    frame["timestamp"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame["prior"] = frame[price_col].map(_to_probability)
    return frame[["market_ticker", "timestamp", "prior"]].dropna()


def _hours_to_close(row: pd.Series) -> float | None:
    deadline = row.get("deadline")
    timestamp = row.get("timestamp")
    if pd.isna(deadline) or pd.isna(timestamp):
        return None
    return (deadline - timestamp).total_seconds() / 3600.0


def _time_bucket(hours: float | None) -> str:
    if hours is None or not np.isfinite(hours):
        return "unknown"
    minutes = hours * 60.0
    if minutes <= 1:
        return "<=1m"
    if minutes <= 5:
        return "<=5m"
    if minutes <= 15:
        return "<=15m"
    if hours <= 1:
        return "<=1h"
    if hours <= 4:
        return "<=4h"
    if hours <= 24:
        return "<=24h"
    if hours <= 168:
        return "<=7d"
    return ">7d"


def _metrics(outcome: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    baseline = np.clip(baseline, 0.01, 0.99)
    pred = np.clip(pred, 0.01, 0.99)
    baseline_brier = float(np.mean((baseline - outcome) ** 2))
    calibrated_brier = float(np.mean((pred - outcome) ** 2))
    return {
        "baseline_brier": baseline_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": baseline_brier - calibrated_brier,
        "relative_brier_improvement": (
            (baseline_brier - calibrated_brier) / baseline_brier if baseline_brier else 0.0
        ),
        "baseline_log_loss": _log_loss(outcome, baseline),
        "calibrated_log_loss": _log_loss(outcome, pred),
        "mean_abs_delta": float(np.mean(np.abs(pred - baseline))),
    }


def _log_loss(outcome: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(pred, 0.01, 0.99)
    return float(-np.mean(outcome * np.log(pred) + (1.0 - outcome) * np.log(1.0 - pred)))


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_").replace("-", "_") for column in result.columns]
    return result


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _resolved_yes(value: Any) -> float | None:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "yes_won", "settled_yes"}:
        return 1.0
    if text in {"no", "n", "false", "0", "no_won", "settled_no"}:
        return 0.0
    return None


def _to_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if number > 1.0:
        number /= 100.0
    if 0.0 <= number <= 1.0:
        return _clip(number)
    return None


def _odds_cent(value: float) -> int:
    return max(1, min(99, int(round(float(value) * 100.0))))


def _clip(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


if __name__ == "__main__":
    main()
