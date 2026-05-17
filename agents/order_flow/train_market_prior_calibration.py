"""Train an empirical lookup-table calibrator for Kalshi market odds.

The baseline is the raw market-implied probability. The model is only:

    calibrated_p = historical YES frequency for the prior's probability bin

The train/test split is chronological by snapshot timestamp, so test rows are
not used to estimate the lookup table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.market_calibration import DEFAULT_ARTIFACT_PATH
from agents.order_flow.features import build_order_flow_dataset


DEFAULT_BIN_EDGES = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
    0.99,
    1.00,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kalshi_order_flow_labeled_no_llm")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--report", default="reports/market_prior_calibration_backtest.json")
    parser.add_argument("--bins-report", default="reports/market_prior_calibration_bins.csv")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--smoothing-strength", type=float, default=30.0)
    args = parser.parse_args()

    dataset = build_order_flow_dataset(args.data_dir, snapshot_stride=args.snapshot_stride)
    if len(dataset.X) < 200:
        raise SystemExit(f"Need at least 200 labeled snapshots; found {len(dataset.X)}")

    frame = dataset.meta.copy()
    frame["prior"] = dataset.prior.to_numpy(dtype=float)
    frame["resolved_yes"] = dataset.resolved_yes.to_numpy(dtype=float)
    if "template_family" not in frame.columns:
        frame["template_family"] = "generic"
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "prior", "resolved_yes"]).sort_values("timestamp")

    train, test, split_method = _chronological_market_split(frame, args.test_frac)
    if test.empty:
        raise SystemExit("No test rows after chronological split.")

    global_bins = _fit_bins(train, args.smoothing_strength)
    by_template: dict[str, list[dict[str, Any]]] = {}
    for template, group in train.groupby("template_family"):
        template_key = str(template or "generic").lower()
        if len(group) >= args.min_samples * 4:
            by_template[template_key] = _fit_bins(group, args.smoothing_strength)

    test_prior = test["prior"].to_numpy(dtype=float)
    outcome = test["resolved_yes"].to_numpy(dtype=float)
    calibrated = np.array(
        [
            _lookup_probability(
                prior=float(row.prior),
                template=str(row.template_family),
                global_bins=global_bins,
                by_template=by_template,
                min_samples=args.min_samples,
            )
            for row in test.itertuples(index=False)
        ],
        dtype=float,
    )

    report = _metrics(outcome, test_prior, calibrated)
    report.update(
        {
            "artifact": args.artifact,
            "data_dir": args.data_dir,
            "n_rows": int(len(frame)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "split_method": split_method,
            "min_samples": int(args.min_samples),
            "smoothing_strength": float(args.smoothing_strength),
            "bin_edges": DEFAULT_BIN_EDGES,
            "template_tables": sorted(by_template),
            "notes": "Lookup table fit only on chronological market-disjoint train split; baseline is raw market prior.",
        }
    )

    artifact = {
        "version": 1,
        "bin_edges": DEFAULT_BIN_EDGES,
        "min_samples": int(args.min_samples),
        "smoothing_strength": float(args.smoothing_strength),
        "global": global_bins,
        "by_template": by_template,
        "backtest": report,
    }
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    bins_frame = pd.DataFrame(
        [{"table": "global", **row} for row in global_bins]
        + [
            {"table": template, **row}
            for template, rows in by_template.items()
            for row in rows
        ]
    )
    bins_path = Path(args.bins_report)
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    bins_frame.to_csv(bins_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _fit_bins(frame: pd.DataFrame, smoothing_strength: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low, high in zip(DEFAULT_BIN_EDGES[:-1], DEFAULT_BIN_EDGES[1:]):
        if high == 1.0:
            mask = (frame["prior"] >= low) & (frame["prior"] <= high)
        else:
            mask = (frame["prior"] >= low) & (frame["prior"] < high)
        group = frame.loc[mask]
        n = int(len(group))
        mean_prior = float(group["prior"].mean()) if n else float((low + high) / 2.0)
        empirical = float(group["resolved_yes"].mean()) if n else mean_prior
        smoothed = (
            ((empirical * n) + (mean_prior * smoothing_strength))
            / (n + smoothing_strength)
            if n
            else mean_prior
        )
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n": n,
                "mean_prior": _clip(mean_prior),
                "empirical_yes_rate": _clip(empirical),
                "calibrated_probability": _clip(smoothed),
            }
        )
    _make_monotone(rows)
    return rows


def _chronological_market_split(
    frame: pd.DataFrame,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split complete markets chronologically to avoid same-market leakage."""
    if "market_ticker" not in frame.columns:
        split = max(1, int(len(frame) * (1.0 - test_frac)))
        return frame.iloc[:split].copy(), frame.iloc[split:].copy(), "row_chronological"

    market_times = (
        frame.dropna(subset=["market_ticker"])
        .groupby("market_ticker")["timestamp"]
        .max()
        .sort_values()
    )
    if len(market_times) < 2:
        split = max(1, int(len(frame) * (1.0 - test_frac)))
        return frame.iloc[:split].copy(), frame.iloc[split:].copy(), "row_chronological_fallback"

    split_markets = max(1, int(len(market_times) * (1.0 - test_frac)))
    train_tickers = set(market_times.iloc[:split_markets].index)
    train = frame[frame["market_ticker"].isin(train_tickers)].copy()
    test = frame[~frame["market_ticker"].isin(train_tickers)].copy()
    return train, test, "market_disjoint_chronological"


def _make_monotone(rows: list[dict[str, Any]]) -> None:
    """Pool adjacent violators by weighted bin count."""
    blocks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        weight = max(1, int(row["n"]))
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": weight,
                "value": float(row["calibrated_probability"]),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left["weight"] + right["weight"]
            pooled = ((left["value"] * left["weight"]) + (right["value"] * right["weight"])) / weight_sum
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": weight_sum,
                    "value": pooled,
                }
            )
    for block in blocks:
        for idx in range(block["start"], block["end"] + 1):
            rows[idx]["calibrated_probability"] = _clip(block["value"])


def _lookup_probability(
    *,
    prior: float,
    template: str,
    global_bins: list[dict[str, Any]],
    by_template: dict[str, list[dict[str, Any]]],
    min_samples: int,
) -> float:
    template_rows = by_template.get(str(template).lower())
    if template_rows:
        match = _find_bin(template_rows, prior)
        if match and int(match.get("n") or 0) >= min_samples:
            return float(match["calibrated_probability"])
    match = _find_bin(global_bins, prior)
    if match and int(match.get("n") or 0) >= min_samples:
        return float(match["calibrated_probability"])
    return _clip(prior)


def _find_bin(rows: list[dict[str, Any]], prior: float) -> dict[str, Any] | None:
    for row in rows:
        low = float(row["bin_low"])
        high = float(row["bin_high"])
        if low <= prior < high or (prior == 1.0 and high == 1.0):
            return row
    return None


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


def _clip(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


if __name__ == "__main__":
    main()
