"""Train raw tennis empirical odds lookup tables.

This is deliberately simple:

    bucket = 1-5, 6-10, ..., 96-99 odds cents
    p_lookup = mean(resolved_yes) for that bucket

No smoothing, no regression, no LLM. Runtime should use a cell only when
`n > min_samples`; otherwise it falls back to the current market odds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/tennis_15d/tennis_15d_observed_snapshots.csv")
DEFAULT_ARTIFACT = Path("agents/tennis_data/artifacts/tennis_lookup.json")
DEFAULT_REPORT = Path("reports/tennis_lookup_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()

    observed = pd.read_csv(args.input)
    table_rows = build_lookup_rows(observed)
    artifact = {
        "version": 1,
        "table_type": "tennis_bucket_lookup",
        "min_samples_exclusive": int(args.min_samples),
        "bucket_width": 5,
        "fallback_order": [
            "series|day_before_close|bucket",
            "series|ALL|bucket",
            "ALL|day_before_close|bucket",
            "ALL|ALL|bucket",
        ],
        "tables": _rows_to_tables(table_rows),
        "source_csv": args.input,
        "notes": [
            "Raw empirical average resolved_yes by 5-cent odds bucket.",
            "No smoothing, regression, LLM, or synthetic missing-price filling.",
            "Runtime applies a cell only when n > min_samples_exclusive.",
        ],
    }

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(report_path, index=False)

    summary = {
        "input": args.input,
        "artifact": args.artifact,
        "report": args.report,
        "rows": len(observed),
        "table_rows": len(table_rows),
        "cells_gt_min_samples": sum(1 for row in table_rows if row["n"] > args.min_samples),
        "min_samples_exclusive": args.min_samples,
    }
    print(json.dumps(summary, indent=2))


def build_lookup_rows(observed: pd.DataFrame) -> list[dict[str, Any]]:
    frame = observed.copy()
    frame = frame[frame["p_yes"].notna()].copy()
    frame["series_key"] = frame["series"].fillna("").astype(str).str.lower().replace("", "unknown")
    frame["bucket_low"], frame["bucket_high"] = zip(*frame["odds_cent"].map(_bucket_bounds))
    table_specs: list[tuple[str, pd.DataFrame]] = [("ALL|ALL", frame)]

    for day, group in sorted(frame.groupby("day_before_close"), key=lambda item: int(item[0])):
        table_specs.append((f"ALL|{int(day)}", group))
    for series, series_group in sorted(frame.groupby("series_key"), key=lambda item: str(item[0])):
        table_specs.append((f"{series}|ALL", series_group))
        for day, group in sorted(series_group.groupby("day_before_close"), key=lambda item: int(item[0])):
            table_specs.append((f"{series}|{int(day)}", group))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for lookup_key, group in table_specs:
        for (low, high), bucket_group in sorted(group.groupby(["bucket_low", "bucket_high"])):
            key = (lookup_key, int(low), int(high))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "lookup_key": lookup_key,
                    "bucket_low": int(low),
                    "bucket_high": int(high),
                    "n": int(len(bucket_group)),
                    "empirical_p_yes": float(bucket_group["resolved_yes"].mean()),
                    "mean_market_odds": float(bucket_group["p_yes"].mean()),
                }
            )
    return rows


def _rows_to_tables(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lookup_key = str(row["lookup_key"])
        tables.setdefault(lookup_key, []).append(
            {
                "bucket_low": row["bucket_low"],
                "bucket_high": row["bucket_high"],
                "n": row["n"],
                "empirical_p_yes": row["empirical_p_yes"],
                "mean_market_odds": row["mean_market_odds"],
            }
        )
    return tables


def _bucket_bounds(odds_cent: Any) -> tuple[int, int]:
    cent = int(round(float(odds_cent)))
    cent = max(1, min(99, cent))
    low = ((cent - 1) // 5) * 5 + 1
    high = min(99, low + 4)
    return low, high


if __name__ == "__main__":
    main()
