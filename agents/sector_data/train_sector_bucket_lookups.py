"""Train raw sector empirical odds lookup tables.

Each cell estimates the historical YES resolution rate for:

    sector/subtype/day_before_close/5-cent odds bucket

There is no smoothing. Runtime should only use cells with n > min_samples and
otherwise fall back to the current market odds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/sector_15d")
DEFAULT_ARTIFACT = Path("agents/sector_data/artifacts/sector_bucket_lookup.json")
DEFAULT_REPORT = Path("reports/sector_bucket_lookup_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()

    observed = load_observed(Path(args.input_dir))
    if observed.empty:
        raise SystemExit(f"No observed sector CSVs found under {args.input_dir}")

    rows = build_lookup_rows(observed)
    artifact = {
        "version": 1,
        "table_type": "sector_bucket_lookup",
        "min_samples_exclusive": int(args.min_samples),
        "bucket_width": 5,
        "fallback_order": [
            "sector|subtype|day_before_close|bucket",
            "sector|subtype|ALL|bucket",
            "sector|ALL|day_before_close|bucket",
            "sector|ALL|ALL|bucket",
            "ALL|ALL|day_before_close|bucket",
            "ALL|ALL|ALL|bucket",
        ],
        "tables": _rows_to_tables(rows),
        "source_input_dir": args.input_dir,
        "notes": [
            "Raw empirical average resolved_yes by 5-cent odds bucket.",
            "No smoothing, regression, LLM, or synthetic filling.",
            "Runtime should apply a cell only when n > min_samples_exclusive.",
        ],
    }

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False)

    summary = {
        "input_dir": args.input_dir,
        "artifact": args.artifact,
        "report": args.report,
        "observed_rows": int(len(observed)),
        "observed_markets": int(observed["market_ticker"].nunique()),
        "sectors": observed.groupby("sector")["market_ticker"].nunique().astype(int).to_dict(),
        "table_rows": int(len(rows)),
        "cells_gt_min_samples": int(sum(1 for row in rows if row["n"] > args.min_samples)),
        "min_samples_exclusive": int(args.min_samples),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_observed(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("*/*_15d_observed_snapshots.csv")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    observed = pd.concat(frames, ignore_index=True)
    needed = {"sector", "subtype", "day_before_close", "p_yes", "odds_cent", "resolved_yes", "market_ticker"}
    missing = needed.difference(observed.columns)
    if missing:
        raise ValueError(f"Observed CSVs missing required columns: {sorted(missing)}")
    observed = observed.dropna(subset=["sector", "day_before_close", "p_yes", "resolved_yes"]).copy()
    observed["sector"] = observed["sector"].fillna("generic").astype(str).str.lower()
    observed["subtype"] = observed["subtype"].fillna("all").astype(str).str.lower().replace("", "all")
    observed["day_before_close"] = observed["day_before_close"].astype(int)
    observed["p_yes"] = observed["p_yes"].astype(float).clip(0.01, 0.99)
    observed["odds_cent"] = observed["p_yes"].map(lambda value: int(round(float(value) * 100.0)))
    observed["resolved_yes"] = observed["resolved_yes"].astype(float)
    return observed


def build_lookup_rows(observed: pd.DataFrame) -> list[dict[str, Any]]:
    frame = observed.copy()
    frame["bucket_low"], frame["bucket_high"] = zip(*frame["odds_cent"].map(_bucket_bounds))
    table_specs: list[tuple[str, pd.DataFrame]] = [("ALL|ALL|ALL", frame)]

    for day, day_group in sorted(frame.groupby("day_before_close"), key=lambda item: int(item[0])):
        table_specs.append((f"ALL|ALL|{int(day)}", day_group))

    for sector, sector_group in sorted(frame.groupby("sector"), key=lambda item: str(item[0])):
        table_specs.append((f"{sector}|ALL|ALL", sector_group))
        for day, day_group in sorted(sector_group.groupby("day_before_close"), key=lambda item: int(item[0])):
            table_specs.append((f"{sector}|ALL|{int(day)}", day_group))
        for subtype, subtype_group in sorted(sector_group.groupby("subtype"), key=lambda item: str(item[0])):
            table_specs.append((f"{sector}|{subtype}|ALL", subtype_group))
            for day, day_group in sorted(subtype_group.groupby("day_before_close"), key=lambda item: int(item[0])):
                table_specs.append((f"{sector}|{subtype}|{int(day)}", day_group))

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
                    "sector": lookup_key.split("|")[0],
                    "subtype": lookup_key.split("|")[1],
                    "day_before_close": lookup_key.split("|")[2],
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
