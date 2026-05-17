"""Collect a pre-resolution live matrix for curated LLM edge features.

This is the dense collection path we use going forward:

1. Snapshot unresolved Kalshi public markets.
2. Build deterministic market-rule features.
3. Ask the LLM/web-search extractor for curated unstructured features.
4. Save one wide row per market, before resolution.

The output is meant to be labeled later and trained with
`agents.evidence.train_live_edge_model`.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents.evidence.edge_feature_schema import (
    all_runtime_feature_names,
    build_market_rule_features,
    default_llm_feature_row,
)
from agents.evidence.llm_edge_extractor import extract_llm_edge_features
from agents.evidence.live_feature_experiment import (
    _build_candidates,
    _fetch_open_markets,
    _select_by_category,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--max-per-template", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--min-volume", type=float, default=0.0)
    parser.add_argument("--single-leg-only", action="store_true", default=True)
    parser.add_argument("--include-multileg", action="store_true", help="Allow multileg/MVE markets.")
    parser.add_argument("--text-filter", default="", help="Case-insensitive substring filter over ticker/title.")
    parser.add_argument("--force-template", default="", help="Override routed template for selected rows.")
    parser.add_argument("--no-llm", action="store_true", help="Only collect market/title rule features.")
    parser.add_argument("--out-dir", default="reports/live_edge_feature_collection")
    args = parser.parse_args()

    now = datetime.now(UTC)
    end = now + timedelta(hours=args.horizon_hours)
    run_dir = _unique_run_dir(Path(args.out_dir), now.strftime("%Y%m%d_%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)

    markets = _fetch_open_markets(args.max_pages)
    candidates = _build_candidates(
        markets=markets,
        now=now,
        end=end,
        min_volume=args.min_volume,
        single_leg_only=(args.single_leg_only and not args.include_multileg),
    )
    if args.text_filter:
        needle = args.text_filter.lower()
        candidates = [
            row
            for row in candidates
            if needle in f"{row.get('ticker')} {row.get('event_ticker')} {row.get('title')}".lower()
        ]
    if args.force_template:
        for row in candidates:
            row["template"] = args.force_template
    selected = _select_by_category(candidates, args.max_per_template, prefer_liquid=True)[: args.max_rows]
    rows, raw_rows = _build_edge_rows(selected, now, use_llm=not args.no_llm)

    events_path = run_dir / "live_edge_events.csv"
    matrix_path = run_dir / "live_edge_feature_matrix.csv"
    raw_path = run_dir / "live_edge_llm_payloads.jsonl"
    _write_csv(events_path, selected)
    _write_csv(matrix_path, rows)
    with raw_path.open("w", encoding="utf-8") as handle:
        for raw in raw_rows:
            handle.write(json.dumps(raw, ensure_ascii=True, default=str))
            handle.write("\n")

    summary = {
        "snapshot_time": now.isoformat(),
        "horizon_end": end.isoformat(),
        "markets_fetched": len(markets),
        "candidates": len(candidates),
        "selected": len(selected),
        "selected_by_template": pd.DataFrame(selected).groupby("template").size().astype(int).to_dict() if selected else {},
        "llm_enabled_for_run": not args.no_llm,
        "matrix": str(matrix_path),
        "events": str(events_path),
        "raw_payloads": str(raw_path),
        "label_command": f".\\venv\\Scripts\\python.exe -m agents.evidence.label_live_edge_matrix --run-dir {run_dir}",
        "train_command_after_labeling": f".\\venv\\Scripts\\python.exe -m agents.evidence.train_live_edge_model --matrix {run_dir / 'live_edge_feature_matrix_labeled.csv'} --artifact agents\\evidence\\artifacts\\live_edge_residual.joblib --report reports\\live_edge_residual_backtest.json",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _build_edge_rows(
    events: list[dict[str, Any]],
    snapshot_time: datetime,
    *,
    use_llm: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    feature_names = all_runtime_feature_names()
    for index, event in enumerate(events, start=1):
        prior = float(event.get("prior") or 0.5)
        row: dict[str, Any] = {
            "snapshot_time": snapshot_time.isoformat(),
            "finish_time": event.get("finish_time"),
            "template": event.get("template"),
            "ticker": event.get("ticker"),
            "event_ticker": event.get("event_ticker"),
            "title": event.get("title"),
            "prior": prior,
            "llm_extraction_status": "disabled",
        }
        for feature in feature_names:
            row[feature] = 0.0

        row.update(build_market_rule_features(event, prior, now=snapshot_time))
        if use_llm:
            llm_features, payload = extract_llm_edge_features(
                event,
                str(event.get("template") or "generic"),
                as_of_time=snapshot_time.isoformat(),
            )
            row.update(llm_features)
            row["llm_extraction_status"] = _status_from_payload(payload)
            raw_rows.append(
                {
                    "row_index": index,
                    "ticker": event.get("ticker"),
                    "template": event.get("template"),
                    "title": event.get("title"),
                    "payload": payload,
                }
            )
        rows.append(row)
    return rows, raw_rows


def _status_from_payload(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return "error"
    if payload.get("_source_validation_error"):
        return "invalid_sources"
    if payload.get("skipped"):
        return "skipped"
    return "ok"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unique_run_dir(root: Path, name: str) -> Path:
    candidate = root / name
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = root / f"{name}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique run dir under {root}")


if __name__ == "__main__":
    main()
