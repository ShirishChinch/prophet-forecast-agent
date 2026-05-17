"""Let the LLM discover and value reusable unstructured features for live events.

This is intentionally separate from the fixed feature catalog. The LLM proposes
features that are:

- unstructured/public-source based,
- reusable across similar events,
- numeric,
- queryable at runtime,
- not probability forecasts.

The output can be labeled after resolution and correlated with
`resolved_yes - kalshi_prior`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_PROMPT = """\
You design and extract reusable unstructured features for prediction-market
research.

Hard rules:
- Do not forecast probability.
- Do not recommend a bet.
- Use only public unstructured or semi-structured information that can be
  queried at runtime: news, official pages, live score pages, injury reports,
  lineup reports, weather discussions, expert commentary, social/reddit-like
  discussion, market descriptions, source timestamps, public market pages.
- Do not use Kalshi price/order-flow as a feature here.
- Do not use final results after snapshot_time.
- Features must be reusable across similar events, not one-off narrative facts.
- Each feature must be numeric and signed so positive supports YES.
- If current value cannot be measured, use 0 and explain why.
- Return JSON only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-per-template", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    run_dir = Path(args.run_dir)
    events = pd.read_csv(run_dir / "live_events.csv")
    selected = _select_events(events, args.max_rows, args.max_per_template)
    out_path = Path(args.out) if args.out else run_dir / "live_llm_discovered_feature_vectors.csv"
    raw_path = out_path.with_suffix(".raw.jsonl")

    completed = _completed_keys(raw_path)
    rows: list[dict[str, Any]] = []
    if out_path.exists() and out_path.stat().st_size > 0:
        rows.extend(pd.read_csv(out_path).to_dict(orient="records"))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as raw_handle:
        remaining = [row for row in selected.to_dict(orient="records") if str(row["ticker"]) not in completed]
        for batch in _chunks(remaining, args.batch_size):
            payload = _extract_batch(batch, args.model)
            batch_rows = _rows_from_payload(batch, payload)
            rows.extend(batch_rows)
            raw_handle.write(json.dumps({"events": batch, "payload": payload}, ensure_ascii=True, default=str))
            raw_handle.write("\n")
            _write_csv(out_path, rows)

    _write_csv(out_path, rows)
    feature_summary = _summarize_features(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(feature_summary, indent=2), encoding="utf-8")
    print(json.dumps({"n_rows": len(rows), "out": str(out_path), "summary": str(summary_path)}, indent=2))


def _select_events(events: pd.DataFrame, max_rows: int, max_per_template: int) -> pd.DataFrame:
    frame = events.copy()
    frame["volume_fp"] = pd.to_numeric(frame.get("volume_fp", 0.0), errors="coerce").fillna(0.0)
    frame["open_interest_fp"] = pd.to_numeric(frame.get("open_interest_fp", 0.0), errors="coerce").fillna(0.0)
    frame["spread"] = pd.to_numeric(frame.get("spread", 1.0), errors="coerce").fillna(1.0)
    frame["quality_score"] = (
        frame["volume_fp"]
        + frame["open_interest_fp"]
        - 500.0 * frame["spread"]
        - 25.0 * frame["title"].astype(str).str.count(",")
    )
    selected = (
        frame.sort_values("quality_score", ascending=False)
        .groupby("template", group_keys=False)
        .head(max_per_template)
        .sort_values("quality_score", ascending=False)
        .head(max_rows)
        .drop(columns=["quality_score"])
    )
    return selected.reset_index(drop=True)


def _extract_batch(events: list[dict[str, Any]], model: str | None) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = model or os.environ.get("EVIDENCE_DISCOVERY_MODEL") or os.environ.get("FORECAST_MODEL") or "gpt-4o-mini"
    prompt = {
        "task": "For each event, discover reusable unstructured numeric features and give current values.",
        "events": [_compact_event(event) for event in events],
        "feature_requirements": {
            "numeric_range": "Prefer [-1, 1], where positive supports YES. Use [0, 1] only for confidence/coverage features.",
            "reusable": "Feature must apply across many similar markets.",
            "queryable": "Must be discoverable from public web/search/source pages at runtime.",
            "not_allowed": ["Kalshi prior", "Kalshi order flow", "final result", "probability forecast"],
        },
        "required_json_shape": {
            "events": [
                {
                    "ticker": "ticker",
                    "features": [
                        {
                            "name": "template__feature_name",
                            "value": 0.0,
                            "range_min": -1.0,
                            "range_max": 1.0,
                            "direction": "positive supports YES",
                            "source_query": "search query that can reproduce it",
                            "source_urls": ["..."],
                            "source_timestamps": ["YYYY-MM-DDTHH:MM:SSZ|unknown"],
                            "why_reusable": "short reason",
                            "measurement_method": "how value was computed",
                            "confidence": 0.0,
                            "temporal_leakage_risk": 0.0
                        }
                    ],
                    "short_rationale": "one sentence; no probability"
                }
            ]
        },
    }
    try:
        response = client.responses.create(
            model=model_name,
            tools=[{"type": "web_search_preview"}],
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
        )
        text = getattr(response, "output_text", "") or "{}"
        return json.loads(_extract_json(text))
    except Exception as exc:
        return {"events": [], "error": f"{type(exc).__name__}: {exc}"}


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": event.get("ticker"),
        "template": event.get("template"),
        "title": event.get("title"),
        "snapshot_time": event.get("snapshot_time"),
        "finish_time": event.get("finish_time"),
        "market_prior": event.get("prior"),
        "yes_sub_title": event.get("yes_sub_title"),
        "no_sub_title": event.get("no_sub_title"),
    }


def _rows_from_payload(events: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_ticker = {str(item.get("ticker")): item for item in payload.get("events") or [] if isinstance(item, dict)}
    rows: list[dict[str, Any]] = []
    for event in events:
        ticker = str(event["ticker"])
        item = by_ticker.get(ticker, {})
        base = {
            "snapshot_time": event.get("snapshot_time"),
            "finish_time": event.get("finish_time"),
            "template": event.get("template"),
            "ticker": ticker,
            "event_ticker": event.get("event_ticker"),
            "title": event.get("title"),
            "prior": event.get("prior"),
            "short_rationale": item.get("short_rationale", ""),
        }
        features = item.get("features")
        if not isinstance(features, list) or not features:
            rows.append({**base, "feature_name": "no_feature_found", "feature_value": 0.0, "confidence": 0.0, "temporal_leakage_risk": 1.0})
            continue
        for feature in features:
            if not isinstance(feature, dict):
                continue
            rows.append(
                {
                    **base,
                    "feature_name": _safe_feature_name(str(feature.get("name") or "unknown__feature")),
                    "feature_value": _to_float(feature.get("value")),
                    "range_min": _to_float(feature.get("range_min"), -1.0),
                    "range_max": _to_float(feature.get("range_max"), 1.0),
                    "direction": feature.get("direction"),
                    "source_query": feature.get("source_query"),
                    "source_urls": "|".join(str(url) for url in feature.get("source_urls") or []),
                    "source_timestamps": "|".join(str(ts) for ts in feature.get("source_timestamps") or []),
                    "why_reusable": feature.get("why_reusable"),
                    "measurement_method": feature.get("measurement_method"),
                    "confidence": _unit(feature.get("confidence")),
                    "temporal_leakage_risk": _unit(feature.get("temporal_leakage_risk"), default=1.0),
                }
            )
    return rows


def _summarize_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0, "features": []}
    frame = pd.DataFrame(rows)
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce").fillna(0.0)
    frame["confidence"] = pd.to_numeric(frame.get("confidence", 0.0), errors="coerce").fillna(0.0)
    grouped = []
    for (template, feature), group in frame.groupby(["template", "feature_name"]):
        values = group["feature_value"]
        grouped.append(
            {
                "template": template,
                "feature_name": feature,
                "n": int(len(group)),
                "nonzero_share": float((values.abs() > 1e-12).mean()),
                "mean_value": float(values.mean()),
                "mean_confidence": float(group["confidence"].mean()),
            }
        )
    return {
        "n_rows": len(rows),
        "n_tickers": int(frame["ticker"].nunique()),
        "features": sorted(grouped, key=lambda row: (row["template"], -row["nonzero_share"], row["feature_name"])),
    }


def _completed_keys(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    keys: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        for event in item.get("events") or []:
            ticker = event.get("ticker")
            if ticker:
                keys.add(str(ticker))
    return keys


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


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


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start:end + 1]
    return "{}"


def _safe_feature_name(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(char for char in cleaned if char.isalnum() or char == "_") or "unknown__feature"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unit(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _to_float(value, default)))


if __name__ == "__main__":
    main()
