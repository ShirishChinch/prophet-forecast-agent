"""Extract live queryable feature values with OpenAI web search.

The LLM is not allowed to forecast probability. It fills numeric feature
contracts from timestamped public sources so we can later test correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from agents.evidence.queryable_features import get_queryable_feature_catalog


SYSTEM_PROMPT = """\
You extract numeric public-evidence features for prediction-market research.

Hard rules:
- Do not output a probability forecast.
- Do not recommend a bet.
- Do not use final results unless the market has already finished before the
  supplied snapshot time.
- Prefer public sources that are current at or before snapshot_time.
- If a feature cannot be measured from public sources, return its default value.
- Return JSON only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--templates", nargs="*", default=["sports", "generic", "crypto_price", "weather", "macro"])
    parser.add_argument("--max-per-template", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    run_dir = Path(args.run_dir)
    events = pd.read_csv(run_dir / "live_events.csv")
    selected = _select_events(events, set(args.templates), args.max_per_template)
    out_path = Path(args.out) if args.out else run_dir / "live_llm_feature_rows.csv"
    raw_path = out_path.with_suffix(".raw.jsonl")
    done = _completed_keys(raw_path)
    catalog = _catalog_by_template()

    rows: list[dict[str, Any]] = []
    if out_path.exists() and out_path.stat().st_size > 0:
        rows.extend(pd.read_csv(out_path).to_dict(orient="records"))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as raw_handle:
        for _, event in selected.iterrows():
            key = str(event["ticker"])
            if key in done:
                continue
            template = str(event["template"])
            feature_specs = catalog.get(template, catalog.get("generic", []))
            payload = _extract_features(event.to_dict(), feature_specs, args.model)
            row = _row_from_payload(event.to_dict(), feature_specs, payload)
            rows.append(row)
            raw_handle.write(json.dumps({"event": event.to_dict(), "payload": payload}, ensure_ascii=True, default=str))
            raw_handle.write("\n")
            _write_csv(out_path, rows)

    _write_csv(out_path, rows)
    print(json.dumps({"n_rows": len(rows), "out": str(out_path), "raw": str(raw_path)}, indent=2))


def _select_events(events: pd.DataFrame, templates: set[str], max_per_template: int) -> pd.DataFrame:
    frame = events[events["template"].astype(str).isin(templates)].copy()
    if frame.empty:
        return frame
    frame["volume_fp"] = pd.to_numeric(frame.get("volume_fp", 0.0), errors="coerce").fillna(0.0)
    frame["open_interest_fp"] = pd.to_numeric(frame.get("open_interest_fp", 0.0), errors="coerce").fillna(0.0)
    frame["spread"] = pd.to_numeric(frame.get("spread", 1.0), errors="coerce").fillna(1.0)
    frame["score"] = frame["volume_fp"] + frame["open_interest_fp"] - 1000.0 * frame["spread"]
    return (
        frame.sort_values("score", ascending=False)
        .groupby("template", group_keys=False)
        .head(max_per_template)
        .drop(columns=["score"])
        .reset_index(drop=True)
    )


def _catalog_by_template() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for feature in get_queryable_feature_catalog():
        if feature.priority <= 4:
            grouped.setdefault(feature.template, []).append(feature.to_dict())
    return grouped


def _extract_features(
    event: dict[str, Any],
    feature_specs: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = model or os.environ.get("EVIDENCE_EXTRACTION_MODEL") or os.environ.get("FORECAST_MODEL") or "gpt-4o-mini"
    prompt = {
        "task": "Fill numeric feature values for this event. Do not forecast probability.",
        "event": {
            "ticker": event.get("ticker"),
            "title": event.get("title"),
            "template": event.get("template"),
            "snapshot_time": event.get("snapshot_time"),
            "finish_time": event.get("finish_time"),
            "market_prior": event.get("prior"),
            "yes_sub_title": event.get("yes_sub_title"),
            "no_sub_title": event.get("no_sub_title"),
        },
        "feature_specs": feature_specs,
        "required_json_shape": {
            "features": {"feature_name": 0.0},
            "source_urls": ["..."],
            "source_timestamps": ["YYYY-MM-DDTHH:MM:SSZ or unknown"],
            "extraction_confidence": 0.0,
            "temporal_leakage_risk": 0.0,
            "short_rationale": "one sentence, no probability",
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
    except Exception as exc:
        return {
            "features": {},
            "source_urls": [],
            "source_timestamps": [],
            "extraction_confidence": 0.0,
            "temporal_leakage_risk": 1.0,
            "short_rationale": f"Extraction failed: {type(exc).__name__}",
            "error": str(exc),
        }
    return json.loads(_extract_json(text))


def _row_from_payload(
    event: dict[str, Any],
    feature_specs: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    extracted = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    row: dict[str, Any] = {
        "snapshot_time": event.get("snapshot_time"),
        "finish_time": event.get("finish_time"),
        "template": event.get("template"),
        "ticker": event.get("ticker"),
        "event_ticker": event.get("event_ticker"),
        "title": event.get("title"),
        "prior": event.get("prior"),
        "source_urls": "|".join(str(url) for url in payload.get("source_urls") or []),
        "source_timestamps": "|".join(str(ts) for ts in payload.get("source_timestamps") or []),
        "extraction_confidence": _unit(payload.get("extraction_confidence")),
        "temporal_leakage_risk": _unit(payload.get("temporal_leakage_risk"), default=1.0),
        "short_rationale": str(payload.get("short_rationale") or ""),
    }
    for spec in feature_specs:
        name = str(spec["name"])
        row[name] = _clamp(
            extracted.get(name, spec.get("default_value", 0.0)),
            float(spec["range_min"]),
            float(spec["range_max"]),
            float(spec.get("default_value", 0.0)),
        )
    return row


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
        ticker = ((item.get("event") or {}).get("ticker"))
        if ticker:
            keys.add(str(ticker))
    return keys


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


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _clamp(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, number))


if __name__ == "__main__":
    main()
