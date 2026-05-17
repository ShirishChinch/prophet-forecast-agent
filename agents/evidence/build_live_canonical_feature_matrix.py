"""Build a canonical feature matrix from live LLM-discovered features.

The LLM is allowed to discover messy feature names. This script normalizes those
into a small reusable matrix, adds deterministic market/title features, and
keeps missing LLM coverage explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


SPORTS_FEATURES = [
    "sports__llm_team_recent_form_edge",
    "sports__llm_player_recent_form_edge",
    "sports__llm_hitter_hit_edge",
    "sports__llm_pitcher_strikeout_edge",
    "sports__llm_total_runs_edge",
    "sports__llm_weather_edge",
    "sports__llm_fight_finish_edge",
    "sports__llm_injury_lineup_edge",
    "sports__llm_feature_coverage",
    "sports__llm_mean_confidence",
    "sports__llm_nonzero_count",
    "sports__title_leg_count",
    "sports__title_has_player_prop",
    "sports__title_has_team_win",
    "sports__title_has_run_spread",
    "sports__title_has_total_runs",
    "sports__title_has_fight_finish",
    "sports__title_yes_leg_share",
    "sports__title_no_leg_share",
    "sports__market_spread",
    "sports__market_volume_log",
    "sports__hours_to_finish",
]

CRYPTO_FEATURES = [
    "crypto__llm_sentiment_edge",
    "crypto__llm_feature_coverage",
    "crypto__llm_mean_confidence",
    "crypto__title_has_target_price",
    "crypto__title_leg_count",
    "crypto__market_spread",
    "crypto__market_volume_log",
    "crypto__hours_to_finish",
]

GENERIC_FEATURES = [
    "generic__llm_official_confirmation_edge",
    "generic__llm_news_consensus_edge",
    "generic__llm_player_prop_edge",
    "generic__llm_count_threshold_edge",
    "generic__llm_feature_coverage",
    "generic__llm_mean_confidence",
    "generic__title_leg_count",
    "generic__title_yes_leg_share",
    "generic__title_no_leg_share",
    "generic__market_spread",
    "generic__market_volume_log",
    "generic__hours_to_finish",
]

MACRO_FEATURES = [
    "macro__llm_consensus_edge",
    "macro__llm_nowcast_edge",
    "macro__llm_rates_market_edge",
    "macro__llm_feature_coverage",
    "macro__llm_mean_confidence",
    "macro__title_leg_count",
    "macro__market_spread",
    "macro__market_volume_log",
    "macro__hours_to_finish",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    events = pd.read_csv(run_dir / "live_events.csv")
    discovered_path = run_dir / "live_llm_discovered_feature_vectors.csv"
    discovered = pd.read_csv(discovered_path) if discovered_path.exists() and discovered_path.stat().st_size else pd.DataFrame()

    selected = _select_events(events, args.max_rows)
    rows = [_build_row(event, discovered) for event in selected.to_dict(orient="records")]

    out_path = Path(args.out) if args.out else run_dir / "canonical_feature_matrix_1000.csv"
    _write_csv(out_path, rows)
    summary = _summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "columns": len(rows[0]) if rows else 0, "out": str(out_path), "summary": str(summary_path)}, indent=2))


def _select_events(events: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    frame = events.copy()
    frame["volume_fp"] = pd.to_numeric(frame.get("volume_fp", 0.0), errors="coerce").fillna(0.0)
    frame["open_interest_fp"] = pd.to_numeric(frame.get("open_interest_fp", 0.0), errors="coerce").fillna(0.0)
    frame["spread"] = pd.to_numeric(frame.get("spread", 1.0), errors="coerce").fillna(1.0)
    frame["title_leg_count"] = frame["title"].astype(str).map(_leg_count)
    frame["quality_score"] = (
        frame["volume_fp"]
        + frame["open_interest_fp"]
        - 400.0 * frame["spread"]
        - 10.0 * frame["title_leg_count"].clip(lower=1)
    )
    return frame.sort_values("quality_score", ascending=False).head(max_rows).reset_index(drop=True)


def _build_row(event: dict[str, Any], discovered: pd.DataFrame) -> dict[str, Any]:
    template = str(event.get("template") or "generic")
    row = {
        "snapshot_time": event.get("snapshot_time"),
        "finish_time": event.get("finish_time"),
        "template": template,
        "ticker": event.get("ticker"),
        "event_ticker": event.get("event_ticker"),
        "title": event.get("title"),
        "prior": _to_float(event.get("prior"), 0.5),
    }
    for feature in SPORTS_FEATURES + CRYPTO_FEATURES + GENERIC_FEATURES + MACRO_FEATURES:
        row[feature] = 0.0

    title = str(event.get("title") or "")
    prefix = template if template in {"sports", "crypto_price", "generic", "macro"} else "generic"
    if prefix == "crypto_price":
        prefix = "crypto"
    _add_title_features(row, prefix, title, event)
    _add_llm_features(row, template, str(event.get("ticker") or ""), discovered)
    return row


def _add_title_features(row: dict[str, Any], prefix: str, title: str, event: dict[str, Any]) -> None:
    leg_count = _leg_count(title)
    yes_count = len(re.findall(r"\byes\b", title.lower()))
    no_count = len(re.findall(r"\bno\b", title.lower()))
    total_side_count = max(1, yes_count + no_count)
    common = {
        f"{prefix}__title_leg_count": float(leg_count),
        f"{prefix}__market_spread": _to_float(event.get("spread"), 1.0),
        f"{prefix}__market_volume_log": _log1p(_to_float(event.get("volume_fp"), 0.0)),
        f"{prefix}__hours_to_finish": _to_float(event.get("hours_to_finish"), 0.0),
    }
    for key, value in common.items():
        if key in row:
            row[key] = value
    for key, value in {
        f"{prefix}__title_yes_leg_share": yes_count / total_side_count,
        f"{prefix}__title_no_leg_share": no_count / total_side_count,
    }.items():
        if key in row:
            row[key] = value

    lower = title.lower()
    if prefix == "sports":
        row["sports__title_has_player_prop"] = float(bool(re.search(r":\s*\d\+|strikeout|hits?|points?|rebounds?|assists?", lower)))
        row["sports__title_has_team_win"] = float(" win" in lower or "yes " in lower)
        row["sports__title_has_run_spread"] = float("wins by over" in lower)
        row["sports__title_has_total_runs"] = float("runs scored" in lower or "over " in lower)
        row["sports__title_has_fight_finish"] = float("fight ends" in lower or "ko/tko" in lower or "decision" in lower)
    elif prefix == "crypto":
        row["crypto__title_has_target_price"] = float("target price" in lower or "$" in lower)


def _add_llm_features(row: dict[str, Any], template: str, ticker: str, discovered: pd.DataFrame) -> None:
    if discovered.empty or "ticker" not in discovered.columns:
        return
    group = discovered[discovered["ticker"].astype(str).eq(ticker)].copy()
    if group.empty:
        return
    group["feature_value"] = pd.to_numeric(group.get("feature_value", 0.0), errors="coerce").fillna(0.0)
    group["confidence"] = pd.to_numeric(group.get("confidence", 0.0), errors="coerce").fillna(0.0)
    group["temporal_leakage_risk"] = pd.to_numeric(group.get("temporal_leakage_risk", 1.0), errors="coerce").fillna(1.0)
    group["weighted_value"] = group["feature_value"] * group["confidence"] * (1.0 - group["temporal_leakage_risk"])

    canonical_values: dict[str, list[float]] = {}
    for _, feature in group.iterrows():
        canonical = _canonical_feature(template, str(feature.get("feature_name") or ""))
        if canonical is None or canonical not in row:
            continue
        canonical_values.setdefault(canonical, []).append(float(feature["weighted_value"]))
    for canonical, values in canonical_values.items():
        row[canonical] = _mean_clamped(values)

    prefix = template if template in {"sports", "generic", "macro"} else "crypto" if template == "crypto_price" else "generic"
    coverage_key = f"{prefix}__llm_feature_coverage"
    confidence_key = f"{prefix}__llm_mean_confidence"
    nonzero_key = f"{prefix}__llm_nonzero_count"
    if coverage_key in row:
        row[coverage_key] = min(1.0, len(group) / 4.0)
    if confidence_key in row:
        row[confidence_key] = float(group["confidence"].mean())
    if nonzero_key in row:
        row[nonzero_key] = float((group["feature_value"].abs() > 1e-12).sum())


def _canonical_feature(template: str, name: str) -> str | None:
    text = name.lower()
    if template == "crypto_price":
        if any(token in text for token in ("sentiment", "news", "market")):
            return "crypto__llm_sentiment_edge"
        return None
    if template == "macro":
        if "consensus" in text or "expert" in text:
            return "macro__llm_consensus_edge"
        if "nowcast" in text:
            return "macro__llm_nowcast_edge"
        if "rate" in text or "fed" in text or "ois" in text:
            return "macro__llm_rates_market_edge"
        return None
    if template == "generic":
        if "official" in text or "confirm" in text:
            return "generic__llm_official_confirmation_edge"
        if "news" in text or "consensus" in text:
            return "generic__llm_news_consensus_edge"
        if "player" in text or "hit" in text or "strikeout" in text or "goal" in text:
            return "generic__llm_player_prop_edge"
        if "count" in text or "threshold" in text or "over" in text:
            return "generic__llm_count_threshold_edge"
        return None
    # Sports.
    if "weather" in text:
        return "sports__llm_weather_edge"
    if "injur" in text or "lineup" in text:
        return "sports__llm_injury_lineup_edge"
    if "fight" in text or "decision" in text or "ko" in text or "tko" in text:
        return "sports__llm_fight_finish_edge"
    if "strikeout" in text or "pitcher" in text or "_k" in text:
        return "sports__llm_pitcher_strikeout_edge"
    if "hit" in text or "hitter" in text or "batting" in text:
        return "sports__llm_hitter_hit_edge"
    if "run" in text or "score" in text or "total" in text or "over_" in text:
        return "sports__llm_total_runs_edge"
    if "player" in text or "performance" in text or "trend" in text:
        return "sports__llm_player_recent_form_edge"
    if "team" in text or "cincinnati" in text or "cleveland" in text or "milwaukee" in text or "atlanta" in text:
        return "sports__llm_team_recent_form_edge"
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    feature_cols = [column for column in frame.columns if "__" in column]
    feature_summary = []
    for column in feature_cols:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        feature_summary.append(
            {
                "feature": column,
                "nonzero_share": float((values.abs() > 1e-12).mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )
    return {
        "n_rows": len(rows),
        "n_features": len(feature_cols),
        "rows_by_template": frame.groupby("template").size().astype(int).to_dict(),
        "feature_summary": sorted(feature_summary, key=lambda row: row["nonzero_share"], reverse=True),
    }


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


def _leg_count(title: str) -> int:
    return max(1, str(title).count(",") + 1)


def _mean_clamped(values: list[float]) -> float:
    if not values:
        return 0.0
    value = sum(values) / len(values)
    return max(-1.0, min(1.0, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log1p(value: float) -> float:
    import math

    return math.log1p(max(0.0, value))


if __name__ == "__main__":
    main()
