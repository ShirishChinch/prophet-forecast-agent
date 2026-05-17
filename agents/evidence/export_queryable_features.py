"""Export the runtime-queryable unstructured feature catalog."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agents.evidence.queryable_features import QueryableFeature, get_queryable_feature_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="reports/queryable_features")
    parser.add_argument("--formats", nargs="+", default=["csv", "json", "md"], choices=["csv", "json", "md"])
    parser.add_argument("--max-priority", type=int, default=99)
    args = parser.parse_args()

    features = [
        feature
        for feature in get_queryable_feature_catalog()
        if feature.priority <= args.max_priority
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    if "csv" in args.formats:
        path = out_dir / "queryable_unstructured_features.csv"
        _write_csv(path, features)
        written.append(str(path))
    if "json" in args.formats:
        path = out_dir / "queryable_unstructured_features.json"
        path.write_text(json.dumps([feature.to_dict() for feature in features], indent=2, sort_keys=True), encoding="utf-8")
        written.append(str(path))
    if "md" in args.formats:
        path = out_dir / "queryable_unstructured_features.md"
        path.write_text(_markdown(features), encoding="utf-8")
        written.append(str(path))

    print(json.dumps({"n_features": len(features), "written": written}, indent=2))


def _write_csv(path: Path, features: list[QueryableFeature]) -> None:
    fieldnames = [
        "template",
        "name",
        "priority",
        "description",
        "numeric_definition",
        "range_min",
        "range_max",
        "default_value",
        "direction",
        "runtime_queries",
        "preferred_sources",
        "source_freshness",
        "historical_backtest_plan",
        "leakage_controls",
        "expected_coverage",
        "expected_signal",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            row: dict[str, Any] = feature.to_dict()
            for key in ("runtime_queries", "preferred_sources", "leakage_controls"):
                row[key] = " | ".join(row[key])
            writer.writerow(row)


def _markdown(features: list[QueryableFeature]) -> str:
    lines: list[str] = [
        "# Queryable Unstructured Features",
        "",
        "These are not LLM probability guesses. They are numeric facts/signals the runtime can query, extract, and later backtest as residual features against the Kalshi prior.",
        "",
    ]
    current_template = ""
    for feature in features:
        if feature.template != current_template:
            current_template = feature.template
            lines.extend(["", f"## {current_template}", ""])
        lines.extend(
            [
                f"### {feature.priority}. `{feature.name}`",
                "",
                f"- Description: {feature.description}",
                f"- Numeric definition: {feature.numeric_definition}",
                f"- Range/default: [{feature.range_min}, {feature.range_max}], default {feature.default_value}",
                f"- Direction: {feature.direction}",
                f"- Query examples: {'; '.join(feature.runtime_queries)}",
                f"- Preferred sources: {'; '.join(feature.preferred_sources)}",
                f"- Freshness: {feature.source_freshness}",
                f"- Backtest plan: {feature.historical_backtest_plan}",
                f"- Leakage controls: {'; '.join(feature.leakage_controls)}",
                f"- Expected coverage/signal: {feature.expected_coverage} / {feature.expected_signal}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
