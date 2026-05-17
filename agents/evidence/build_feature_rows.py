"""Build numeric LLM evidence feature rows from events and source snippets.

Input snippets JSONL format:
{"market_ticker": "...", "as_of_time": "...", "sources": [{"url": "...", "published_at": "...", "text": "..."}]}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agents.evidence.extractor import extract_evidence_features
from agents.evidence.schemas import EvidenceSource
from agents.order_flow.features import classify_market_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--snippets-jsonl", required=True)
    parser.add_argument("--out", default="data/evidence_feature_rows.csv")
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    snippets = _load_snippets(Path(args.snippets_jsonl))
    rows: list[dict[str, Any]] = []

    for event in events[: args.max_events or None]:
        if not isinstance(event, dict):
            continue
        market_ticker = str(event.get("market_ticker") or event.get("event_ticker") or "")
        snippet_payload = snippets.get(market_ticker)
        if not snippet_payload:
            continue
        as_of_time = str(snippet_payload.get("as_of_time") or event.get("close_time") or "")
        sources = [
            EvidenceSource(**source)
            for source in snippet_payload.get("sources", [])
            if isinstance(source, dict)
        ]
        template = classify_market_template(str(event.get("title") or ""), market_ticker)
        extracted = extract_evidence_features(
            event=event,
            template=template,
            as_of_time=as_of_time,
            sources=sources,
        )
        rows.append(extracted.flat_row())

    _write_csv(Path(args.out), rows)
    print(f"Wrote {len(rows)} evidence feature rows to {args.out}")


def _load_snippets(path: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            ticker = str(payload.get("market_ticker") or payload.get("event_ticker") or "")
            if ticker:
                payloads[ticker] = payload
    return payloads


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

