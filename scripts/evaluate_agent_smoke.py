"""Smoke-test a Prophet forecast endpoint.

This is intentionally small and dependency-light so organizers can run it in a
standard Python environment:

    python scripts/evaluate_agent_smoke.py --url http://127.0.0.1:8000/predict
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAMPLE_EVENTS: list[dict[str, Any]] = [
    {
        "event_ticker": "smoke-binary",
        "market_ticker": "smoke-binary",
        "title": "Will BTC exceed $100,000 by June 30?",
        "category": "Financials",
        "rules": "Resolves YES if BTC exceeds the threshold.",
        "close_time": "2026-06-30T23:59:59Z",
        "outcomes": ["Yes", "No"],
        "best_bid": 0.52,
        "best_ask": 0.56,
    },
    {
        "event_ticker": "smoke-multi",
        "market_ticker": "smoke-multi",
        "title": "Who will win the NBA championship 2026?",
        "category": "Sports",
        "rules": "Resolves to the listed team that wins the championship.",
        "close_time": "2026-06-30T23:59:59Z",
        "outcomes": [
            "Boston Celtics",
            "Denver Nuggets",
            "New York Knicks",
            "Oklahoma City Thunder",
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a Prophet forecast endpoint.")
    parser.add_argument("--url", required=True, help="Forecast endpoint URL, e.g. https://host/predict")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    for event in SAMPLE_EVENTS:
        payload = _post_json(args.url, event, timeout=args.timeout)
        _validate_response(event, payload)
        compact = {
            "market_ticker": event["market_ticker"],
            "probabilities": payload["probabilities"],
        }
        print(json.dumps(compact, indent=2))
    print("Smoke test passed.")
    return 0


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "prophet-smoke-test/1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Response was not valid JSON: {body[:500]}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("Response JSON must be an object")
    return parsed


def _validate_response(event: dict[str, Any], payload: dict[str, Any]) -> None:
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise SystemExit("Response must include a non-empty probabilities list")

    expected_markets = [str(outcome) for outcome in event.get("outcomes") or []]
    received_markets = []
    total = 0.0
    for item in probabilities:
        if not isinstance(item, dict):
            raise SystemExit("Each probability entry must be an object")
        market = str(item.get("market") or "")
        if not market:
            raise SystemExit("Each probability entry must include market")
        try:
            probability = float(item["probability"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("Each probability entry must include numeric probability") from exc
        if probability < 0.0 or probability > 1.0:
            raise SystemExit(f"Probability for {market} is outside [0, 1]: {probability}")
        received_markets.append(market)
        total += probability

    if expected_markets and received_markets != expected_markets:
        raise SystemExit(
            "Markets must exactly match event outcomes in order. "
            f"Expected {expected_markets}, got {received_markets}"
        )
    if abs(total - 1.0) > 0.02:
        raise SystemExit(f"Probabilities should sum close to 1.0; got {total:.4f}")


if __name__ == "__main__":
    sys.exit(main())

