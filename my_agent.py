"""Local Prophet Arena forecast agent entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents.router_forecast_agent import RouterForecastAgent


def predict(event: dict[str, Any]) -> dict[str, Any]:
    """Forecast a single event using the router-based modular agent."""
    _load_dotenv()
    agent = RouterForecastAgent()
    return agent.predict(event)


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overwriting env vars."""
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _sample_events() -> list[dict[str, Any]]:
    return [
        {
            "event_ticker": "BTC-90K-JUN30",
            "market_ticker": "BTC-90K-JUN30",
            "title": "Will BTC exceed $90,000 by June 30, 2026?",
            "description": "Resolves YES if Bitcoin exceeds the threshold before close.",
            "category": "Financials",
            "rules": "Binary threshold event.",
            "close_time": "2026-06-30T23:59:59+00:00",
            "outcomes": ["Yes", "No"],
            "best_bid": 0.56,
            "best_ask": 0.60,
        },
        {
            "event_ticker": "US-CPI-MAY26",
            "market_ticker": "US-CPI-MAY26",
            "title": "What will US CPI year-over-year be for May 2026?",
            "description": "Macro release bucket event.",
            "category": "Economics",
            "rules": "Binary bucket-style macro release event.",
            "close_time": "2026-06-10T12:30:00+00:00",
            "outcomes": ["Above 2.8%", "2.8% or below"],
            "best_bid": 0.47,
            "best_ask": 0.51,
        },
        {
            "event_ticker": "NBA-GAME-6",
            "market_ticker": "NBA-GAME-6",
            "title": "Will Cleveland beat Detroit in NBA Eastern Conference Game 6?",
            "description": "Sports head-to-head event.",
            "category": "Sports",
            "rules": "YES if Cleveland wins.",
            "close_time": "2026-05-20T23:00:00+00:00",
            "outcomes": ["Cleveland", "Detroit"],
            "best_bid": 0.61,
            "best_ask": 0.65,
        },
        {
            "event_ticker": "OPENAI-ANNOUNCEMENT",
            "market_ticker": "OPENAI-ANNOUNCEMENT",
            "title": "Will OpenAI announce a major new multimodal product before July 2026?",
            "description": "Company/technology announcement event.",
            "category": "Technology",
            "rules": "YES if an official public launch announcement occurs before deadline.",
            "close_time": "2026-07-31T23:59:59+00:00",
            "outcomes": ["Yes", "No"],
            "best_bid": 0.42,
            "best_ask": 0.46,
        },
        {
            "event_ticker": "NYC-HEAT",
            "market_ticker": "NYC-HEAT",
            "title": "Will the temperature in New York exceed 95F on July 4, 2026?",
            "description": "Weather threshold event.",
            "category": "Weather",
            "rules": "YES if official daily max temperature exceeds 95F.",
            "close_time": "2026-07-04T23:59:59+00:00",
            "outcomes": ["Yes", "No"],
            "best_bid": 0.33,
            "best_ask": 0.37,
        },
    ]


if __name__ == "__main__":
    for sample_event in _sample_events():
        result = predict(sample_event)
        print(f"{sample_event['market_ticker']} | {sample_event['title']}")
        print(json.dumps(result, indent=2))
        print()
