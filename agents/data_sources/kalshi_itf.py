"""Helpers for clean Kalshi ITF tennis match markets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from agents.order_flow.kalshi_client import KalshiPublicClient


def fetch_itf_match_markets(
    *,
    horizon_hours: float = 24.0,
    max_pages: int = 3,
    timeout: float = 20.0,
    series_tickers: tuple[str, ...] = ("KXITFMATCH", "KXITFWMATCH"),
) -> list[dict[str, Any]]:
    """Fetch active KXITFMATCH winner markets expected to resolve soon."""
    client = KalshiPublicClient(timeout=timeout)
    now = datetime.now(UTC)
    end = now + timedelta(hours=horizon_hours)
    markets: list[dict[str, Any]] = []
    for series_ticker in series_tickers:
        markets.extend(
            client.paginated(
                "/markets",
                params={"series_ticker": series_ticker, "status": "open"},
                item_key="markets",
                limit=200,
                max_pages=max_pages,
            )
        )
    rows: list[dict[str, Any]] = []
    for market in markets:
        finish = _parse_time(market.get("expected_expiration_time") or market.get("expiration_time") or market.get("close_time"))
        if finish is None or not (now <= finish <= end):
            continue
        row = dict(market)
        row["finish_time"] = finish.isoformat()
        row["hours_to_finish"] = (finish - now).total_seconds() / 3600.0
        bid = _probability(market.get("yes_bid_dollars") or market.get("yes_bid"))
        ask = _probability(market.get("yes_ask_dollars") or market.get("yes_ask"))
        last = _probability(market.get("last_price_dollars") or market.get("last_price"))
        if bid is not None and ask is not None and ask >= bid:
            row["prior"] = (bid + ask) / 2.0
            row["prior_source"] = "yes_bid_ask_midpoint"
            row["spread"] = ask - bid
        elif last is not None:
            row["prior"] = last
            row["prior_source"] = "last_price"
            row["spread"] = None
        else:
            continue
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("hours_to_finish") or 999.0),
            -float(row.get("volume_fp") or 0.0),
        ),
    )


def fetch_event_markets(event_ticker: str, *, timeout: float = 20.0) -> list[dict[str, Any]]:
    """Fetch the two winner markets for a known KXITFMATCH event ticker."""
    client = KalshiPublicClient(timeout=timeout)
    payload = client.get_json(f"/events/{event_ticker}")
    markets = payload.get("markets") or []
    return [market for market in markets if isinstance(market, dict)]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:
        number /= 100.0
    if number < 0.0 or number > 1.0:
        return None
    return number
