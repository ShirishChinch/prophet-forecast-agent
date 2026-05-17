"""Read-only public Kalshi market data helpers."""

from __future__ import annotations

from functools import lru_cache
import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KALSHI_PUBLIC_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def get_public_market_for_event(event: dict[str, Any], timeout: float = 10.0) -> dict[str, Any] | None:
    """Find the best matching public Kalshi market for a forecast event.

    The sample datasets often use an event-family ticker in `market_ticker`
    rather than a specific market ticker. We first try the ticker as an exact
    market id, then fall back to listing markets by event_ticker.
    """
    market_ticker = str(event.get("market_ticker") or "").strip()
    event_ticker = str(event.get("event_ticker") or "").strip()

    if event_ticker and _looks_like_event_family(event):
        markets = _get_markets_by_event_ticker(event_ticker, timeout=timeout)
        if markets:
            return _select_best_market(event, markets)

    for ticker in (market_ticker, event_ticker):
        if not ticker:
            continue
        exact = _get_market_by_ticker(ticker, timeout=timeout)
        if exact is not None:
            return exact

    family_ticker = event_ticker or market_ticker
    if not family_ticker:
        return None

    markets = _get_markets_by_event_ticker(family_ticker, timeout=timeout)
    if not markets:
        return None
    return _select_best_market(event, markets)


def extract_bid_ask(market: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract YES bid/ask probabilities from a Kalshi market payload."""
    bid = _as_probability(
        market.get("yes_bid_dollars")
        or market.get("yes_bid")
        or market.get("best_bid")
    )
    ask = _as_probability(
        market.get("yes_ask_dollars")
        or market.get("yes_ask")
        or market.get("best_ask")
    )
    return bid, ask


@lru_cache(maxsize=512)
def _get_market_by_ticker(ticker: str, timeout: float = 10.0) -> dict[str, Any] | None:
    path = f"/markets/{ticker}"
    try:
        payload = _get_json(path, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except (URLError, TimeoutError, ValueError):
        return None
    market = payload.get("market")
    return market if isinstance(market, dict) else None


@lru_cache(maxsize=512)
def _get_markets_by_event_ticker(event_ticker: str, timeout: float = 10.0) -> tuple[dict[str, Any], ...]:
    query = urlencode({"event_ticker": event_ticker, "limit": 200})
    try:
        payload = _get_json(f"/markets?{query}", timeout=timeout)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ()
    markets = payload.get("markets")
    if not isinstance(markets, list):
        return ()
    return tuple(market for market in markets if isinstance(market, dict))


def _get_json(path: str, timeout: float) -> dict[str, Any]:
    request = Request(
        f"{KALSHI_PUBLIC_BASE_URL}{path}",
        headers={"User-Agent": "ai-prophet-local-agent/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(0.20 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise ValueError("Kalshi public response failed")


def _looks_like_event_family(event: dict[str, Any]) -> bool:
    outcomes = event.get("outcomes") if isinstance(event.get("outcomes"), list) else []
    if len(outcomes) > 2:
        return True
    market_ticker = str(event.get("market_ticker") or "")
    event_ticker = str(event.get("event_ticker") or "")
    title = str(event.get("title") or "").lower()
    if market_ticker and market_ticker == event_ticker and title.startswith(("what ", "where ")):
        return True
    return False


def _select_best_market(event: dict[str, Any], markets: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    event_rules = _normalize_text(str(event.get("rules") or event.get("description") or ""))
    event_title = _normalize_text(str(event.get("title") or ""))
    outcomes = event.get("outcomes") if isinstance(event.get("outcomes"), list) else []
    first_outcome = _normalize_text(str(outcomes[0])) if outcomes else ""

    def score(market: dict[str, Any]) -> tuple[int, int]:
        rules = _normalize_text(str(market.get("rules_primary") or ""))
        title = _normalize_text(str(market.get("title") or ""))
        subtitle = _normalize_text(
            str(market.get("yes_sub_title") or market.get("subtitle") or "")
        )
        ticker = _normalize_text(str(market.get("ticker") or ""))
        value = 0

        if event_rules and rules == event_rules:
            value += 100
        elif event_rules and _token_overlap(event_rules, rules) >= 0.85:
            value += 70
        elif event_rules and _token_overlap(event_rules, rules) >= 0.60:
            value += 40

        if first_outcome and (first_outcome == subtitle or first_outcome in rules):
            value += 50
        if event_title and _token_overlap(event_title, title) >= 0.50:
            value += 15
        if first_outcome and first_outcome in ticker:
            value += 5

        # Prefer tighter markets on ties.
        bid, ask = extract_bid_ask(market)
        spread_score = 0
        if bid is not None and ask is not None:
            spread_score = int(max(0.0, 1.0 - abs(ask - bid)) * 1000)
        return value, spread_score

    return max(markets, key=score)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in re.split(r"[^a-z0-9.]+", left) if token}
    right_tokens = {token for token in re.split(r"[^a-z0-9.]+", right) if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _as_probability(value: Any) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if prob > 1.0:
        prob = prob / 100.0
    if prob < 0.0 or prob > 1.0:
        return None
    return prob
