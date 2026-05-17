"""Small unauthenticated Kalshi public market-data client.

This module intentionally avoids trading/authenticated endpoints. It is for
public market data only: markets, trades, candlesticks, and order books.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


@dataclass(frozen=True)
class KalshiPublicClient:
    """Minimal public Kalshi REST client."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = 20.0
    user_agent: str = "ai-prophet-order-flow/0.1"

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a JSON endpoint with light retrying."""
        query = f"?{urlencode(_clean_params(params or {}), doseq=True)}" if params else ""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}{query}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                if exc.code in {400, 401, 403, 404}:
                    raise
            except (URLError, TimeoutError, ValueError) as exc:
                last_error = exc
            time.sleep(0.35 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Kalshi request failed")

    def paginated(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        item_key: str,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Collect a cursor-paginated list endpoint."""
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        page = 0
        while True:
            query = dict(params or {})
            query["limit"] = limit
            if cursor:
                query["cursor"] = cursor
            payload = self.get_json(path, query)
            items = payload.get(item_key) or []
            if isinstance(items, list):
                rows.extend(item for item in items if isinstance(item, dict))
            cursor = payload.get("cursor")
            page += 1
            if not cursor or (max_pages is not None and page >= max_pages):
                break
        return rows

    def get_markets(self, **params: Any) -> list[dict[str, Any]]:
        return self.paginated("/markets", params=params, item_key="markets")

    def get_historical_markets(self, **params: Any) -> list[dict[str, Any]]:
        return self.paginated("/historical/markets", params=params, item_key="markets")

    def get_trades(self, **params: Any) -> list[dict[str, Any]]:
        return self.paginated("/markets/trades", params=params, item_key="trades")

    def get_historical_trades(self, **params: Any) -> list[dict[str, Any]]:
        return self.paginated("/historical/trades", params=params, item_key="trades")

    def get_orderbook(self, market_ticker: str, depth: int = 100) -> dict[str, Any]:
        return self.get_json(f"/markets/{market_ticker}/orderbook", {"depth": depth})

    def get_candlesticks(
        self,
        market_tickers: list[str],
        *,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[dict[str, Any]]:
        payload = self.get_json(
            "/markets/candlesticks",
            {
                "market_tickers": ",".join(market_tickers),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        markets = payload.get("markets") or []
        return [market for market in markets if isinstance(market, dict)]


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None and value != ""}

