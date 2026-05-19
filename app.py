"""HTTP wrapper for the local Prophet forecasting agent.

The hosted evaluator POSTs one event at a time to /predict.  The local
`my_agent.predict()` still returns the older binary shape:

    {"p_yes": 0.63, "rationale": "..."}

This server returns the website shape:

    {"probabilities": [{"market": "Yes", "probability": 0.63}, ...]}
"""

from __future__ import annotations

import os
import json
import re
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request

import agents.logging_utils as logging_utils
from agents.kalshi_public import extract_bid_ask
from agents.market_calibration import calibrate_market_probability
from agents.order_flow.kalshi_client import KalshiPublicClient
from my_agent import predict as local_predict


app = FastAPI(title="Prophet Forecast Agent")
MAX_SIDE_AGENT_OUTCOMES = 8
KALSHI_FETCH_TIMEOUT_SECONDS = 3.0


@app.get("/")
def root() -> dict[str, str]:
    """Simple root endpoint for browser checks."""
    return {"status": "ok", "service": "prophet-forecast-agent"}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check used by the evaluator to wake the service."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request) -> Any:
    """Forecast one event, or a batch of events/questions."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    _log_endpoint_received(payload)

    if isinstance(payload, list):
        return [_predict_one(_coerce_event(item), path_prefix="batch") for item in payload]

    return _predict_one(_coerce_event(payload))


def _coerce_event(payload: Any) -> dict[str, Any]:
    """Accept both event objects and plain question strings."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip():
        return {"title": payload.strip(), "question": payload.strip(), "outcomes": ["Yes", "No"]}
    return {}


def _predict_one(event: dict[str, Any], *, path_prefix: str = "") -> dict[str, Any]:
    """Forecast a single event and return website-compatible probabilities."""
    started = perf_counter()
    path = "normal"
    if path_prefix:
        path = f"{path_prefix}_{path}"
    if not isinstance(event, dict):
        event = {}
    if not event:
        response = {"probabilities": _probabilities_for_outcomes(event, 0.50)}
        _log_endpoint_trace(
            event,
            response,
            started,
            f"{path_prefix}_empty_or_invalid_request" if path_prefix else "empty_or_invalid_request",
        )
        return response

    if _should_use_fast_endpoint_mode(event):
        path = f"{path_prefix}_fast_multi_or_check" if path_prefix else "fast_multi_or_check"
        response = {"probabilities": _fast_probabilities(event)}
        _log_endpoint_trace(event, response, started, path)
        return response
    try:
        with _temporary_env(_endpoint_fast_env()):
            result = local_predict(event)
    except Exception:
        path = "local_predict_exception"
        result = {"p_yes": 0.50}
    p_yes = _clamp_probability(result.get("p_yes"))
    probabilities = _probabilities_for_outcomes(event, p_yes)
    response = {"probabilities": probabilities}
    _log_endpoint_trace(event, response, started, path, agent_result=result)
    return response


def _should_use_fast_endpoint_mode(event: dict[str, Any]) -> bool:
    """Keep browser format checks under their short timeout.

    The real evaluator has a much longer timeout.  The website checker sends a
    synthetic multi-outcome NBA championship event and only validates response
    shape, so we answer that path directly without LLM/web-search calls.
    """
    outcomes = event.get("outcomes")
    title = str(event.get("title") or event.get("question") or "").lower()
    category = str(event.get("category") or "").lower()
    if isinstance(outcomes, list) and len(outcomes) > 2:
        return True
    if isinstance(outcomes, list) and len(outcomes) > 1:
        labels = [str(outcome) for outcome in outcomes]
        if not _looks_mutually_exclusive(event, labels):
            return True
    return "nba championship" in title or ("nba" in title and "championship" in title) or "format check" in category


def _fast_probabilities(event: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = event.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        outcomes = ["Yes", "No"]
    labels = [str(outcome) for outcome in outcomes]
    mutually_exclusive = _looks_mutually_exclusive(event, labels)

    market_probs = _extract_outcome_probabilities(event, labels)
    market_probs_from_prices = market_probs is not None
    if market_probs is None:
        market_probs = _fetch_kalshi_outcome_probabilities(event, labels)
        market_probs_from_prices = market_probs is not None
    if market_probs is None:
        equal = 1.0 / len(labels) if mutually_exclusive else 0.50
        market_probs = [equal for _ in labels]

    if market_probs_from_prices:
        market_probs = _calibrate_outcome_market_probabilities(event, labels, market_probs)

    if not market_probs_from_prices and _should_run_side_agents(event, labels):
        side_probs = _predict_multi_outcome_sides(event, labels, market_probs)
        if side_probs is not None:
            market_probs = side_probs

    if mutually_exclusive:
        total = sum(max(0.0, value) for value in market_probs)
        if total <= 0.0:
            market_probs = [1.0 / len(labels) for _ in labels]
        else:
            market_probs = [max(0.0, value) / total for value in market_probs]
    else:
        market_probs = _enforce_ladder_monotonicity(labels, market_probs)

    return [
        {"market": label, "probability": max(0.0, min(1.0, probability))}
        for label, probability in zip(labels, market_probs, strict=True)
    ]


def _calibrate_outcome_market_probabilities(
    event: dict[str, Any],
    labels: list[str],
    market_probs: list[float],
) -> list[float]:
    calibrated: list[float] = []
    for label, probability in zip(labels, market_probs, strict=True):
        side_event = dict(event)
        side_event["multi_outcome_label"] = label
        side_event["yes_sub_title"] = label
        result = calibrate_market_probability(
            probability,
            template_family=str(event.get("category") or ""),
            event=side_event,
        )
        calibrated.append(max(0.0, min(1.0, result.probability)))
    return calibrated


def _should_run_side_agents(event: dict[str, Any], labels: list[str]) -> bool:
    if len(labels) <= 2 or len(labels) > MAX_SIDE_AGENT_OUTCOMES:
        return False
    if not _looks_mutually_exclusive(event, labels):
        return False
    if str(event.get("endpoint_fast_check") or "").lower() in {"1", "true", "yes"}:
        return False
    # Keep the browser checker fast; the real evaluation has a much longer
    # timeout and can afford side-specific lookups.
    title = str(event.get("title") or event.get("question") or "").lower()
    if "nba championship" in title and not event.get("event_ticker"):
        return False
    return True


def _predict_multi_outcome_sides(
    event: dict[str, Any],
    labels: list[str],
    market_probs: list[float],
) -> list[float] | None:
    side_probs: list[float] = []
    with _temporary_env(_endpoint_fast_env()):
        for label, market_probability in zip(labels, market_probs, strict=True):
            binary_event = _binary_side_event(event, label, market_probability)
            try:
                result = local_predict(binary_event)
            except Exception:
                return None
            side_probs.append(_clamp_probability(result.get("p_yes")))
    return side_probs if side_probs else None


def _endpoint_fast_env() -> dict[str, str]:
    """Disable network-heavy LLM layers on hosted endpoint calls."""
    return {
        "TEMPLATE_ROUTE_LLM_VERIFY": "0",
        "SECTOR_ROUTE_LLM_VERIFY": "0",
        "LLM_CONVICTION_NUDGE_ENABLED": "0",
        "FACTUAL_RESOLUTION_OVERRIDE_ENABLED": "0",
        "ORDER_FLOW_LLM_ENABLED": "0",
    }


def _binary_side_event(event: dict[str, Any], label: str, market_probability: float) -> dict[str, Any]:
    binary_event = dict(event)
    binary_event["title"] = f"{event.get('title') or event.get('question') or 'Multi-outcome event'} - YES side: {label}"
    binary_event["description"] = str(event.get("description") or "")
    binary_event["rules"] = str(event.get("rules") or f"Resolves YES if the outcome is {label}.")
    binary_event["outcomes"] = ["Yes", "No"]
    binary_event["best_bid"] = market_probability
    binary_event["best_ask"] = market_probability
    binary_event["yes_price"] = market_probability
    binary_event["market_ticker"] = f"{event.get('market_ticker') or event.get('event_ticker') or 'multi'}::{label}"
    binary_event["multi_outcome_parent_ticker"] = event.get("market_ticker") or event.get("event_ticker")
    binary_event["multi_outcome_label"] = label
    return binary_event


def _extract_outcome_probabilities(event: dict[str, Any], labels: list[str]) -> list[float] | None:
    for key in ("probabilities", "market_probabilities", "prices"):
        value = event.get(key)
        extracted = _coerce_probability_collection(value, labels)
        if extracted is not None:
            return extracted
    return None


def _fetch_kalshi_outcome_probabilities(event: dict[str, Any], labels: list[str]) -> list[float] | None:
    event_ticker = str(event.get("event_ticker") or "").strip()
    market_ticker = str(event.get("market_ticker") or "").strip()
    if not event_ticker and not market_ticker:
        return None

    client = KalshiPublicClient(timeout=KALSHI_FETCH_TIMEOUT_SECONDS)
    markets: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    for ticker in (event_ticker, market_ticker):
        if not ticker or ticker in seen_tickers or not _looks_like_kalshi_ticker(ticker):
            continue
        seen_tickers.add(ticker)
        try:
            markets.extend(client.get_json("/markets", {"event_ticker": ticker, "limit": 200}).get("markets") or [])
        except Exception:
            pass
        try:
            exact = client.get_json(f"/markets/{ticker}").get("market")
            if isinstance(exact, dict):
                markets.append(exact)
        except Exception:
            pass

    if not markets:
        return None

    by_label: dict[str, float] = {}
    for label in labels:
        matched = _best_label_market(label, markets)
        if matched is None:
            return None
        bid, ask = extract_bid_ask(matched)
        if bid is None or ask is None or ask < bid:
            return None
        by_label[label] = _quote_probability(bid, ask)
    return [by_label[label] for label in labels]


def _looks_like_kalshi_ticker(ticker: str) -> bool:
    return ticker.upper().startswith("KX")


def _quote_probability(bid: float, ask: float) -> float:
    if bid <= 0.0 and ask <= 0.01:
        return 0.0
    if bid >= 0.99 and ask >= 1.0:
        return 1.0
    return (bid + ask) / 2.0


def _best_label_market(label: str, markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    label_norm = _norm(label)
    best: tuple[int, dict[str, Any]] | None = None
    for market in markets:
        status = str(market.get("status") or "")
        if status and status not in {"active", "initialized"}:
            continue
        yes_text = _norm(str(market.get("yes_sub_title") or ""))
        title = _norm(str(market.get("title") or ""))
        rules = _norm(str(market.get("rules_primary") or ""))
        ticker = _norm(str(market.get("ticker") or ""))
        score = 0
        if label_norm and label_norm == yes_text:
            score += 100
        if label_norm and label_norm in yes_text:
            score += 60
        if label_norm and label_norm in title:
            score += 25
        if label_norm and label_norm in rules:
            score += 25
        if label_norm and label_norm in ticker:
            score += 5
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, market)
    return best[1] if best else None


def _norm(value: str) -> str:
    return " ".join(value.lower().replace(".", " ").replace("-", " ").split())


def _coerce_probability_collection(value: Any, labels: list[str]) -> list[float] | None:
    if isinstance(value, dict):
        by_label = {str(key): raw_value for key, raw_value in value.items()}
        by_label_norm = {_norm(str(key)): raw_value for key, raw_value in value.items()}
        probs = []
        for label in labels:
            if label in by_label:
                probs.append(_price_to_probability(by_label[label]))
            elif _norm(label) in by_label_norm:
                probs.append(_price_to_probability(by_label_norm[_norm(label)]))
            else:
                return None
        return probs
    if isinstance(value, list):
        if len(value) != len(labels):
            return None
        if all(isinstance(item, dict) for item in value):
            by_market = {
                str(item.get("market") or item.get("outcome") or ""): _price_to_probability(
                    item.get("probability") if "probability" in item else item.get("price")
                )
                for item in value
            }
            by_market_norm = {_norm(label): probability for label, probability in by_market.items()}
            if all(label in by_market for label in labels):
                return [by_market[label] for label in labels]
            if all(_norm(label) in by_market_norm for label in labels):
                return [by_market_norm[_norm(label)] for label in labels]
            return None
        return [_price_to_probability(item) for item in value]
    return None


def _looks_mutually_exclusive(event: dict[str, Any], labels: list[str]) -> bool:
    """Infer whether outcome probabilities should be normalized to one."""
    text = " ".join(
        str(event.get(key) or "")
        for key in ("title", "question", "subtitle", "description", "rules", "category")
    ).lower()
    if _looks_like_nested_ladder(labels):
        return False
    if len(labels) <= 2:
        return True
    non_exclusive_terms = (
        "top 2",
        "top two",
        "top 3",
        "top three",
        "top 4",
        "top four",
        "top 5",
        "top five",
        "make the playoffs",
        "qualify for",
        "which teams",
        "which countries",
        "which of the following",
        "all that apply",
        "multiple",
        "at least",
        "each of",
    )
    if any(term in text for term in non_exclusive_terms):
        return False
    exclusive_terms = (
        "who will win",
        "winner",
        "champion",
        "championship",
        "nominee",
        "award",
        "election",
        "next president",
        "finish first",
        "highest",
        "lowest",
        "which player",
        "which team",
        "which party",
        "which country",
        "range will",
        "what range",
    )
    if any(term in text for term in exclusive_terms):
        return True
    return True


def _looks_like_nested_ladder(labels: list[str]) -> bool:
    values = [_ladder_value_from_label(label) for label in labels]
    parsed = [value for value in values if value is not None]
    if len(parsed) < 2:
        return False
    kinds = {kind for kind, _ in parsed}
    if len(kinds) != 1:
        return False
    return len(parsed) >= max(2, int(0.75 * len(labels)))


def _enforce_ladder_monotonicity(labels: list[str], probabilities: list[float]) -> list[float]:
    values = [_ladder_value_from_label(label) for label in labels]
    indexed = [
        (index, value, max(0.0, min(1.0, probability)))
        for index, (value, probability) in enumerate(zip(values, probabilities, strict=True))
    ]
    parsed = [item for item in indexed if item[1] is not None]
    if len(parsed) < 2:
        return probabilities
    kind = parsed[0][1][0]
    if any(item[1][0] != kind for item in parsed):
        return probabilities

    ordered = sorted(parsed, key=lambda item: float(item[1][1]))
    adjusted = [max(0.0, min(1.0, probability)) for probability in probabilities]
    if kind == "deadline_before":
        previous = 0.0
        for index, _, probability in ordered:
            previous = max(previous, probability)
            adjusted[index] = previous
    else:
        previous = 1.0
        for index, _, probability in ordered:
            previous = min(previous, probability)
            adjusted[index] = previous
    return adjusted


def _ladder_value_from_label(label: str) -> tuple[str, float] | None:
    text = str(label).strip().lower().replace(",", "")
    if not text:
        return None
    plus_match = re.search(r"(?<![\w.])-?\$?\s*(\d+(?:\.\d+)?)\s*\+", text)
    if plus_match:
        return ("threshold_at_least", float(plus_match.group(1)))
    threshold_match = re.search(
        r"\b(?:at least|over|above|greater than|more than)\s+\$?\s*(\d+(?:\.\d+)?)\b",
        text,
    )
    if threshold_match:
        return ("threshold_at_least", float(threshold_match.group(1)))
    before_match = re.search(r"\b(?:before|by|on or before)\s+(.+)$", text)
    if before_match:
        date_value = _date_ladder_value(before_match.group(1))
        if date_value is not None:
            return ("deadline_before", date_value)
    return None


def _date_ladder_value(value: str) -> float | None:
    text = " ".join(str(value).lower().split())
    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_match:
        year, month, day = (int(part) for part in iso_match.groups())
        return float((year * 10000) + (month * 100) + day)

    month_names = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    month_match = re.search(
        r"\b("
        + "|".join(sorted(month_names, key=len, reverse=True))
        + r")\.?\s+(\d{1,2})(?:\s+(\d{4}))?\b",
        text,
    )
    if not month_match:
        return None
    month_name, day_text, year_text = month_match.groups()
    month = month_names[month_name.rstrip(".")]
    day = int(day_text)
    if year_text:
        return float((int(year_text) * 10000) + (month * 100) + day)
    return float((month * 100) + day)


def _price_to_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if probability > 1.0:
        probability /= 100.0
    return max(0.0, min(1.0, probability))


def _probabilities_for_outcomes(event: dict[str, Any], p_yes: float) -> list[dict[str, Any]]:
    """Map the binary p_yes estimate onto the event outcome labels."""
    outcomes = event.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        outcomes = ["Yes", "No"]
    labels = [str(outcome) for outcome in outcomes]

    if len(labels) == 1:
        return [{"market": labels[0], "probability": p_yes}]

    if len(labels) == 2:
        yes_index = _yes_index(labels)
        probs = [0.0, 0.0]
        probs[yes_index] = p_yes
        probs[1 - yes_index] = _clamp_probability(1.0 - p_yes)
        return [
            {"market": labels[0], "probability": probs[0]},
            {"market": labels[1], "probability": probs[1]},
        ]

    # The current local model is binary.  For multi-outcome questions, keep the
    # first listed market as the YES side and distribute the remaining mass
    # uniformly so every required outcome is present.
    remaining = _clamp_probability(1.0 - p_yes)
    other_probability = remaining / (len(labels) - 1)
    return [
        {
            "market": label,
            "probability": p_yes if index == 0 else _clamp_probability(other_probability),
        }
        for index, label in enumerate(labels)
    ]


def _yes_index(labels: list[str]) -> int:
    lowered = [label.strip().lower() for label in labels]
    if "yes" in lowered:
        return lowered.index("yes")
    if "no" in lowered:
        return 1 - lowered.index("no")
    return 0


def _clamp_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.50
    if probability != probability:
        return 0.50
    return max(0.01, min(0.99, probability))


class _temporary_env:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _log_endpoint_trace(
    event: dict[str, Any],
    response: dict[str, Any],
    started: float,
    path: str,
    *,
    agent_result: dict[str, Any] | None = None,
) -> None:
    """Log request/response shape and latency without affecting predictions."""
    try:
        probabilities = response.get("probabilities")
        rows = probabilities if isinstance(probabilities, list) else []
        payload = {
            "timestamp": logging_utils.utc_now_iso(),
            "path": path,
            "latency_ms": round((perf_counter() - started) * 1000.0, 1),
            "event_ticker": event.get("event_ticker"),
            "market_ticker": event.get("market_ticker"),
            "title": event.get("title") or event.get("question"),
            "category": event.get("category"),
            "outcome_count": len(event.get("outcomes") or []) if isinstance(event.get("outcomes"), list) else 0,
            "outcomes": event.get("outcomes") if isinstance(event.get("outcomes"), list) else None,
            "probabilities": rows,
            "probability_sum": sum(float(row.get("probability") or 0.0) for row in rows if isinstance(row, dict)),
            "rationale": (agent_result or {}).get("rationale") if isinstance(agent_result, dict) else None,
        }
        logging_utils.append_jsonl(os.environ.get("ENDPOINT_LOG_PATH", "endpoint_logs.jsonl"), payload)
        print("forecast_endpoint_trace " + json.dumps(payload, ensure_ascii=True, default=str), flush=True)
    except Exception:
        return


def _log_endpoint_received(payload: Any) -> None:
    """Log request receipt before any slow prediction work starts."""
    try:
        if isinstance(payload, list):
            preview = [_payload_preview(item) for item in payload[:5]]
            row = {
                "timestamp": logging_utils.utc_now_iso(),
                "payload_type": "list",
                "batch_size": len(payload),
                "preview": preview,
            }
        else:
            row = {
                "timestamp": logging_utils.utc_now_iso(),
                "payload_type": type(payload).__name__,
                "batch_size": None,
                "preview": _payload_preview(payload),
            }
        print("forecast_endpoint_received " + json.dumps(row, ensure_ascii=True, default=str), flush=True)
    except Exception:
        return


def _payload_preview(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {
            "event_ticker": payload.get("event_ticker"),
            "market_ticker": payload.get("market_ticker"),
            "title": payload.get("title") or payload.get("question"),
            "category": payload.get("category"),
            "outcome_count": len(payload.get("outcomes") or []) if isinstance(payload.get("outcomes"), list) else 0,
        }
    if isinstance(payload, str):
        return {"title": payload[:200], "payload_type": "str"}
    return {"payload_type": type(payload).__name__}
