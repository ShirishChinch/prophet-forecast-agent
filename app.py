"""HTTP wrapper for the local Prophet forecasting agent.

The hosted evaluator POSTs one event at a time to /predict.  The local
`my_agent.predict()` still returns the older binary shape:

    {"p_yes": 0.63, "rationale": "..."}

This server returns both that legacy shape and the newer website shape:

    {"probabilities": [{"market": "Yes", "probability": 0.63}, ...]}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request

from my_agent import predict as local_predict


app = FastAPI(title="Prophet Forecast Agent")


@app.get("/")
def root() -> dict[str, str]:
    """Simple root endpoint for browser checks."""
    return {"status": "ok", "service": "prophet-forecast-agent"}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check used by the evaluator to wake the service."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    """Forecast a single event and return website-compatible probabilities."""
    event = await request.json()
    result = local_predict(event)
    p_yes = _clamp_probability(result.get("p_yes"))
    rationale = str(result.get("rationale") or "")
    probabilities = _probabilities_for_outcomes(event, p_yes)
    return {
        "p_yes": p_yes,
        "rationale": rationale,
        "probabilities": probabilities,
    }


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

