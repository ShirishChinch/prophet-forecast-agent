# Prophet Forecast Agent

Forecasting endpoint for the Prophet Hacks forecasting track.

The deployed agent accepts one forecasting event at a time and returns a
probability distribution over the event outcomes.

Deployed endpoint:

```text
https://prophet-forecast-agent.onrender.com/predict
```

Health endpoint:

```text
https://prophet-forecast-agent.onrender.com/health
```

## Architecture

The agent is built around a conservative market-prior pipeline:

```text
event
-> deterministic parser/router
-> template family
-> market prior extraction
-> tennis/sector empirical bucket lookup
-> fast public order-flow residual model
-> optional LLM route/evidence checks
-> conservative blend/calibration
-> probabilities response
```

Main ideas:

- Use Kalshi-style market prices as the primary prior when available.
- Use empirical bucket lookup tables to correct recurring market calibration
  patterns by sector, with special handling for tennis.
- Use a lightweight public order-flow residual model only as a small nudge.
- Use LLMs for routing verification and high-conviction public evidence checks,
  not as a standalone probability oracle.
- Always return a valid prediction; failures fall back to the market prior.

## Endpoint Contract

`POST /predict` accepts an event shaped like the Prophet forecast quick-start:

```json
{
  "event_ticker": "task-001",
  "market_ticker": "task-001",
  "title": "Will BTC exceed $90,000 by March 21?",
  "category": "Financials",
  "rules": "Based on Coinbase spot price at close time.",
  "close_time": "2026-03-21T23:59:59Z",
  "outcomes": ["Yes", "No"],
  "best_bid": 0.52,
  "best_ask": 0.56
}
```

It returns:

```json
{
  "probabilities": [
    {"market": "Yes", "probability": 0.55},
    {"market": "No", "probability": 0.45}
  ]
}
```

The server may also include diagnostic fields such as `p_yes` and `rationale`.

## Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction smoke test:

```bash
python scripts/evaluate_agent_smoke.py --url http://127.0.0.1:8000/predict
```

To test the deployed endpoint:

```bash
python scripts/evaluate_agent_smoke.py --url https://prophet-forecast-agent.onrender.com/predict
```

## Environment Variables

Required for LLM-enabled evidence checks:

```text
OPENAI_API_KEY
```

Recommended runtime flags:

```text
TEMPLATE_ROUTE_LLM_VERIFY=1
SECTOR_ROUTE_LLM_VERIFY=1
LLM_CONVICTION_NUDGE_ENABLED=1
LLM_CONVICTION_WEB_SEARCH=1
LLM_CONVICTION_NUDGE_SIZE=0.05
ORDER_FLOW_LLM_ENABLED=1
ORDER_FLOW_OPENAI_WEB_SEARCH=1
ORDER_FLOW_ALLOW_NO_WEB_FALLBACK=1
```

Do not commit API keys.

## License

MIT recommended for hackathon submission.

