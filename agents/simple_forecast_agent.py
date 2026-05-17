"""Backward-compatible alias for the router-based forecast agent."""

from __future__ import annotations

from agents.router_forecast_agent import RouterForecastAgent


class SimpleForecastAgent(RouterForecastAgent):
    """Compatibility wrapper around the new router-based agent."""

