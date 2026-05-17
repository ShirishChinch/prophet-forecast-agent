"""Economics-specific forecasting components."""

from agents.economics.event_parser import EconomicEventSpec, parse_economic_event
from agents.economics.model_router import EconomicModelRouter

__all__ = [
    "EconomicEventSpec",
    "EconomicModelRouter",
    "parse_economic_event",
]
