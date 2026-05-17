"""Typed contracts for LLM-discovered and LLM-extracted evidence features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class NumericFeatureSpec:
    """One numeric feature the LLM may extract from unstructured evidence."""

    name: str
    description: str
    direction: str
    min_value: float
    max_value: float
    default_value: float = 0.0
    required_source_types: tuple[str, ...] = ()
    historical_measurement: str = ""
    leakage_risk: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceTemplateSpec:
    """Feature checklist for one reusable event category."""

    template: str
    description: str
    preferred_source_types: tuple[str, ...]
    features: tuple[NumericFeatureSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "description": self.description,
            "preferred_source_types": list(self.preferred_source_types),
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass(frozen=True)
class EvidenceSource:
    """One timestamped public source snippet."""

    url: str
    title: str = ""
    published_at: str | None = None
    source_type: str = "unknown"
    text: str = ""

    def is_known_before(self, as_of_time: str | None) -> bool:
        """Return False if source timing is after as_of_time or unparseable."""
        if not self.published_at or not as_of_time:
            return False
        published = parse_datetime(self.published_at)
        as_of = parse_datetime(as_of_time)
        if published is None or as_of is None:
            return False
        return published <= as_of

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedEvidenceFeatures:
    """Numeric row produced by the LLM extraction step."""

    template: str
    event_ticker: str
    market_ticker: str
    as_of_time: str
    features: dict[str, float]
    source_urls: list[str] = field(default_factory=list)
    source_count: int = 0
    usable_source_count: int = 0
    extraction_confidence: float = 0.0
    temporal_leakage_risk: float = 1.0
    short_rationale: str = ""

    def flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "template": self.template,
            "event_ticker": self.event_ticker,
            "market_ticker": self.market_ticker,
            "as_of_time": self.as_of_time,
            "source_count": self.source_count,
            "usable_source_count": self.usable_source_count,
            "extraction_confidence": self.extraction_confidence,
            "temporal_leakage_risk": self.temporal_leakage_risk,
            "source_urls": "|".join(self.source_urls),
            "short_rationale": self.short_rationale,
        }
        row.update(self.features)
        return row


def parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO-ish datetimes as UTC-aware datetimes."""
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def clamp_unit(value: Any, default: float = 0.0) -> float:
    """Clamp a value into [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def clamp_feature(value: Any, spec: NumericFeatureSpec) -> float:
    """Clamp one extracted feature according to its declared numeric bounds."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(spec.default_value)
    return max(float(spec.min_value), min(float(spec.max_value), number))

