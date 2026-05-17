"""Structured evidence extraction stubs for web/LLM retrieval.

This module defines the contract for LLM-backed evidence. It intentionally does
not make claims or assign probabilities unless a retrieval implementation has
populated sources.
"""

from __future__ import annotations

from typing import Any


def extract_public_evidence_stub(event: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Return neutral structured evidence until retrieval is added."""
    _ = (event, spec)
    return {
        "direction": "neutral",
        "strength": "weak",
        "freshness": "unknown",
        "credibility": "unknown",
        "already_priced_likelihood": "high",
        "reasoning_adjustment": 0.0,
        "source_count": 0,
        "sources": [],
        "summary": "No web/LLM retrieval performed; numeric public-data collectors only.",
    }

