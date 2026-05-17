from __future__ import annotations

from agents.blending import blend
from agents.template_models import ModelOutput


def test_factual_override_can_dominate_prior_when_high_confidence() -> None:
    result = blend(
        prior=0.62,
        model_outputs=[
            ModelOutput(
                p_model=0.01,
                confidence=0.96,
                data_quality=0.93,
                explanation="Official result already settles NO.",
                model_name="llm_factual_resolution_override",
            )
        ],
        template_confidence=0.80,
    )

    assert result.probability == 0.01
    assert result.diagnostics["llm_factual_override_target"] == 0.01


def test_low_confidence_factual_override_is_ignored() -> None:
    result = blend(
        prior=0.62,
        model_outputs=[
            ModelOutput(
                p_model=0.01,
                confidence=0.80,
                data_quality=0.93,
                explanation="Too weak.",
                model_name="llm_factual_resolution_override",
            )
        ],
        template_confidence=0.80,
    )

    assert result.probability == 0.62
    assert "llm_factual_override_target" not in result.diagnostics
