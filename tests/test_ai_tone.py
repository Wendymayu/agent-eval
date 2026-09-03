import pytest

import agenteval
from agenteval import (
    AIToneMetric,
    EvaluationResult,
    LLMTestCase,
    evaluate,
)
from agenteval.errors import MissingTestCaseParamsError
from agenteval.metrics.g_eval.schema import ReasonScore
from agenteval.models import DeepEvalBaseLLM, EvaluationCost


class AIToneMock(DeepEvalBaseLLM):
    """Returns a ``(ReasonScore, EvaluationCost)`` tuple for the extract path
    used by ``measure`` and ``evaluate``."""

    def __init__(self):
        super().__init__(model="aitone-mock")

    def load_model(self, *a, **k):
        return None

    def get_model_name(self, *a, **k):
        return "aitone-mock"

    async def generate(self, prompt, schema=None):
        return (
            ReasonScore(
                score=0.8,
                reason="heavy 首先/其次 transitions and 综上所述 boilerplate",
            ),
            EvaluationCost(0.0, 10, 20),
        )


def test_public_api_exports_ai_tone():
    assert hasattr(agenteval, "AIToneMetric")


async def test_ai_tone_scores_with_mock_model():
    # score=0.8 is the AI-flavor density; default threshold 0.5 is the
    # tolerable ceiling, so 0.8 > 0.5 -> not successful (too much AI flavor).
    m = AIToneMetric(model=AIToneMock())
    tc = LLMTestCase(
        input="q", actual_output="首先，其次，最后，综上所述"
    )
    await m.measure(tc)
    assert m.score == 0.8
    assert m.reason.startswith("heavy")
    assert m.evaluation_model == "aitone-mock"
    assert m.is_successful() is False  # 0.8 > 0.5 (exceeds ceiling)
    assert m.input_tokens == 10
    assert m.output_tokens == 20


async def test_ai_tone_below_threshold_passes():
    # threshold 0.9 -> 0.8 <= 0.9 -> successful (AI flavor tolerable).
    m = AIToneMetric(model=AIToneMock(), threshold=0.9)
    tc = LLMTestCase(input="q", actual_output="something")
    await m.measure(tc)
    assert m.score == 0.8
    assert m.is_successful() is True


def test_ai_tone_e2e_evaluate():
    # evaluate() runs the async engine, which calls measure.
    m = AIToneMetric(model=AIToneMock())
    cases = [
        LLMTestCase(
            input="q", actual_output="首先，其次，最后，综上所述"
        )
    ]
    result = evaluate(cases, [m])
    assert isinstance(result, EvaluationResult)
    tr = result.test_results[0]
    md = tr.metrics_data[0]
    assert md.name == "AI Tone"
    assert md.score == 0.8
    assert md.evaluation_model == "aitone-mock"
    assert md.input_tokens == 10
    assert md.success is False  # 0.8 > 0.5 (default ceiling)


async def test_ai_tone_missing_actual_output_raises():
    m = AIToneMetric(model=AIToneMock())
    tc = LLMTestCase(input="q", actual_output=None)
    with pytest.raises(MissingTestCaseParamsError):
        await m.measure(tc)


async def test_ai_tone_empty_actual_output_raises():
    m = AIToneMetric(model=AIToneMock())
    tc = LLMTestCase(input="q", actual_output="")
    with pytest.raises(MissingTestCaseParamsError):
        await m.measure(tc)
