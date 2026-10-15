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
    used by ``measure`` and ``evaluate``.

    ``score`` models the strict 0/1 verdict the judge is now told to emit:
    1 = AI-flavored, 0 = reads naturally. Pass a non-binary value to exercise
    the metric's defensive coercion.
    """

    def __init__(self, score: float = 1):
        self._score = score
        super().__init__(model="aitone-mock")

    def load_model(self, *a, **k):
        return None

    def get_model_name(self, *a, **k):
        return "aitone-mock"

    async def generate(self, prompt, schema=None):
        return (
            ReasonScore(
                score=self._score,
                reason="heavy 首先/其次 transitions and 综上所述 boilerplate",
            ),
            EvaluationCost(0.0, 10, 20),
        )


def test_public_api_exports_ai_tone():
    assert hasattr(agenteval, "AIToneMetric")


def test_ai_tone_rejects_threshold_param():
    # Binary metrics have no user-tunable threshold — tuning a 0/1 score is a
    # no-op — so passing it is a programmer error, not a silently-ignored knob.
    with pytest.raises(TypeError):
        AIToneMetric(model=AIToneMock(), threshold=0.9)


async def test_ai_tone_scores_with_mock_model():
    # Binary verdict: score=1 means AI-flavored; default threshold 0.5 is the
    # tolerable ceiling, so 1 > 0.5 -> not successful (too much AI flavor).
    m = AIToneMetric(model=AIToneMock(score=1))
    tc = LLMTestCase(input="q", actual_output="首先，其次，最后，综上所述")
    await m.measure(tc)
    assert m.score == 1.0
    assert m.reason.startswith("heavy")
    assert m.evaluation_model == "aitone-mock"
    assert m.is_successful() is False  # AI-flavored -> exceeds ceiling
    assert m.input_tokens == 10
    assert m.output_tokens == 20


async def test_ai_tone_natural_passes():
    # score=0 means reads naturally; 0 <= 0.5 -> successful.
    m = AIToneMetric(model=AIToneMock(score=0))
    tc = LLMTestCase(input="q", actual_output="something")
    await m.measure(tc)
    assert m.score == 0.0
    assert m.is_successful() is True


async def test_ai_tone_coerces_nonbinary_judge():
    # The judge is told to return strict 0/1, but `score` must always honour
    # the binary contract: 0.8 (leans AI-flavored) coerces to 1.0 -> not
    # successful.
    m = AIToneMetric(model=AIToneMock(score=0.8))
    tc = LLMTestCase(input="q", actual_output="x")
    await m.measure(tc)
    assert m.score == 1.0
    assert m.is_successful() is False


def test_ai_tone_e2e_evaluate():
    # evaluate() runs the async engine, which calls measure.
    m = AIToneMetric(model=AIToneMock(score=1))
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
    assert md.score == 1.0
    assert md.evaluation_model == "aitone-mock"
    assert md.input_tokens == 10
    assert md.success is False  # AI-flavored -> exceeds default ceiling


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
