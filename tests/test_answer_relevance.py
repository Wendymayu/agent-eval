import pytest

import agenteval
from agenteval import (
    AnswerRelevanceMetric,
    EvaluationResult,
    LLMTestCase,
    evaluate,
)
from agenteval.errors import MissingTestCaseParamsError
from agenteval.metrics.g_eval.schema import ReasonScore
from agenteval.models import DeepEvalBaseLLM, EvaluationCost


class AnswerRelevanceMock(DeepEvalBaseLLM):
    """Returns a ``(ReasonScore, EvaluationCost)`` tuple, matching the async
    extract path used by ``a_measure`` and ``evaluate`` (which dispatches to
    ``a_measure`` via the async engine). The sync ``measure`` path does not
    unpack such tuples, so tests exercise the async path — the default."""

    def __init__(self):
        super().__init__(model="armock")

    def load_model(self, *a, **k):
        return None

    def get_model_name(self, *a, **k):
        return "armock"

    def generate(self, prompt, schema=None):
        return (
            ReasonScore(
                score=0.8, reason="output directly addresses the input"
            ),
            EvaluationCost(0.0, 10, 20),
        )

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def test_public_api_exports_answer_relevance():
    assert hasattr(agenteval, "AnswerRelevanceMetric")


def test_answer_relevance_scores_with_mock_model():
    # async_mode=True (default) -> measure delegates to a_measure, whose
    # extract path unpacks the (ReasonScore, EvaluationCost) tuple the mock
    # returns. get_or_create_event_loop applies nest_asyncio so this works
    # under pytest-asyncio's auto mode.
    m = AnswerRelevanceMetric(model=AnswerRelevanceMock())
    tc = LLMTestCase(input="What is 2+2?", actual_output="4")
    m.measure(tc)
    assert m.score == 0.8
    assert m.reason == "output directly addresses the input"
    assert m.evaluation_model == "armock"
    assert m.is_successful() is True  # 0.8 >= 0.5 (default threshold)
    assert m.input_tokens == 10
    assert m.output_tokens == 20


def test_answer_relevance_threshold_fail():
    m = AnswerRelevanceMetric(model=AnswerRelevanceMock(), threshold=0.9)
    tc = LLMTestCase(input="q", actual_output="something")
    m.measure(tc)
    assert m.score == 0.8
    assert m.is_successful() is False  # 0.8 < 0.9


def test_answer_relevance_e2e_evaluate():
    # evaluate() defaults to run_async=True -> async engine -> a_measure.
    m = AnswerRelevanceMetric(model=AnswerRelevanceMock())
    cases = [LLMTestCase(input="What is 2+2?", actual_output="4")]
    result = evaluate(cases, [m])
    assert isinstance(result, EvaluationResult)
    tr = result.test_results[0]
    md = tr.metrics_data[0]
    assert md.name == "Answer Relevance"
    assert md.score == 0.8
    assert md.reason == "output directly addresses the input"
    assert md.evaluation_model == "armock"
    assert md.input_tokens == 10
    assert md.success is True


def test_answer_relevance_missing_actual_output_raises():
    # actual_output is Optional[str]; None reaches the metric's missing-params
    # check. (input is a required str at the pydantic layer, so it can never
    # be None by the time the metric sees the test case.)
    m = AnswerRelevanceMetric(model=AnswerRelevanceMock())
    tc = LLMTestCase(input="q", actual_output=None)
    with pytest.raises(MissingTestCaseParamsError):
        m.measure(tc)


def test_answer_relevance_empty_actual_output_raises():
    m = AnswerRelevanceMetric(model=AnswerRelevanceMock())
    tc = LLMTestCase(input="q", actual_output="")
    with pytest.raises(MissingTestCaseParamsError):
        m.measure(tc)
