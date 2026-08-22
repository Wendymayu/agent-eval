import pytest
from pydantic import BaseModel

from agenteval.metrics import GEval, ExactMatchMetric
from agenteval.test_case import LLMTestCase, SingleTurnParams
from agenteval.models import DeepEvalBaseLLM, EvaluationCost
from agenteval.metrics.g_eval import schema as gschema


def test_exact_match_pass():
    m = ExactMatchMetric(threshold=0.5)
    m.measure(LLMTestCase(input="q", actual_output="4", expected_output="4"))
    assert m.score == 1.0
    assert m.is_successful() is True


def test_exact_match_fail():
    m = ExactMatchMetric(threshold=0.5)
    m.measure(LLMTestCase(input="q", actual_output="5", expected_output="4"))
    assert m.score == 0.0
    assert m.is_successful() is False


def test_geval_scores_with_mock_model():
    # GEval's a_measure path: _a_generate_evaluation_steps + _a_evaluate both
    # call a_generate_with_schema_and_extract, which (unlike the sync extract)
    # unpacks the (instance, EvaluationCost) tuples this mock returns. The
    # _a_evaluate logprobs path raises AttributeError on the missing
    # a_generate_raw_response and falls back to the schema call, so exactly two
    # model calls happen: first Steps, then ReasonScore.
    #
    # NOTE: async_mode=True (the default) runs a_measure via
    # loop.run_until_complete. The sync path (async_mode=False) does NOT unpack
    # (instance, cost) tuples, so a tuple-returning mock must use the async
    # path. get_or_create_event_loop applies nest_asyncio when a loop is
    # already running, so this works under pytest-asyncio's auto mode.
    class GEvalMock(DeepEvalBaseLLM):
        def __init__(self):
            self._calls = 0
            super().__init__(model="gmock")

        def load_model(self, *a, **k):
            return None

        def get_model_name(self, *a, **k):
            return "gmock"

        def generate(self, prompt, schema=None):
            self._calls += 1
            if self._calls == 1:
                return (
                    gschema.Steps(steps=["Compare actual to expected"]),
                    EvaluationCost(0.0, 5, 5),
                )
            return (
                gschema.ReasonScore(score=3, reason="matches well"),
                EvaluationCost(0.0, 8, 8),
            )

        async def a_generate(self, prompt, schema=None):
            return self.generate(prompt, schema)

    g = GEval(
        name="correctness",
        criteria="is the answer correct",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=GEvalMock(),
        async_mode=True,
    )
    tc = LLMTestCase(input="q", actual_output="4", expected_output="4")
    g.measure(tc)
    # score_range defaults to (0, 10) when no rubric is set, so a raw score of 3
    # normalises to 0.3.
    assert g.score is not None
    assert 0.0 <= g.score <= 1.0
    assert g.reason == "matches well"
