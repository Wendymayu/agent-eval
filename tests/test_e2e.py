from pydantic import BaseModel
import agenteval
from agenteval import (
    evaluate,
    LLMTestCase,
    GEval,
    DeepEvalBaseLLM,
    EvaluationCost,
    EvaluationResult,
)
from agenteval.test_case.llm_test_case import SingleTurnParams
from agenteval.metrics.g_eval import schema as gschema


class GEvalMock(DeepEvalBaseLLM):
    """Two-call mock: steps then score."""

    def __init__(self):
        self._calls = 0
        super().__init__(model="gmock")

    def load_model(self, *a, **k):
        return None

    def get_model_name(self, *a, **k):
        return "gmock"

    async def generate(self, prompt, schema=None):
        self._calls += 1
        if self._calls == 1:
            return gschema.Steps(steps=["Compare actual to expected"]), EvaluationCost(0.0, 5, 5)
        return gschema.ReasonScore(score=4, reason="correct"), EvaluationCost(0.0, 8, 8)


def test_public_api_exports():
    for name in ["evaluate", "LLMTestCase", "GEval", "DeepEvalBaseLLM", "OpenAIModel"]:
        assert hasattr(agenteval, name)


def test_e2e_geval_evaluate_with_mock_model():
    g = GEval(
        name="correctness",
        criteria="the answer is correct",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=GEvalMock(),
        threshold=0.5,
    )
    cases = [
        LLMTestCase(input="2+2=?", actual_output="4", expected_output="4"),
    ]
    result = evaluate(cases, [g])
    assert isinstance(result, EvaluationResult)
    tr = result.test_results[0]
    assert tr.metrics_data[0].name.startswith("correctness")
    assert tr.metrics_data[0].score is not None
    assert tr.metrics_data[0].evaluation_model == "gmock"
    assert tr.metrics_data[0].input_tokens == 13  # 5 + 8
