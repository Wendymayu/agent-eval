from agenteval.evaluate import evaluate, EvaluationResult
from agenteval.metrics import ExactMatchMetric
from agenteval.test_case import LLMTestCase


def test_evaluate_sync_returns_evaluation_result():
    cases = [
        LLMTestCase(input="q", actual_output="4", expected_output="4"),
        LLMTestCase(input="q", actual_output="5", expected_output="4"),
    ]
    metrics = [ExactMatchMetric(threshold=0.5)]
    from agenteval.evaluate.configs import AsyncConfig, ErrorConfig

    result = evaluate(
        cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(),
    )
    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 2
    assert result.test_results[0].success is True
    assert result.test_results[1].success is False
    assert result.test_results[0].metrics_data[0].name == "Exact Match"

    # Guard against the pydantic alias-drop bug: input/actual_output/
    # expected_output must propagate onto the TestResult (they reach
    # LLMApiTestCase only via the *alias* kwargs).
    tr = result.test_results[0]
    assert tr.input == "q"
    assert tr.actual_output == "4"
    assert tr.expected_output == "4"
    assert tr.conversational is False
    assert tr.multimodal is False
    assert tr.index == 0


def test_evaluate_async_returns_evaluation_result():
    cases = [LLMTestCase(input="q", actual_output="4", expected_output="4")] * 3
    metrics = [ExactMatchMetric(threshold=0.5)]
    from agenteval.evaluate.configs import AsyncConfig, ErrorConfig

    result = evaluate(
        cases,
        metrics,
        async_config=AsyncConfig(run_async=True),
        error_config=ErrorConfig(),
    )
    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 3
    assert all(r.success for r in result.test_results)
    assert result.test_results[0].metrics_data[0].name == "Exact Match"
    # Alias-drop guard for the async path too.
    assert result.test_results[0].actual_output == "4"
    assert result.test_results[0].expected_output == "4"
