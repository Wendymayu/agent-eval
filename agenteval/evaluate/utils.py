from typing import Optional, List, Union

from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.test_run import LLMApiTestCase, ConversationalApiTestCase, MetricData
from agenteval.evaluate.types import TestResult


def create_metric_data(metric: BaseMetric) -> MetricData:
    """Build a :class:`MetricData` from a metric's (mutated) instance state.

    Uses the alias kwargs (``strictMode``/``evaluationModel``/``evaluationCost``
    /``inputTokenCount``/``outputTokenCount``/``verboseLogs``) that
    :class:`MetricData` defines in agenteval (Task 8). The metric attrs read
    here — ``__name__``, ``threshold``, ``score``, ``reason``,
    ``is_successful()``, ``strict_mode``, ``flaky``, ``evaluation_model``,
    ``error``, ``evaluation_cost``, ``input_tokens``, ``output_tokens``,
    ``verbose_logs`` — are all defined on ``BaseMetric``.
    """
    if metric.error is not None:
        return MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=metric.is_successful(),
            strictMode=metric.strict_mode,
            flaky=metric.flaky,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            inputTokenCount=metric.input_tokens,
            outputTokenCount=metric.output_tokens,
            verboseLogs=metric.verbose_logs,
        )
    return MetricData(
        name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        reason=metric.reason,
        success=metric.is_successful(),
        strictMode=metric.strict_mode,
        flaky=metric.flaky,
        evaluationModel=metric.evaluation_model,
        error=None,
        evaluationCost=metric.evaluation_cost,
        inputTokenCount=metric.input_tokens,
        outputTokenCount=metric.output_tokens,
        verboseLogs=metric.verbose_logs,
    )


def create_test_result(
    api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
) -> TestResult:
    name = api_test_case.name
    index = api_test_case.order
    if isinstance(api_test_case, ConversationalApiTestCase):
        return TestResult(
            name=name,
            success=api_test_case.success,
            metrics_data=api_test_case.metrics_data,
            conversational=True,
            index=index,
            metadata=api_test_case.metadata,
            turns=api_test_case.turns,
        )
    return TestResult(
        name=name,
        success=api_test_case.success,
        metrics_data=api_test_case.metrics_data,
        input=api_test_case.input,
        actual_output=api_test_case.actual_output,
        expected_output=api_test_case.expected_output,
        context=api_test_case.context,
        retrieval_context=api_test_case.retrieval_context,
        conversational=False,
        index=index,
        multimodal=bool(api_test_case.images_mapping),
        metadata=api_test_case.metadata,
    )


def check_valid_test_cases_type(test_cases):
    if not isinstance(test_cases, list):
        raise ValueError("test_cases must be a list.")
    for tc in test_cases:
        if not isinstance(tc, (LLMTestCase, ConversationalTestCase)):
            raise TypeError(
                f"Invalid test_case type: {type(tc).__name__}. "
                "Expected LLMTestCase or ConversationalTestCase."
            )


def validate_evaluate_inputs(test_cases, metrics):
    if metrics is None:
        raise ValueError("You must provide 'metrics'.")
    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("'metrics' must be a non-empty list.")
    for test_case in test_cases:
        for metric in metrics:
            if isinstance(test_case, LLMTestCase) and not isinstance(
                metric, BaseMetric
            ):
                raise ValueError(
                    f"Metric {metric.__name__} is not a valid metric for LLMTestCase."
                )
            if isinstance(test_case, ConversationalTestCase) and not isinstance(
                metric, BaseConversationalMetric
            ):
                raise ValueError(
                    f"Metric {metric.__name__} is not a valid metric for ConversationalTestCase."
                )
