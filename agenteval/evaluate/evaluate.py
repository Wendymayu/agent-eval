from typing import List, Optional, Union

from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
from agenteval.evaluate.execute import execute_test_cases, a_execute_test_cases
from agenteval.evaluate.types import EvaluationResult
from agenteval.evaluate.utils import (
    check_valid_test_cases_type,
    validate_evaluate_inputs,
)
from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.metrics.utils import check_at_least_one_metric_has_threshold
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.utils import get_or_create_event_loop


def evaluate(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    metrics: Optional[
        Union[List[BaseMetric], List[BaseConversationalMetric]]
    ] = None,
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> EvaluationResult:
    """Evaluate ``test_cases`` against ``metrics``.

    The judge model lives on each metric (no ``model`` param here). The engine
    deep-copies metrics per case so concurrent cases don't share mutable metric
    state. Stripped vs. deepeval: reporting, persistence, Confident cloud,
    cache, timeout, ``metric_collection``.
    """
    check_valid_test_cases_type(test_cases)
    validate_evaluate_inputs(test_cases, metrics)
    check_at_least_one_metric_has_threshold(metrics)

    if async_config.run_async:
        loop = get_or_create_event_loop()
        test_results = loop.run_until_complete(
            a_execute_test_cases(
                test_cases,
                metrics,
                async_config=async_config,
                error_config=error_config,
            )
        )
    else:
        test_results = execute_test_cases(
            test_cases, metrics, error_config=error_config
        )

    return EvaluationResult(
        test_results=test_results,
        confident_link=None,
        test_run_id=None,
    )
