import asyncio
from typing import List, Optional, Union

from agenteval.errors import MissingTestCaseParamsError
from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
from agenteval.evaluate.utils import create_metric_data, create_test_result
from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.metrics.utils import copy_metrics
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.test_run import LLMApiTestCase, ConversationalApiTestCase


def _handle_missing_params(
    metric: BaseMetric, e: MissingTestCaseParamsError, ec: ErrorConfig
) -> bool:
    """Apply the skip/error policy for a missing-params failure.

    Returns ``True`` when the metric is skipped (no :class:`MetricData` should
    be emitted for it — mirroring deepeval's ``"skip"`` ``continue``), ``False``
    when it was recorded as an error (a :class:`MetricData` *should* still be
    emitted), and re-raises when errors are not ignored.
    """
    if ec.skip_on_missing_params:
        metric.skipped = True
        metric.error = None
        metric.success = None
        return True  # skip
    if ec.ignore_errors:
        metric.error = str(e)
        metric.success = False
        return False
    raise


async def _run_metric(
    metric: BaseMetric,
    test_case: Union[LLMTestCase, ConversationalTestCase],
    ec: ErrorConfig,
) -> bool:
    """Run ``await metric.measure`` with error handling.

    Returns ``True`` if the metric was skipped (caller must NOT emit
    :class:`MetricData` for it), ``False`` otherwise.
    """
    try:
        await metric.measure(test_case, _show_indicator=False)
    except MissingTestCaseParamsError as e:
        return _handle_missing_params(metric, e, ec)
    except TypeError:
        # Some metrics don't accept ``_show_indicator``; retry without it.
        try:
            await metric.measure(test_case)
        except MissingTestCaseParamsError as e:
            return _handle_missing_params(metric, e, ec)
        except Exception as e:
            if ec.ignore_errors:
                metric.error = str(e)
                metric.success = False
            else:
                raise
    except Exception as e:
        if ec.ignore_errors:
            metric.error = str(e)
            metric.success = False
        else:
            raise
    return False


def _build_api_test_case(test_case: LLMTestCase, index: int) -> LLMApiTestCase:
    """Map an :class:`LLMTestCase` to an :class:`LLMApiTestCase`.

    Uses the alias kwargs (``actualOutput``/``expectedOutput``/
    ``retrievalContext``/``toolsCalled``/``expectedTools``/``imagesMapping``
    /``tokenCost``/``completionTime``/``metricsData``/``runDuration``/
    ``evaluationCost``) because :class:`LLMApiTestCase` defines those fields
    with ``alias=`` (not ``AliasChoices``) and ``populate_by_name`` is not
    enabled, so the field-name kwargs would be silently dropped by pydantic.
    Mirrors deepeval's ``test_case.api.create_api_test_case`` (LLM branch),
    stripped of the ``trace``/PYTEST_RUN_TEST_NAME plumbing.
    """
    name = test_case.name or str(test_case.input)[:50] or f"TestCase_{index}"

    # Normalize RetrievedContextData -> str, exactly like deepeval's builder.
    retrieval_context = None
    if test_case.retrieval_context:
        retrieval_context = [
            rc.context if hasattr(rc, "context") else rc
            for rc in test_case.retrieval_context
        ]

    return LLMApiTestCase(
        name=name,
        input=test_case.input,
        actualOutput=test_case.actual_output,
        expectedOutput=test_case.expected_output,
        retrievalContext=retrieval_context,
        context=test_case.context,
        imagesMapping=test_case._get_images_mapping(),
        toolsCalled=test_case.tools_called,
        expectedTools=test_case.expected_tools,
        tokenCost=test_case.token_cost,
        completionTime=test_case.completion_time,
        success=True,
        metricsData=[],
        runDuration=None,
        evaluationCost=None,
        order=index,
        metadata=test_case.metadata,
        comments=test_case.comments,
        tags=test_case.tags,
        flaky=test_case.flaky,
    )


async def execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    async_config: AsyncConfig,
    error_config: ErrorConfig,
) -> List:
    """Run ``measure`` over every case/metric and build a :class:`TestResult`
    per case.

    When ``async_config.run_async`` is set, cases run concurrently (bounded by
    ``max_concurrent``) and metrics within a case run concurrently on distinct
    copied instances; otherwise cases and metrics run sequentially. Metrics are
    deep-copied per case so cases never share mutable metric state.
    """
    if async_config.run_async:
        sem = asyncio.Semaphore(async_config.max_concurrent)

        async def run_one(idx: int, tc: LLMTestCase):
            async with sem:
                metrics_copy = copy_metrics(metrics)
                api_tc = _build_api_test_case(tc, idx)
                # Each copied metric is a distinct instance, so running them
                # concurrently on the same test case is safe.
                skip_flags = await asyncio.gather(
                    *[_run_metric(m, tc, error_config) for m in metrics_copy]
                )
                for metric, skipped in zip(metrics_copy, skip_flags):
                    if skipped:
                        # Skipped metrics emit no MetricData (matches
                        # deepeval's `if res == "skip": continue`).
                        continue
                    api_tc.update_metric_data(create_metric_data(metric))
                return create_test_result(api_tc)

        return await asyncio.gather(
            *[run_one(i, tc) for i, tc in enumerate(test_cases)]
        )

    # Sequential path
    test_results = []
    for idx, tc in enumerate(test_cases):
        metrics_copy = copy_metrics(metrics)
        api_tc = _build_api_test_case(tc, idx)
        for metric in metrics_copy:
            skipped = await _run_metric(metric, tc, error_config)
            if skipped:
                continue
            api_tc.update_metric_data(create_metric_data(metric))
        test_results.append(create_test_result(api_tc))
    return test_results
