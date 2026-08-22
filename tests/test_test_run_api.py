import pytest

from agenteval.test_run import LLMApiTestCase, MetricData


def test_update_metric_data_aggregates_success_and_cost():
    tc = LLMApiTestCase(name="t", input="q")
    assert tc.success is None
    tc.update_metric_data(MetricData(name="m1", success=True, score=1.0, evaluationCost=0.1))
    tc.update_metric_data(MetricData(name="m2", success=False, score=0.2, evaluationCost=0.2))
    assert tc.success is False  # any False -> False
    assert tc.evaluation_cost == pytest.approx(0.3)
    assert len(tc.metrics_data) == 2


def test_flaky_metric_does_not_decide_success():
    tc = LLMApiTestCase(name="t", input="q")
    tc.update_metric_data(MetricData(name="flaky", success=False, score=0.0, flaky=True))
    assert tc.success is None  # flaky doesn't set case success
