from typing import Optional, List, Union, Dict
from dataclasses import dataclass

from agenteval.test_run.api import MetricData, TurnApi
from agenteval.test_case import MLLMImage


@dataclass
class TestResult:
    """A single evaluated test case's outcome.

    Mirrors deepeval's ``TestResult`` but is a plain dataclass: the
    orchestration layer no longer persists it through a TestRun manager.
    """

    __test__ = False
    name: str
    success: bool
    metrics_data: Union[List[MetricData], None]
    conversational: bool
    index: Optional[int] = None
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    turns: Optional[List[TurnApi]] = None
    metadata: Optional[Dict] = None


class EvaluationResult:
    """The aggregate result returned by :func:`evaluate`.

    ``confident_link``/``test_run_id`` are kept on the shape (deepeval's
    ``EvaluationResult`` exposes them) but are always ``None`` here:
    reporting, persistence, and Confident cloud are stripped in agenteval.
    """

    def __init__(
        self,
        test_results: List[TestResult],
        confident_link: Optional[str] = None,
        test_run_id: Optional[str] = None,
    ):
        self.test_results = test_results
        self.confident_link = confident_link
        self.test_run_id = test_run_id

    def __repr__(self) -> str:
        n = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.success)
        return f"EvaluationResult(total={n}, passed={passed})"
