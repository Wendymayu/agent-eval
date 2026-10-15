"""Answer relevance metric.

An LLM-judge that decides whether ``actual_output`` is relevant to ``input``:
does the output address what the input asks, or does it dodge / go off-topic?
The judge returns a *binary 0/1 verdict* plus a reason (single LLM call):
1 = relevant, 0 = not relevant. ``score >= threshold`` passes, where the
threshold is fixed at the coerce midpoint (0.5) and not user-tunable — tuning
a 0/1 score is a no-op.

This is the minimal *direct-scoring* variant. The original deepeval
``AnswerRelevancyMetric`` algorithm — break the output into statements, emit a
yes/no/idk verdict per statement, then aggregate — is intentionally NOT
ported here to keep this minimal. The per-statement algorithm
(``generate_statements`` / ``generate_verdicts`` / ``generate_reason``) is
intentionally not ported; add those prompts to ``template.py`` later if you
want the upgrade path.
"""

from typing import List, Optional, Tuple, Type

from agenteval.metrics import BaseMetric
from agenteval.metrics.answer_relevance.template import AnswerRelevanceTemplate
from agenteval.metrics.g_eval.schema import ReasonScore
from agenteval.metrics.indicator import metric_progress_indicator
from agenteval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
)
from agenteval.test_case import LLMTestCase, SingleTurnParams


class AnswerRelevanceMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[AnswerRelevanceTemplate] = AnswerRelevanceTemplate,
    ):
        # Binary verdict: the decision boundary is fixed at the coerce
        # midpoint (0.5) — not user-tunable, since tuning a 0/1 score is a
        # no-op. Kept non-None so the framework's "verdict-capable" signal
        # (check_at_least_one_metric_has_threshold) still treats this metric
        # as deciding pass/fail.
        self.threshold = 0.5
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()

    async def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            score, reason = await self._score(test_case)
            self._set_result(score, reason)
            return self.score

    def _set_result(self, score: float, reason: str) -> None:
        # Binary verdict: coerce whatever the judge returned to {0.0, 1.0} so
        # the score contract holds even if the judge slips a non-binary value.
        self.score = 1.0 if score >= 0.5 else 0.0
        self.reason = reason
        self.success = self.is_successful()
        if self.verbose_mode:
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Score: {self.score:.0f}",
                    f"Reason: {self.reason}",
                ],
            )

    def _prompt(self, test_case: LLMTestCase) -> str:
        return self._get_prompt(
            "generate_relevance_score",
            input=test_case.input,
            actual_output=test_case.actual_output,
            multimodal=test_case.multimodal,
        )

    async def _score(self, test_case: LLMTestCase) -> Tuple[float, str]:
        return await generate_with_schema_and_extract(
            metric=self,
            prompt=self._prompt(test_case),
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reason),
            extract_json=lambda d: (d["score"], d["reason"]),
        )

    @property
    def __name__(self):
        return "Answer Relevance"
