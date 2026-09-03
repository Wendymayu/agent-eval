"""AI tone (AI-flavor) metric.

An LLM-judge that scores how strongly ``actual_output`` reads as
machine-generated — its "AI flavor": templated transitions (首先/其次/最后),
boilerplate (综上所述/值得注意的是), inflated wording, hollow summaries,
and unnatural evenness. The judge returns a single 0-1 score plus a reason
(single LLM call).

Score direction is inverted vs. most metrics: ``score`` is the *AI-flavor
density* (0.0 = reads as naturally human, 1.0 = saturated with AI tone), so
higher is *worse*. ``is_successful`` is overridden so a case passes when
``score <= threshold`` — i.e. ``threshold`` is the tolerable AI-flavor ceiling,
not a bar to clear.
"""

from typing import List, Optional, Tuple, Type

from agenteval.metrics import BaseMetric
from agenteval.metrics.ai_tone.template import AIToneTemplate
from agenteval.metrics.g_eval.schema import ReasonScore
from agenteval.metrics.indicator import metric_progress_indicator
from agenteval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
)
from agenteval.test_case import LLMTestCase, SingleTurnParams


class AIToneMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        threshold: float = 0.5,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[AIToneTemplate] = AIToneTemplate,
    ):
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()

    def is_successful(self) -> Optional[bool]:
        # AI-flavor is a "lower is better" metric: score is the AI-flavor
        # density, so a case passes when score does not exceed threshold.
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score <= self.threshold
            except TypeError:
                self.success = False
        return self.success

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
        self.score = float(score)
        self.reason = reason
        self.success = self.is_successful()
        if self.verbose_mode:
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Score: {self.score:.2f}",
                    f"Reason: {self.reason}",
                ],
            )

    def _prompt(self, test_case: LLMTestCase) -> str:
        return self._get_prompt(
            "generate_ai_tone_score",
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
        return "AI Tone"
