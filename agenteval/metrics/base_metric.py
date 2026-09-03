from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, List

from agenteval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    SingleTurnParams,
)
from agenteval.templates.render import filter_template_kwargs

if TYPE_CHECKING:
    from agenteval.models import DeepEvalBaseLLM


class PromptMixin:
    """Renders a metric prompt via the metric's ``evaluation_template``.

    Each metric sets ``evaluation_template`` to a Template class whose
    ``staticmethod`` methods return a rendered Jinja2 prompt. Subclass the
    Template and override any method to customize that single prompt only.
    """

    def _get_prompt(
        self,
        method: str,
        *,
        multimodal: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> str:
        context = {**kwargs, "multimodal": multimodal, "strict": strict}
        render = getattr(
            getattr(self, "evaluation_template", None), method, None
        )
        if render is None:
            raise ValueError(
                f"No prompt method `{method}` on "
                f"{self.__class__.__name__}'s evaluation_template; set "
                f"`evaluation_template` to a Template class that defines it."
            )
        return render(**filter_template_kwargs(render, context))


class BaseMetric(PromptMixin):
    _required_params = List[SingleTurnParams]
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    requires_trace: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


class BaseConversationalMetric(PromptMixin):
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @abstractmethod
    async def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Conversational Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens
