"""agenteval — minimal LLM/Agent evaluation framework (extracted from deepeval)."""

__version__ = "0.1.0"

from agenteval.errors import AgentEvalError
from agenteval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
)
from agenteval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    GEval,
    ExactMatchMetric,
)
from agenteval.models import DeepEvalBaseLLM, EvaluationCost
from agenteval.models.llms import OpenAIModel
from agenteval.evaluate import (
    evaluate,
    EvaluationResult,
    TestResult,
    AsyncConfig,
    ErrorConfig,
)

__all__ = [
    "AgentEvalError",
    "LLMTestCase",
    "ConversationalTestCase",
    "Turn",
    "ToolCall",
    "BaseMetric",
    "BaseConversationalMetric",
    "GEval",
    "ExactMatchMetric",
    "DeepEvalBaseLLM",
    "EvaluationCost",
    "OpenAIModel",
    "evaluate",
    "EvaluationResult",
    "TestResult",
    "AsyncConfig",
    "ErrorConfig",
    "__version__",
]
