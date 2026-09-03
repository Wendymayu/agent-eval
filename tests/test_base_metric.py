from agenteval.metrics import BaseMetric
from agenteval.test_case import LLMTestCase


class PassMetric(BaseMetric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    async def measure(self, test_case, *args, **kwargs):
        self.score = 1.0
        self.success = self.is_successful()
        return self.score


def test_subclass_does_not_require_tracing():
    # If __init_subclass__ still imports tracing, this raises at class def.
    m = PassMetric()
    assert m.__name__ == "Base Metric"


async def test_is_successful_threshold_logic():
    m = PassMetric(threshold=0.5)
    await m.measure(LLMTestCase(input="x"))
    assert m.is_successful() is True

    m2 = PassMetric(threshold=1.5)
    await m2.measure(LLMTestCase(input="x"))
    assert m2.is_successful() is False  # 1.0 >= 1.5 is False


def test_initialize_model_lazy_openai_only(monkeypatch):
    # OpenAIModel eagerly builds its client in __init__ (DeepEvalBaseLLM.__init__
    # -> load_model -> require_secret_api_key), so a dummy key is required to
    # construct it. The OpenAI SDK client builds with a dummy key without any
    # network call. See Task 4 note: "do NOT instantiate (needs API key)".
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test-key")
    import sys

    from agenteval.metrics.utils import initialize_model, is_native_model

    # default path (None) must use OpenAI only — no other provider imported
    model, native = initialize_model(None)
    assert native is True
    from agenteval.models.llms.openai_model import OpenAIModel

    assert isinstance(model, OpenAIModel)
    # no anthropic/azure/etc module loaded
    assert "agenteval.models.llms.anthropic_model" not in sys.modules
