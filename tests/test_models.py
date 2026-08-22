import pytest
from typing import Optional, Tuple, Union
from pydantic import BaseModel
from agenteval.models import DeepEvalBaseLLM, EvaluationCost


class MockModel(DeepEvalBaseLLM):
    """Returns canned (output, cost) so tests never hit a real API."""

    def __init__(self, output: str = "mock", schema_out=None, cost: float = 0.0):
        self._output = output
        self._schema_out = schema_out
        self._cost = cost
        super().__init__(model="mock-model")

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self, *args, **kwargs) -> str:
        return "mock-model"

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
        if schema is not None and self._schema_out is not None:
            return self._schema_out, EvaluationCost(self._cost, 10, 20)
        return self._output, EvaluationCost(self._cost, 10, 20)

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
        return self.generate(prompt, schema)


def test_mock_model_is_deepeval_base_llm():
    m = MockModel()
    assert isinstance(m, DeepEvalBaseLLM)


def test_mock_model_generate_contract():
    m = MockModel(output="hello")
    out, cost = m.generate("hi")
    assert out == "hello"
    assert cost.input_tokens == 10
    assert cost.output_tokens == 20


def test_openai_model_class_imports():
    # import only — do NOT instantiate (needs API key)
    from agenteval.models.llms import OpenAIModel
    assert OpenAIModel.__name__ == "OpenAIModel"
