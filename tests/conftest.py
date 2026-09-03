from typing import Optional, Tuple, Union
from pydantic import BaseModel
from agenteval.models import DeepEvalBaseLLM, EvaluationCost


class MockModel(DeepEvalBaseLLM):
    """Canned-response model so tests never hit a real API.

    Returns ``(output, EvaluationCost)`` tuples for both the schema and plain
    paths, matching the contract ``generate_with_schema_and_extract`` unpacks.
    Not registered as a pytest fixture; import it explicitly where needed.
    """

    def __init__(self, output: str = "mock", schema_out=None, cost: float = 0.0):
        self._output = output
        self._schema_out = schema_out
        self._cost = cost
        super().__init__(model="mock-model")

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self, *args, **kwargs) -> str:
        return "mock-model"

    async def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        if schema is not None and self._schema_out is not None:
            return self._schema_out, EvaluationCost(self._cost, 10, 20)
        return self._output, EvaluationCost(self._cost, 10, 20)
