# agenteval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract deepeval v4.1.8's core four (TestCase / Metric / Model / evaluate) into a new minimal `agenteval` package — single-turn LLM eval working end-to-end, with conversational/tool-call data classes carried as the Agent extension slot.

**Architecture:** Faithful fork of deepeval's leaf components (TestCase, base_model, base_metric, OpenAIModel, GEval, ExactMatch, templates) with surgical removal of tracing/telemetry/confident/settings/cache; clean minimal rewrites of the deeply-entangled orchestration glue (evaluate, execution engine, configs, test_run API). pydantic v2 only. Internal class names stay `DeepEval*` (faithful); only the package path (`agenteval`) and root exception (`AgentEvalError`) are rebranded.

**Tech Stack:** Python ≥3.9, pydantic v2, openai SDK, jinja2, nest_asyncio, pytest (dev).

**Source repo:** `D:\code\opensource\github\deepeval` (read-only reference). **Target:** `D:\code\opensource\github\agent-eval`.

**Global rename rule (applied to EVERY copied file):** replace `deepeval` → `agenteval` in all imports/strings (`from deepeval` → `from agenteval`, `import deepeval` → `import agenteval`, `deepeval.` → `agenteval.`, and the resource string `"deepeval.templates"` → `"agenteval.templates"`). A helper script does this mechanically (Task 1).

---

## File Structure

**Written fresh (minimal):**
- `pyproject.toml`, `.gitignore`
- `agenteval/__init__.py` — public API
- `agenteval/errors.py` — `AgentEvalError`, `MissingTestCaseParamsError`
- `agenteval/utils.py` — `make_model_config`, `get_or_create_event_loop`, `prettify_list`, `check_if_multimodal`, `convert_to_multi_modal_array`
- `agenteval/metrics/indicator.py` — no-op `metric_progress_indicator`
- `agenteval/evaluate/configs.py` — `AsyncConfig`, `ErrorConfig`
- `agenteval/evaluate/types.py` — `TestResult`, `EvaluationResult`
- `agenteval/evaluate/utils.py` — `create_metric_data`, `create_test_result`, `validate_evaluate_inputs`, `check_valid_test_cases_type`
- `agenteval/evaluate/execute/e2e.py` — `execute_test_cases`, `a_execute_test_cases`
- `agenteval/evaluate/evaluate.py` — `evaluate()`
- `agenteval/test_run/api.py` — `LLMApiTestCase`, `ConversationalApiTestCase`, `TurnApi`, `MetricData`
- all `__init__.py` re-export shims

**Copied from deepeval + surgical edits:**
- `agenteval/test_case/llm_test_case.py`, `conversational_test_case.py`
- `agenteval/models/base_model.py`, `utils.py`, `llms/utils.py`, `llms/constants.py`, `llms/openai_model.py`
- `agenteval/metrics/base_metric.py`, `utils.py`, `g_eval/{g_eval,utils,schema}.py`, `exact_match/exact_match.py`
- `agenteval/templates/resolver.py`, `template_class.py`, `metrics/templates.json`

**Tests (fresh):** `tests/conftest.py` (MockModel), `tests/test_*.py` per layer, `tests/test_e2e.py`, `tests/test_no_optional_imports.py`.

---

## Task 1: Project scaffold + rename helper

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `agenteval/__init__.py`, `agenteval/errors.py`, `scripts/rename_deepeval.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agenteval"
version = "0.1.0"
description = "Minimal LLM/Agent evaluation framework (extracted from deepeval)"
requires-python = ">=3.9"
dependencies = [
    "pydantic>=2",
    "openai",
    "jinja2",
    "nest_asyncio",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
anthropic = ["anthropic"]
litellm = ["litellm"]

[tool.setuptools.packages.find]
include = ["agenteval*"]

[tool.setuptools.package-data]
agenteval = ["templates/metrics/templates.json"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.venv/
.env
```

- [ ] **Step 3: Write `agenteval/errors.py`** (root exception rebranded; keep deepeval error semantics for the ones the core raises)

```python
class AgentEvalError(Exception):
    """Base class for framework-originated errors."""


class MissingTestCaseParamsError(AgentEvalError):
    """Required test case fields are missing for a metric."""
    pass
```

- [ ] **Step 4: Write `scripts/rename_deepeval.py`** (mechanical rename used by copy tasks)

```python
"""Copy a deepeval source file into agenteval, rewriting deepeval->agenteval.

Usage: python scripts/rename_deepeval.py <src_abs_path> <dest_abs_path>
"""
import sys
from pathlib import Path

REPLACEMENTS = [
    ("from deepeval", "from agenteval"),
    ("import deepeval", "import agenteval"),
    ('"deepeval.', '"agenteval.'),
    ("deepeval.templates", "agenteval.templates"),
]


def main(src: str, dest: str) -> None:
    text = Path(src).read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # generic catch-all for any remaining bare `deepeval.` references
    text = text.replace("deepeval.", "agenteval.")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(text, encoding="utf-8")
    print(f"copied {src} -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 5: Write a stub `agenteval/__init__.py`** (filled in Task 10)

```python
"""agenteval — minimal LLM/Agent evaluation framework (extracted from deepeval)."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Install + smoke test**

Run: `pip install -e .`
Expected: installs `agenteval` editable.

Run: `python -c "import agenteval; print(agenteval.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffold agenteval package + rename helper"
```

---

## Task 2: Minimal utils helpers

**Files:**
- Create: `agenteval/utils.py`

- [ ] **Step 1: Write the failing test** `tests/test_utils.py`

```python
from agenteval.utils import (
    make_model_config,
    get_or_create_event_loop,
    prettify_list,
    check_if_multimodal,
    convert_to_multi_modal_array,
)


def test_make_model_config_returns_configdict():
    from pydantic import ConfigDict
    cfg = make_model_config(extra="ignore", arbitrary_types_allowed=True)
    assert isinstance(cfg, ConfigDict)


def test_get_or_create_event_loop_returns_loop():
    import asyncio
    loop = get_or_create_event_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)


def test_prettify_list_strings():
    assert prettify_list([]) == "[]"
    out = prettify_list(["a", "b"])
    assert '"a"' in out and '"b"' in out


def test_check_if_multimodal():
    assert check_if_multimodal("plain text") is False
    assert check_if_multimodal("img [DEEPEVAL:IMAGE:url] end") is True


def test_convert_to_multi_modal_array_plain_string():
    arr = convert_to_multi_modal_array("plain text")
    assert arr == ["plain text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL (ModuleNotFoundError: agenteval.utils).

- [ ] **Step 3: Write `agenteval/utils.py`** (minimal subset of deepeval/utils.py; v2-only, no tqdm/rich/settings)

```python
import asyncio
import json
import re
from typing import Any, List, Union

import nest_asyncio
from pydantic import BaseModel, ConfigDict


def make_model_config(**kwargs):
    """pydantic v2-only ConfigDict wrapper (replaces deepeval's dual-version)."""
    return ConfigDict(**kwargs)


def prettify_list(lst: List[Any]):
    if len(lst) == 0:
        return "[]"
    formatted_elements = []
    for item in lst:
        if isinstance(item, str):
            formatted_elements.append(f'"{item}"')
        elif isinstance(item, BaseModel):
            jsonObj = item.model_dump()
            formatted_elements.append(
                json.dumps(jsonObj, indent=4, ensure_ascii=True).replace(
                    "\n", "\n    "
                )
            )
        else:
            formatted_elements.append(repr(item))
    formatted_list = ",\n    ".join(formatted_elements)
    return f"[\n    {formatted_list}\n]"


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def convert_to_multi_modal_array(input: Union[str, List[str]]):
    from agenteval.test_case import MLLMImage

    if isinstance(input, str):
        return MLLMImage.parse_multimodal_string(input)
    elif isinstance(input, list):
        new_list = []
        for context in input:
            parsed_array = MLLMImage.parse_multimodal_string(context)
            new_list.extend(parsed_array)
        return new_list


def check_if_multimodal(input: str):
    pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]"
    matches = list(re.finditer(pattern, input))
    return bool(matches)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py -v`
Expected: PASS (5 tests). Note: `test_convert_to_multi_modal_array_plain_string` depends on `MLLMImage.parse_multimodal_string` from Task 3 — it will fail until Task 3; that's expected. Move it to `tests/test_test_case.py` in Task 3, or mark it `@pytest.mark.skip(reason="needs MLLMImage from Task 3")` for now. Keep the other 4 passing.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(utils): minimal helpers (config/loop/prettify/multimodal)"
```

---

## Task 3: TestCase layer

**Files:**
- Create: `agenteval/test_case/__init__.py`
- Copy+edit: `agenteval/test_case/llm_test_case.py` (from `deepeval/test_case/llm_test_case.py`)
- Copy+edit: `agenteval/test_case/conversational_test_case.py` (from `deepeval/test_case/conversational_test_case.py`)

- [ ] **Step 1: Copy both source files with rename**

Run:
```bash
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/test_case/llm_test_case.py agenteval/test_case/llm_test_case.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/test_case/conversational_test_case.py agenteval/test_case/conversational_test_case.py
```

- [ ] **Step 2: Surgical edit — collapse `make_model_config` to `ConfigDict`**

In `agenteval/test_case/llm_test_case.py`, find the import of `make_model_config` and its usage. Replace:

```python
from agenteval.utils import make_model_config
```
with nothing (remove the import), and replace every `make_model_config(...)` call with `ConfigDict(...)` adding `from pydantic import ConfigDict` at the top. Concretely, in `LLMTestCase`:

```python
# before
model_config = make_model_config(extra="ignore")
# after
model_config = ConfigDict(extra="ignore")
```

Apply the same `make_model_config(...)` → `ConfigDict(...)` replacement wherever it appears in either file (grep the files for `make_model_config` to find all sites). Add `from pydantic import ConfigDict` to each file's imports if not present.

- [ ] **Step 3: Surgical edit — the MCP/`get_or_create_event_loop` dependency**

`llm_test_case.py` and `conversational_test_case.py` import `mcp` helpers from `agenteval.test_case.mcp` (a module we are NOT porting in v1 — MCP is optional). The original `deepeval/test_case/__init__.py` exports `MCPServer` etc. To keep v1 minimal, stub the missing `mcp` module. Create `agenteval/test_case/mcp.py`:

```python
"""MCP support is an optional extension (not wired in v1).

These stubs let llm_test_case.py import cleanly. Populate from
deepeval/test_case/mcp.py when adding Agent/MCP evaluation.
"""
from typing import Any, List


class MCPServer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MCP support is not wired in agenteval v1.")


class MCPToolCall:
    pass


class MCPPromptCall:
    pass


class MCPResourceCall:
    pass


def normalize_mcp_servers(mcp_servers: Any) -> Any:
    return mcp_servers


def validate_mcp_servers(mcp_servers: Any) -> None:
    return None


def get_available_mcp_tool_names(*args, **kwargs):
    return []
```

Verify: `grep -n "from agenteval.test_case.mcp" agenteval/test_case/*.py` shows only the lazy/guarded imports the original already had — the stub satisfies them. (The original imports `mcp` lazily inside functions; the stub module existing is enough for the top-level `from agenteval.test_case.mcp import ...` lines if any. If a top-level import of a name NOT in the stub appears, add it to the stub.)

- [ ] **Step 4: Write `agenteval/test_case/__init__.py`** (subset of deepeval's; drop arena/mcp re-exports that need unported code)

```python
from agenteval.test_case.llm_test_case import (
    LLMTestCase,
    ToolCall,
    ToolCallType,
    ToolCallParams,
    SingleTurnParams,
    RetrievedContextData,
    MLLMImage,
)
from agenteval.test_case.conversational_test_case import (
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)

__all__ = [
    "LLMTestCase",
    "ToolCall",
    "ToolCallType",
    "ToolCallParams",
    "SingleTurnParams",
    "RetrievedContextData",
    "MLLMImage",
    "ConversationalTestCase",
    "Turn",
    "MultiTurnParams",
]
```

- [ ] **Step 5: Write the failing test** `tests/test_test_case.py`

```python
from agenteval.test_case import LLMTestCase, ConversationalTestCase, Turn, ToolCall


def test_llm_test_case_required_input_only():
    tc = LLMTestCase(input="What is 2+2?")
    assert tc.input == "What is 2+2?"
    assert tc.actual_output is None
    assert tc.expected_output is None


def test_llm_test_case_full_fields():
    tc = LLMTestCase(
        input="q",
        actual_output="4",
        expected_output="4",
        context=["2+2=4"],
        tools_called=[ToolCall(name="calc")],
        expected_tools=[ToolCall(name="calc")],
    )
    assert tc.tools_called[0].name == "calc"
    assert tc.multimodal is False


def test_conversational_test_case_turns():
    tc = ConversationalTestCase(
        turns=[Turn(role="user", content="hi"), Turn(role="assistant", content="hello")]
    )
    assert tc.turns[0].role == "user"
    assert len(tc.turns) == 2
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_test_case.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(test_case): port LLMTestCase + ConversationalTestCase (extension slot)"
```

---

## Task 4: Model layer (DeepEvalBaseLLM + OpenAIModel)

**Files:**
- Copy+edit: `agenteval/models/base_model.py`, `agenteval/models/utils.py`, `agenteval/models/llms/utils.py`, `agenteval/models/llms/constants.py`, `agenteval/models/llms/openai_model.py`
- Create: `agenteval/models/__init__.py`, `agenteval/models/llms/__init__.py`

- [ ] **Step 1: Copy the five source files with rename**

```bash
for pair in \
  "deepeval/models/base_model.py:agenteval/models/base_model.py" \
  "deepeval/models/utils.py:agenteval/models/utils.py" \
  "deepeval/models/llms/utils.py:agenteval/models/llms/utils.py" \
  "deepeval/models/llms/constants.py:agenteval/models/llms/constants.py" \
  "deepeval/models/llms/openai_model.py:agenteval/models/llms/openai_model.py"; do
  src="D:/code/opensource/github/deepeval/$pair"
  python scripts/rename_deepeval.py "${src%%:*}" "${src##*:}"
done
```

- [ ] **Step 2: Surgical edit — strip tracing from `base_model.py`**

In `agenteval/models/base_model.py`, the `DeepEvalBaseLLM.__init_subclass__` (lines ~51-66) imports `deepeval.tracing.internal.observe_methods` and wraps methods. Replace the whole method body with a no-op:

```python
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
```

(Delete the `from agenteval.tracing.internal import observe_methods` line and the `observe_methods(...)` call inside `DeepEvalBaseLLM.__init_subclass__`.) Leave `DeepEvalBaseModel` and `DeepEvalBaseEmbeddingModel` as-is (they have no tracing hook).

- [ ] **Step 3: Surgical edit — `openai_model.py`: strip settings, retry, tracing**

`openai_model.py` is the most-edited file. Apply these edits:

(a) Replace the import block (top of file) — remove settings/retry/constants-as-needed/tracing, point utils at `agenteval`:

```python
from openai.types.chat.chat_completion import ChatCompletion
from typing import Any, Optional, Tuple, Union, Dict, List
import os
import warnings
from agenteval.test_case import MLLMImage
from pydantic import BaseModel, SecretStr
from openai import (
    OpenAI,
    AsyncOpenAI,
)

from agenteval.errors import AgentEvalError
from agenteval.utils import check_if_multimodal, convert_to_multi_modal_array
from agenteval.models import DeepEvalBaseLLM
from agenteval.models.llms.utils import trim_and_load_json
from agenteval.models.utils import (
    parse_model_name,
    require_costs,
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
    EvaluationCost,
)
from agenteval.models.llms.constants import (
    DEFAULT_GPT_MODEL,
    OPENAI_MODELS_DATA,
)
```

(Removed: `from agenteval.tracing.context import update_llm_span, update_current_span`, `from agenteval.config.settings import get_settings`, `from agenteval.constants import ProviderSlug as PS`, `from agenteval.models.retry_policy import ...`, `retry_openai = create_retry_decorator(PS.OPENAI)`.)

(b) Replace `_request_timeout_seconds` and `_client_kwargs`/`_build_client` to drop settings/retry:

```python
def _request_timeout_seconds() -> float:
    timeout = float(os.getenv("AGENTEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "0") or 0)
    return timeout if timeout > 0 else 30.0


# inside OpenAIModel: replace _client_kwargs body
    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        kwargs["max_retries"] = 0
        if not kwargs.get("timeout"):
            kwargs["timeout"] = _request_timeout_seconds()
        return kwargs
```

(c) In `OpenAIModel.__init__`, replace every `settings.X` read with an env-var/arg default. Replace the body that reads `get_settings()` with:

```python
        # model + cost + api_key + temperature from args/env (no settings object)
        model = model or os.getenv("OPENAI_MODEL_NAME") or DEFAULT_GPT_MODEL
        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else _env_float("OPENAI_COST_PER_INPUT_TOKEN")
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else _env_float("OPENAI_COST_PER_OUTPUT_TOKEN")
        )
        if api_key is not None:
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            env_key = os.getenv("OPENAI_API_KEY")
            self.api_key = SecretStr(env_key) if env_key else None
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        if temperature is not None:
            temperature = float(temperature)
        elif os.getenv("TEMPERATURE") is not None:
            temperature = float(os.getenv("TEMPERATURE"))
        else:
            temperature = 0.0
        if isinstance(model, str):
            model = parse_model_name(model)
        self.model_data = OPENAI_MODELS_DATA.get(model)
        if getattr(self.model_data, "supports_temperature", True) is False:
            temperature = 1
        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "OPENAI_COST_PER_INPUT_TOKEN",
            "OPENAI_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )
        self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = cost_per_output_token
        if temperature < 0:
            raise AgentEvalError("Temperature must be >= 0.")
        self.temperature = temperature
        self.async_http_client = normalized_kwargs.pop("async_http_client", None)
        self.kwargs = normalized_kwargs
        self.kwargs.pop("temperature", None)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)
        super().__init__(model)
```

Add the helper near the top of the file (after imports):

```python
def _env_float(name: str):
    v = os.getenv(name)
    return float(v) if v is not None else None
```

(d) Replace `DeepEvalError` → `AgentEvalError` everywhere in this file (the rename script leaves `DeepEvalError` because it's a class name, not the `deepeval.` prefix). Run: `grep -n DeepEvalError agenteval/models/llms/openai_model.py` and replace each with `AgentEvalError`. (Also do this in `agenteval/models/llms/utils.py` and `agenteval/models/utils.py`: `grep -rn DeepEvalError agenteval/models/` then replace with `AgentEvalError`.)

(e) Make `_update_llm_span_from_completion` a no-op (strip tracing calls). Replace its body:

```python
    def _update_llm_span_from_completion(
        self,
        completion: ChatCompletion,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        return None
```

(f) Remove every `@retry_openai` decorator line above `generate`, `a_generate`, `generate_raw_response`, `a_generate_raw_response`, `generate_samples`.

- [ ] **Step 4: Verify `constants.py` has no heavy deps**

Run: `python -c "from agenteval.models.llms import constants"`
Expected: imports cleanly. If it errors on `from agenteval.config.settings import ...` or similar, remove that import line and hardcode the values it provided (the registry is pure data: `DeepEvalModelData` instances + `DEFAULT_GPT_MODEL = "gpt-4o"` if `DEFAULT_GPT_MODEL` referenced a settings default). Inspect `agenteval/models/llms/constants.py` top imports; strip any `settings`/`key_handler` import and replace with literal defaults.

- [ ] **Step 5: Write `agenteval/models/__init__.py`**

```python
from agenteval.models.base_model import (
    DeepEvalBaseModel,
    DeepEvalBaseLLM,
    DeepEvalModelData,
)
from agenteval.models.utils import EvaluationCost

__all__ = [
    "DeepEvalBaseModel",
    "DeepEvalBaseLLM",
    "DeepEvalModelData",
    "EvaluationCost",
]
```

- [ ] **Step 6: Write `agenteval/models/llms/__init__.py`**

```python
from agenteval.models.llms.openai_model import OpenAIModel
from agenteval.models.llms.utils import trim_and_load_json

__all__ = ["OpenAIModel", "trim_and_load_json"]
```

- [ ] **Step 7: Write the failing test** `tests/test_models.py` (MockModel — no API key needed)

```python
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
    from agenteval.models import DeepEvalBaseLLM
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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat(models): port DeepEvalBaseLLM + OpenAIModel (stripped settings/retry/tracing)"
```

---

## Task 5: Templates layer (jinja2 prompt bundle)

**Files:**
- Copy+edit: `agenteval/templates/resolver.py`, `agenteval/templates/template_class.py`, `agenteval/templates/__init__.py`
- Copy verbatim: `agenteval/templates/metrics/templates.json`

- [ ] **Step 1: Copy the data bundle verbatim**

```bash
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/templates/metrics/templates.json agenteval/templates/metrics/templates.json
```

- [ ] **Step 2: Copy + rename the three Python files**

```bash
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/templates/resolver.py agenteval/templates/resolver.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/templates/template_class.py agenteval/templates/template_class.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/templates/__init__.py agenteval/templates/__init__.py
```

- [ ] **Step 3: Surgical edit — fix the hardcoded resource package name in `resolver.py`**

The rename script turns `"deepeval.templates"` into `"agenteval.templates"` via the catch-all, but `resources.files("deepeval.templates")` becomes `resources.files("agenteval.templates")` — verify:

Run: `grep -n 'resources.files' agenteval/templates/resolver.py`
Expected: shows `resources.files("agenteval.templates")`. If it still says `deepeval`, fix it manually to `agenteval`.

- [ ] **Step 4: Write the failing test** `tests/test_templates.py`

```python
from agenteval.templates import make_template_class, resolve_template


def test_make_geval_template_class():
    cls = make_template_class("GEval")
    assert cls.__name__ == "GEvalTemplate"
    assert hasattr(cls, "generate_evaluation_steps")


def test_resolve_geval_step_template():
    out = resolve_template(
        "metrics",
        "GEval",
        "generate_evaluation_steps",
        criteria="is correct",
        parameters="input, actual_output",
        multimodal=False,
    )
    assert isinstance(out, str)
    assert "is correct" in out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_templates.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(templates): port jinja2 prompt bundle + resolver"
```

---

## Task 6: Metric base layer (BaseMetric + minimal indicator + utils)

**Files:**
- Copy+edit: `agenteval/metrics/base_metric.py`, `agenteval/metrics/utils.py`
- Create: `agenteval/metrics/indicator.py`, `agenteval/metrics/__init__.py`

- [ ] **Step 1: Copy base_metric.py + utils.py with rename**

```bash
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/base_metric.py agenteval/metrics/base_metric.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/utils.py agenteval/metrics/utils.py
```

- [ ] **Step 2: Surgical edit — strip tracing from `base_metric.py`**

`base_metric.py` defines `BaseMetric`, `BaseConversationalMetric`, `BaseArenaMetric`, each with an `__init_subclass__` that imports `observe_methods`. Apply:

(a) Replace each `__init_subclass__` body (3 occurrences) with the no-op:

```python
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
```

(b) Drop `BaseArenaMetric` entirely (it needs `ArenaTestCase`, which we are not porting) — delete the `class BaseArenaMetric(PromptMixin):` block (lines ~202-256 in the source) and remove `ArenaTestCase` from the top import:

```python
# change
from agenteval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    SingleTurnParams,
    ArenaTestCase,
)
# to
from agenteval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    SingleTurnParams,
)
```

- [ ] **Step 3: Surgical edit — rewrite `metrics/utils.py` (the chokepoint)**

`utils.py` blanket-imports every provider at top and references all of them in `initialize_model`/`is_native_model`/`MULTIMODAL_SUPPORTED_MODELS`. Replace the import header (lines ~17-83) with:

```python
import inspect
import json
import re
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from agenteval.errors import (
    MissingTestCaseParamsError,
)
from agenteval.utils import convert_to_multi_modal_array
from agenteval.models import (
    DeepEvalBaseLLM,
    EvaluationCost,
)
from agenteval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
)
from agenteval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ConversationalTestCase,
    MLLMImage,
    Turn,
    ToolCall,
    MultiTurnParams,
)
```

(Removed: all concrete provider imports, model-data constants, `key_handler`, `BaseArenaMetric`, `ArenaTestCase`, `DeepEvalBaseEmbeddingModel`, `get_settings`/`SETTINGS`.)

Then delete these now-unused functions entirely: `print_tools_called` (it called `serialize_to_json`), and the entire embedding section (`should_use_*_embedding`, `initialize_embedding_model`), and `check_arena_test_case_params`. Also delete `MULTIMODAL_SUPPORTED_MODELS = {...}` dict and simplify the multimodal branches in `check_llm_test_case_params` / `check_conversational_test_case_params` to:

```python
    if multimodal:
        if not model or not model.supports_multimodal():
            raise ValueError(
                f"The evaluation model {getattr(model, 'name', model)} does not "
                "support multimodal inputs."
            )
```

(replacing the two `if model and type(model) in MULTIMODAL_SUPPORTED_MODELS.keys(): ...` blocks in each function).

Then replace `initialize_model` and `is_native_model` with lazy OpenAI-only versions:

```python
def is_native_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> bool:
    from agenteval.models.llms.openai_model import OpenAIModel
    return isinstance(model, OpenAIModel)


def initialize_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """Return (model, using_native_model). Defaults to OpenAIModel."""
    if is_native_model(model):
        return model, True
    if isinstance(model, DeepEvalBaseLLM):
        return model, False
    # str or None -> default to OpenAI (lazy import keeps other providers out)
    from agenteval.models.llms.openai_model import OpenAIModel
    return OpenAIModel(model=model), True
```

Delete all the `should_use_*_model()` functions (they referenced the removed `SETTINGS`/`KEY_FILE_HANDLER`).

Keep unchanged: `check_at_least_one_metric_has_threshold`, `copy_metrics`, `format_turns`, `convert_turn_to_dict`, `get_turns_in_sliding_window`, `get_unit_interactions`, `print_verbose_logs`, `construct_verbose_logs`, `check_conversational_test_case_params`, `check_llm_test_case_params`, `trimAndLoadJson`, `accrue_token_usage`, `verdict_from_json`, `generate_with_schema_and_extract`, `a_generate_with_schema_and_extract`.

- [ ] **Step 4: Write `agenteval/metrics/indicator.py`** (minimal no-op — every metric wraps `measure()` in this CM)

```python
from contextlib import contextmanager
from typing import Optional


@contextmanager
def metric_progress_indicator(
    metric,
    async_mode: Optional[bool] = None,
    total: int = 9999,
    transient: bool = True,
    _show_indicator: bool = True,
    _in_component: bool = False,
):
    """No-op progress indicator (tracing/rich progress stripped in agenteval)."""
    yield
```

- [ ] **Step 5: Write `agenteval/metrics/__init__.py`**

```python
from agenteval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    PromptMixin,
)

__all__ = ["BaseMetric", "BaseConversationalMetric", "PromptMixin"]
```

(GEval / ExactMatchMetric are re-exported from here in Task 7 — update `__init__.py` then.)

- [ ] **Step 6: Write the failing test** `tests/test_base_metric.py`

```python
from agenteval.metrics import BaseMetric
from agenteval.test_case import LLMTestCase


class PassMetric(BaseMetric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        self.score = 1.0
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)


def test_subclass_does_not_require_tracing():
    # If __init_subclass__ still imports tracing, this raises at class def.
    m = PassMetric()
    assert m.__name__ == "Base Metric"


def test_is_successful_threshold_logic():
    m = PassMetric(threshold=0.5)
    m.measure(LLMTestCase(input="x"))
    assert m.is_successful() is True

    m2 = PassMetric(threshold=0.9)
    m2.measure(LLMTestCase(input="x"))
    assert m2.is_successful() is False  # 1.0 >= 0.9 is True actually
    # NOTE: 1.0 >= 0.9 is True -> fix expectation:
```

Fix the last assertion: `1.0 >= 0.9` is `True`, so change `test_is_successful_threshold_logic`'s second check to use `threshold=1.5`:

```python
    m2 = PassMetric(threshold=1.5)
    m2.measure(LLMTestCase(input="x"))
    assert m2.is_successful() is False
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_base_metric.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(metrics): port BaseMetric + minimal indicator + lazy initialize_model"
```

---

## Task 7: GEval + ExactMatchMetric

**Files:**
- Copy+edit: `agenteval/metrics/g_eval/g_eval.py`, `agenteval/metrics/g_eval/utils.py`
- Copy verbatim: `agenteval/metrics/g_eval/schema.py`, `agenteval/metrics/g_eval/__init__.py`, `agenteval/metrics/exact_match/exact_match.py`, `agenteval/metrics/exact_match/__init__.py`
- Modify: `agenteval/metrics/__init__.py` (add re-exports)

- [ ] **Step 1: Copy the g_eval + exact_match files**

```bash
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/g_eval/g_eval.py agenteval/metrics/g_eval/g_eval.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/g_eval/utils.py agenteval/metrics/g_eval/utils.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/g_eval/schema.py agenteval/metrics/g_eval/schema.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/g_eval/__init__.py agenteval/metrics/g_eval/__init__.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/exact_match/exact_match.py agenteval/metrics/exact_match/exact_match.py
python scripts/rename_deepeval.py D:/code/opensource/github/deepeval/deepeval/metrics/exact_match/__init__.py agenteval/metrics/exact_match/__init__.py
```

- [ ] **Step 2: Surgical edit — `g_eval/g_eval.py`: strip confident + settings**

(a) Remove these imports:
```python
from agenteval.config.settings import get_settings
from agenteval.confident.api import Api, Endpoints, HttpMethods
```

(b) Delete the `upload(self)` method and the `pull(self)` method entirely (they call Confident AI).

(c) In `measure(self, ...)` replace the async branch's settings-driven timeout:

```python
            if self.async_mode:
                from agenteval.utils import get_or_create_event_loop
                loop = get_or_create_event_loop()
                coro = self.a_measure(
                    test_case,
                    _show_indicator=False,
                    _in_component=_in_component,
                    _additional_context=_additional_context,
                )
                loop.run_until_complete(coro)
```

(Delete the `settings = get_settings()` and `asyncio.wait_for(coro, timeout=...)` lines; run the coro directly.)

(d) The imports `from agenteval.utils import get_or_create_event_loop, prettify_list` stay valid (Task 2 provides both). `from agenteval.metrics.indicator import metric_progress_indicator` resolves to the no-op (Task 6). `from agenteval.templates import make_template_class` resolves (Task 5).

- [ ] **Step 3: Surgical edit — `g_eval/utils.py`: strip confident/settings top imports**

Run: `grep -nE "from agenteval.(confident|config|telemetry|tracing)" agenteval/metrics/g_eval/utils.py`
For each hit: if the imported name is only used by the deleted `upload`/`pull` (e.g. `construct_geval_upload_payload`, `construct_geval_pull_evaluation_params`, `MetricPullResponse`, `G_EVAL_API_PARAMS`), either keep the function definitions (harmless, unused) but remove the broken top-level import, OR delete the unused functions. Simplest: remove the broken import lines; if a kept function references an unported symbol, delete that function too (it's unused after upload/pull removal). Functions that GEval's `measure` path actually needs: `Rubric`, `construct_g_eval_params_string`, `construct_test_case_string`, `format_rubrics`, `no_log_prob_support`, `calculate_weighted_summed_score`, `validate_and_sort_rubrics`, `validate_criteria_and_evaluation_steps`, `number_evaluation_steps`, `get_score_range`, `ensure_required_params` — keep these. Drop the upload/pull-only ones if they import unported modules.

- [ ] **Step 4: Update `agenteval/metrics/__init__.py`**

```python
from agenteval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    PromptMixin,
)
from agenteval.metrics.g_eval.g_eval import GEval
from agenteval.metrics.exact_match.exact_match import ExactMatchMetric

__all__ = [
    "BaseMetric",
    "BaseConversationalMetric",
    "PromptMixin",
    "GEval",
    "ExactMatchMetric",
]
```

- [ ] **Step 5: Write the failing test** `tests/test_metrics.py`

```python
import pytest
from pydantic import BaseModel
from agenteval.metrics import GEval, ExactMatchMetric
from agenteval.test_case import LLMTestCase
from agenteval.test_case.llm_test_case import SingleTurnParams
from tests.conftest import MockModel


def test_exact_match_pass():
    m = ExactMatchMetric(threshold=0.5)
    m.measure(LLMTestCase(input="q", actual_output="4", expected_output="4"))
    assert m.score == 1.0
    assert m.is_successful() is True


def test_exact_match_fail():
    m = ExactMatchMetric(threshold=0.5)
    m.measure(LLMTestCase(input="q", actual_output="5", expected_output="4"))
    assert m.score == 0.0
    assert m.is_successful() is False


def test_geval_scores_with_mock_model():
    # GEval's a_measure path: _a_generate_evaluation_steps + _a_evaluate both
    # call generate_with_schema_and_extract. Mock the model to return canned JSON.
    from agenteval.metrics.g_eval import schema as gschema
    # Build a mock that returns schema instances for the two calls.
    # First call -> Steps(steps=["step1"]); second call -> ReasonScore(score=3, reason="ok")
    class GEvalMock(MockModel):
        def __init__(self):
            self._calls = 0
            super().__init__(model="gmock")

        def generate(self, prompt, schema=None):
            self._calls += 1
            if self._calls == 1:
                return gschema.Steps(steps=["Compare actual to expected"]), 0.0
            return gschema.ReasonScore(score=3, reason="matches well"), 0.0

        async def a_generate(self, prompt, schema=None):
            return self.generate(prompt, schema)

    g = GEval(
        name="correctness",
        criteria="is the answer correct",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=GEvalMock(),
        async_mode=False,
    )
    tc = LLMTestCase(input="q", actual_output="4", expected_output="4")
    g.measure(tc)
    assert g.score is not None
    assert 0.0 <= g.score <= 1.0
    assert g.reason is not None
```

- [ ] **Step 6: Write `tests/conftest.py`** (shared MockModel for tests)

```python
from typing import Optional, Tuple, Union
from pydantic import BaseModel
from agenteval.models import DeepEvalBaseLLM, EvaluationCost


class MockModel(DeepEvalBaseLLM):
    """Canned-response model so tests never hit a real API."""

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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (3 tests). If `test_geval_scores_with_mock_model` fails because GEval's logprob path calls `generate_raw_response` (which MockModel lacks → AttributeError → falls back to `generate_with_schema_and_extract`), that fallback is the intended path and should work. If the fallback still fails, ensure `GEvalMock` inherits the `generate_raw_response`-absent behavior (it does — AttributeError is raised, caught at g_eval.py:338/407).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(metrics): port GEval (stripped confident) + ExactMatchMetric"
```

---

## Task 8: test_run API (LLMApiTestCase + MetricData)

**Files:**
- Create: `agenteval/test_run/api.py`, `agenteval/test_run/__init__.py`

- [ ] **Step 1: Copy `test_run/api.py` and the `MetricData` model from `tracing/api.py`**

`deepeval/evaluate/types.py` imports `MetricData` from `deepeval.test_run`, which re-exports it from `deepeval.tracing.api`. We are not porting tracing. Instead, define `MetricData` directly in `agenteval/test_run/api.py`.

Write `agenteval/test_run/api.py` (merge of `test_run/api.py` + `MetricData`, with `trace` field stripped and `make_model_config` → our utils):

```python
from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field

from agenteval.test_case import MLLMImage, ToolCall
from agenteval.utils import make_model_config


class MetricData(BaseModel):
    model_config = make_model_config(extra="ignore")

    name: str
    threshold: Optional[float] = None
    success: Optional[bool] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    strict_mode: Optional[bool] = Field(False, alias="strictMode")
    flaky: bool = False
    evaluation_model: Optional[str] = Field(None, alias="evaluationModel")
    error: Optional[str] = None
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    input_tokens: Optional[int] = Field(None, alias="inputTokenCount")
    output_tokens: Optional[int] = Field(None, alias="outputTokenCount")
    verbose_logs: Optional[str] = Field(None, alias="verboseLogs")


class LLMApiTestCase(BaseModel):
    name: str
    input: str
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_called: Optional[list] = Field(None, alias="toolsCalled")
    expected_tools: Optional[list] = Field(None, alias="expectedTools")
    token_cost: Optional[float] = Field(None, alias="tokenCost")
    completion_time: Optional[float] = Field(None, alias="completionTime")
    tags: Optional[List[str]] = Field(None)
    flaky: bool = False
    images_mapping: Optional[Dict[str, MLLMImage]] = Field(
        None, alias="imagesMapping"
    )
    success: Union[bool, None] = Field(None)
    metrics_data: Union[List[MetricData], None] = Field(
        None, alias="metricsData"
    )
    run_duration: Union[float, None] = Field(None, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    order: Union[int, None] = Field(None)
    metadata: Optional[Dict] = Field(None)
    comments: Optional[str] = Field(None)

    model_config = make_model_config(arbitrary_types_allowed=True)

    def update_metric_data(self, metric_data: MetricData):
        if self.metrics_data is None:
            self.metrics_data = [metric_data]
        else:
            self.metrics_data.append(metric_data)
        if not metric_data.flaky:
            if self.success is None:
                self.success = metric_data.success
            else:
                if metric_data.success is False:
                    self.success = False
        evaluationCost = metric_data.evaluation_cost
        if evaluationCost is None:
            return
        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost

    def update_run_duration(self, run_duration: float):
        self.run_duration = run_duration

    def update_status(self, success: Optional[bool], flaky: bool = False):
        if flaky:
            return
        if self.success is None:
            self.success = success
        else:
            if success is False:
                self.success = False


class TurnApi(BaseModel):
    role: str
    content: str
    order: int
    user_id: Optional[str] = Field(None, alias="userId")
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_called: Optional[List[ToolCall]] = Field(None, alias="toolsCalled")
    comments: Optional[str] = Field(None)


class ConversationalApiTestCase(BaseModel):
    name: str
    success: bool
    metrics_data: List[MetricData] = Field(alias="metricsData")
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    turns: List[TurnApi] = Field(default_factory=lambda: [])
    order: Union[int, None] = Field(None)
    scenario: Optional[str] = Field(None)
    expected_outcome: Optional[str] = Field(None, alias="expectedOutcome")
    user_description: Optional[str] = Field(None, alias="userDescription")
    context: Optional[list] = Field(None)
    comments: Optional[str] = Field(None)
    metadata: Optional[Dict] = Field(None)
    images_mapping: Optional[Dict[str, MLLMImage]] = Field(
        None, alias="imagesMapping"
    )
    tags: Optional[List[str]] = Field(None)
    flaky: bool = False

    def update_metric_data(self, metrics_data: MetricData):
        if self.metrics_data is None:
            self.metrics_data = [metrics_data]
        else:
            self.metrics_data.append(metrics_data)
        if metrics_data.success is False and not metrics_data.flaky:
            self.success = False
        evaluationCost = metrics_data.evaluation_cost
        if evaluationCost is None:
            return
        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost

    def update_run_duration(self, run_duration: float):
        self.run_duration += run_duration
```

- [ ] **Step 2: Write `agenteval/test_run/__init__.py`**

```python
from agenteval.test_run.api import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    TurnApi,
    MetricData,
)

__all__ = [
    "LLMApiTestCase",
    "ConversationalApiTestCase",
    "TurnApi",
    "MetricData",
]
```

- [ ] **Step 3: Write the failing test** `tests/test_test_run_api.py`

```python
from agenteval.test_run import LLMApiTestCase, MetricData


def test_update_metric_data_aggregates_success_and_cost():
    tc = LLMApiTestCase(name="t", input="q")
    assert tc.success is None
    tc.update_metric_data(MetricData(name="m1", success=True, score=1.0, evaluation_cost=0.1))
    tc.update_metric_data(MetricData(name="m2", success=False, score=0.2, evaluation_cost=0.2))
    assert tc.success is False  # any False -> False
    assert tc.evaluation_cost == 0.3
    assert len(tc.metrics_data) == 2


def test_flaky_metric_does_not_decide_success():
    tc = LLMApiTestCase(name="t", input="q")
    tc.update_metric_data(MetricData(name="flaky", success=False, score=0.0, flaky=True))
    assert tc.success is None  # flaky doesn't set case success
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_test_run_api.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(test_run): port LLMApiTestCase + MetricData (stripped tracing)"
```

---

## Task 9: evaluate layer (configs + types + utils + engine + evaluate)

**Files:**
- Create: `agenteval/evaluate/configs.py`, `agenteval/evaluate/types.py`, `agenteval/evaluate/utils.py`, `agenteval/evaluate/execute/e2e.py`, `agenteval/evaluate/execute/__init__.py`, `agenteval/evaluate/evaluate.py`, `agenteval/evaluate/__init__.py`

- [ ] **Step 1: Write `agenteval/evaluate/configs.py`** (minimal; no test_run dep)

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class AsyncConfig:
    run_async: bool = True
    throttle_value: float = 0
    max_concurrent: int = 20

    def __post_init__(self):
        if self.max_concurrent < 1:
            raise ValueError("'max_concurrent' must be at least 1")
        if self.throttle_value < 0:
            raise ValueError("'throttle_value' must be at least 0")


@dataclass
class ErrorConfig:
    ignore_errors: bool = True
    skip_on_missing_params: bool = False
```

(`ignore_errors=True` default = a metric error sets `metric.error`/`success=False` and the run continues — the batch-eval-safe behavior from deepeval's `_execute_metric`.)

- [ ] **Step 2: Write `agenteval/evaluate/types.py`**

```python
from typing import Optional, List, Union, Dict
from dataclasses import dataclass

from agenteval.test_run.api import MetricData, TurnApi
from agenteval.test_case import MLLMImage


@dataclass
class TestResult:
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
```

(Plain class instead of pydantic BaseModel — avoids the `TestRun` import deepeval's version had; `TestResult` stays a dataclass like the original.)

- [ ] **Step 3: Write `agenteval/evaluate/utils.py`** (minimal: create_metric_data, create_test_result, validate inputs; copied from deepeval/evaluate/utils.py, tracing/reporting funcs dropped)

```python
from typing import Optional, List, Union

from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.test_run import LLMApiTestCase, ConversationalApiTestCase, MetricData
from agenteval.evaluate.types import TestResult


def create_metric_data(metric: BaseMetric) -> MetricData:
    if metric.error is not None:
        return MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=metric.is_successful(),
            strictMode=metric.strict_mode,
            flaky=metric.flaky,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            inputTokenCount=metric.input_tokens,
            outputTokenCount=metric.output_tokens,
            verboseLogs=metric.verbose_logs,
        )
    return MetricData(
        name=metric.__name__,
        score=metric.score,
        threshold=metric.threshold,
        reason=metric.reason,
        success=metric.is_successful(),
        strictMode=metric.strict_mode,
        flaky=metric.flaky,
        evaluationModel=metric.evaluation_model,
        error=None,
        evaluationCost=metric.evaluation_cost,
        inputTokenCount=metric.input_tokens,
        outputTokenCount=metric.output_tokens,
        verboseLogs=metric.verbose_logs,
    )


def create_test_result(
    api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
) -> TestResult:
    name = api_test_case.name
    index = api_test_case.order
    if isinstance(api_test_case, ConversationalApiTestCase):
        return TestResult(
            name=name,
            success=api_test_case.success,
            metrics_data=api_test_case.metrics_data,
            conversational=True,
            index=index,
            metadata=api_test_case.metadata,
            turns=api_test_case.turns,
        )
    return TestResult(
        name=name,
        success=api_test_case.success,
        metrics_data=api_test_case.metrics_data,
        input=api_test_case.input,
        actual_output=api_test_case.actual_output,
        expected_output=api_test_case.expected_output,
        context=api_test_case.context,
        retrieval_context=api_test_case.retrieval_context,
        conversational=False,
        index=index,
        multimodal=bool(api_test_case.images_mapping),
        metadata=api_test_case.metadata,
    )


def check_valid_test_cases_type(test_cases):
    if not isinstance(test_cases, list):
        raise ValueError("test_cases must be a list.")
    for tc in test_cases:
        if not isinstance(tc, (LLMTestCase, ConversationalTestCase)):
            raise TypeError(
                f"Invalid test_case type: {type(tc).__name__}. "
                "Expected LLMTestCase or ConversationalTestCase."
            )


def validate_evaluate_inputs(test_cases, metrics):
    if metrics is None:
        raise ValueError("You must provide 'metrics'.")
    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("'metrics' must be a non-empty list.")
    for test_case in test_cases:
        for metric in metrics:
            if isinstance(test_case, LLMTestCase) and not isinstance(
                metric, BaseMetric
            ):
                raise ValueError(
                    f"Metric {metric.__name__} is not a valid metric for LLMTestCase."
                )
            if isinstance(test_case, ConversationalTestCase) and not isinstance(
                metric, BaseConversationalMetric
            ):
                raise ValueError(
                    f"Metric {metric.__name__} is not a valid metric for ConversationalTestCase."
                )
```

- [ ] **Step 4: Write `agenteval/evaluate/execute/e2e.py`** (minimal engine; faithful error handling from deepeval's `_execute_metric`/`safe_a_measure`)

```python
import asyncio
from typing import List, Optional, Union

from agenteval.errors import MissingTestCaseParamsError
from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
from agenteval.evaluate.utils import create_metric_data, create_test_result
from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.metrics.utils import copy_metrics
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.test_run import LLMApiTestCase, ConversationalApiTestCase


def _handle_missing_params(metric, e: MissingTestCaseParamsError, ec: ErrorConfig):
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


def _run_metric_sync(metric, test_case, ec: ErrorConfig):
    try:
        metric.measure(test_case, _show_indicator=False)
    except MissingTestCaseParamsError as e:
        if _handle_missing_params(metric, e, ec):
            return
    except TypeError:
        try:
            metric.measure(test_case)
        except MissingTestCaseParamsError as e:
            if _handle_missing_params(metric, e, ec):
                return
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


async def _run_metric_async(metric, test_case, ec: ErrorConfig):
    try:
        await metric.a_measure(test_case, _show_indicator=False)
    except MissingTestCaseParamsError as e:
        if _handle_missing_params(metric, e, ec):
            return
    except TypeError:
        try:
            await metric.a_measure(test_case)
        except MissingTestCaseParamsError as e:
            if _handle_missing_params(metric, e, ec):
                return
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


def _build_api_test_case(test_case: LLMTestCase, index: int) -> LLMApiTestCase:
    return LLMApiTestCase(
        name=test_case.name or str(test_case.input)[:50] or f"TestCase_{index}",
        input=test_case.input,
        actual_output=test_case.actual_output,
        expected_output=test_case.expected_output,
        context=test_case.context,
        retrieval_context=test_case.retrieval_context,
        tools_called=test_case.tools_called,
        expected_tools=test_case.expected_tools,
        metadata=test_case.metadata,
        tags=test_case.tags,
        flaky=test_case.flaky,
        order=index,
    )


def execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    error_config: ErrorConfig,
) -> List:
    test_results = []
    for idx, tc in enumerate(test_cases):
        metrics_copy = copy_metrics(metrics)
        api_tc = _build_api_test_case(tc, idx)
        for metric in metrics_copy:
            _run_metric_sync(metric, tc, error_config)
            api_tc.update_metric_data(create_metric_data(metric))
        test_results.append(create_test_result(api_tc))
    return test_results


async def a_execute_test_cases(
    test_cases: List[LLMTestCase],
    metrics: List[BaseMetric],
    async_config: AsyncConfig,
    error_config: ErrorConfig,
) -> List:
    sem = asyncio.Semaphore(async_config.max_concurrent)

    async def run_one(idx, tc):
        async with sem:
            metrics_copy = copy_metrics(metrics)
            api_tc = _build_api_test_case(tc, idx)
            await asyncio.gather(
                *[_run_metric_async(m, tc, error_config) for m in metrics_copy]
            )
            for metric in metrics_copy:
                api_tc.update_metric_data(create_metric_data(metric))
            return create_test_result(api_tc)

    return await asyncio.gather(
        *[run_one(i, tc) for i, tc in enumerate(test_cases)]
    )
```

- [ ] **Step 5: Write `agenteval/evaluate/execute/__init__.py`**

```python
from agenteval.evaluate.execute.e2e import (
    execute_test_cases,
    a_execute_test_cases,
)

__all__ = ["execute_test_cases", "a_execute_test_cases"]
```

- [ ] **Step 6: Write `agenteval/evaluate/evaluate.py`** (minimal; stripped reporting/persistence/cloud)

```python
from typing import List, Optional, Union

from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
from agenteval.evaluate.execute import execute_test_cases, a_execute_test_cases
from agenteval.evaluate.types import EvaluationResult
from agenteval.evaluate.utils import (
    check_valid_test_cases_type,
    validate_evaluate_inputs,
)
from agenteval.metrics import BaseMetric, BaseConversationalMetric
from agenteval.metrics.utils import check_at_least_one_metric_has_threshold
from agenteval.test_case import LLMTestCase, ConversationalTestCase
from agenteval.utils import get_or_create_event_loop


def evaluate(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    metrics: Optional[
        Union[List[BaseMetric], List[BaseConversationalMetric]]
    ] = None,
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> EvaluationResult:
    check_valid_test_cases_type(test_cases)
    validate_evaluate_inputs(test_cases, metrics)
    check_at_least_one_metric_has_threshold(metrics)

    if async_config.run_async:
        loop = get_or_create_event_loop()
        test_results = loop.run_until_complete(
            a_execute_test_cases(
                test_cases,
                metrics,
                async_config=async_config,
                error_config=error_config,
            )
        )
    else:
        test_results = execute_test_cases(
            test_cases, metrics, error_config=error_config
        )

    return EvaluationResult(
        test_results=test_results,
        confident_link=None,
        test_run_id=None,
    )
```

- [ ] **Step 7: Write `agenteval/evaluate/__init__.py`**

```python
from agenteval.evaluate.evaluate import evaluate
from agenteval.evaluate.types import EvaluationResult, TestResult
from agenteval.evaluate.configs import AsyncConfig, ErrorConfig

__all__ = ["evaluate", "EvaluationResult", "TestResult", "AsyncConfig", "ErrorConfig"]
```

- [ ] **Step 8: Write the failing test** `tests/test_evaluate.py`

```python
import asyncio
from agenteval.evaluate import evaluate, EvaluationResult
from agenteval.metrics import ExactMatchMetric
from agenteval.test_case import LLMTestCase


def test_evaluate_sync_returns_evaluation_result():
    cases = [
        LLMTestCase(input="q", actual_output="4", expected_output="4"),
        LLMTestCase(input="q", actual_output="5", expected_output="4"),
    ]
    metrics = [ExactMatchMetric(threshold=0.5)]
    from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
    result = evaluate(cases, metrics, async_config=AsyncConfig(run_async=False), error_config=ErrorConfig())
    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 2
    assert result.test_results[0].success is True
    assert result.test_results[1].success is False
    assert result.test_results[0].metrics_data[0].name == "Exact Match"


def test_evaluate_async_returns_evaluation_result():
    cases = [LLMTestCase(input="q", actual_output="4", expected_output="4")] * 3
    metrics = [ExactMatchMetric(threshold=0.5)]
    from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
    result = evaluate(cases, metrics, async_config=AsyncConfig(run_async=True), error_config=ErrorConfig())
    assert len(result.test_results) == 3
    assert all(r.success for r in result.test_results)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat(evaluate): minimal evaluate() + engine (stripped reporting/cloud/test_run)"
```

---

## Task 10: Public API + end-to-end (GEval + MockModel)

**Files:**
- Modify: `agenteval/__init__.py`

- [ ] **Step 1: Write the public API** `agenteval/__init__.py`

```python
"""agenteval — minimal LLM/Agent evaluation framework (extracted from deepeval)."""

__version__ = "0.1.0"

from agenteval.errors import AgentEvalError
from agenteval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
)
from agenteval.metrics import BaseMetric, BaseConversationalMetric, GEval, ExactMatchMetric
from agenteval.models import DeepEvalBaseLLM
from agenteval.models.llms import OpenAIModel
from agenteval.evaluate import evaluate, EvaluationResult, TestResult, AsyncConfig, ErrorConfig

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
    "OpenAIModel",
    "evaluate",
    "EvaluationResult",
    "TestResult",
    "AsyncConfig",
    "ErrorConfig",
    "__version__",
]
```

- [ ] **Step 2: Write the end-to-end test** `tests/test_e2e.py`

```python
from pydantic import BaseModel
import agenteval
from agenteval import (
    evaluate,
    LLMTestCase,
    GEval,
    DeepEvalBaseLLM,
    EvaluationCost,
)
from agenteval.test_case.llm_test_case import SingleTurnParams
from agenteval.metrics.g_eval import schema as gschema


class GEvalMock(DeepEvalBaseLLM):
    """Two-call mock: steps then score."""

    def __init__(self):
        self._calls = 0
        super().__init__(model="gmock")

    def load_model(self, *a, **k):
        return None

    def get_model_name(self, *a, **k):
        return "gmock"

    def generate(self, prompt, schema=None):
        self._calls += 1
        if self._calls == 1:
            return gschema.Steps(steps=["Compare actual to expected"]), EvaluationCost(0.0, 5, 5)
        return gschema.ReasonScore(score=4, reason="correct"), EvaluationCost(0.0, 8, 8)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def test_public_api_exports():
    for name in ["evaluate", "LLMTestCase", "GEval", "DeepEvalBaseLLM", "OpenAIModel"]:
        assert hasattr(agenteval, name)


def test_e2e_geval_evaluate_with_mock_model():
    g = GEval(
        name="correctness",
        criteria="the answer is correct",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=GEvalMock(),
        threshold=0.5,
        async_mode=False,
    )
    cases = [
        LLMTestCase(input="2+2=?", actual_output="4", expected_output="4"),
    ]
    result = evaluate(cases, [g])
    assert isinstance(result, EvaluationResult)
    tr = result.test_results[0]
    assert tr.metrics_data[0].name.startswith("correctness")
    assert tr.metrics_data[0].score is not None
    assert tr.metrics_data[0].evaluation_model == "gmock"
    assert tr.metrics_data[0].input_tokens == 13  # 5 + 8
```

- [ ] **Step 3: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS, including `tests/test_e2e.py`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: public API + end-to-end GEval evaluation with mock model"
```

---

## Task 11: Surgical-change verification (no optional imports)

**Files:**
- Create: `tests/test_no_optional_imports.py`

This task verifies the "外科剥离" guarantees: importing agenteval / running evaluate does NOT pull in tracing, telemetry, confident, opentelemetry, or posthog.

- [ ] **Step 1: Write the failing test**

```python
import importlib
import sys

BANNED_MODULES = [
    "agenteval.tracing",
    "agenteval.telemetry",
    "agenteval.confident",
    "opentelemetry",
    "posthog",
    "agenteval.test_run.test_run",  # the heavy persistence manager
]


def _fresh_import(name):
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    return importlib.import_module(name)


def test_import_agenteval_does_not_load_banned():
    _fresh_import("agenteval")
    for banned in BANNED_MODULES:
        assert banned not in sys.modules, f"{banned} was imported by agenteval"


def test_evaluate_does_not_load_banned():
    _fresh_import("agenteval")
    from agenteval import evaluate, LLMTestCase, ExactMatchMetric
    from agenteval.evaluate.configs import AsyncConfig, ErrorConfig
    evaluate(
        [LLMTestCase(input="q", actual_output="4", expected_output="4")],
        [ExactMatchMetric(threshold=0.5)],
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(),
    )
    for banned in BANNED_MODULES:
        assert banned not in sys.modules, f"{banned} was imported during evaluate()"


def test_initialize_model_does_not_import_other_providers():
    _fresh_import("agenteval")
    from agenteval.metrics.utils import initialize_model
    # default path (None) must use OpenAI only — anthropic etc. NOT imported
    # We can't call OpenAIModel() without a key, so just assert the module
    # isn't imported at module-load of metrics.utils.
    import agenteval.metrics.utils as u
    assert "agenteval.models.llms.anthropic_model" not in sys.modules
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_no_optional_imports.py -v`
Expected: PASS (3 tests). If `test_evaluate_does_not_load_banned` fails because some copied file still imports tracing/telemetry at top level, find and strip that import (the surgical tasks should have removed them; this test is the safety net).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: verify no tracing/telemetry/confident imports leak in"
```

---

## Self-Review (completed before handoff)

**Spec coverage:**
- TestCase (§3) → Task 3 ✓
- Metric base + GEval + ExactMatch (§3,§5) → Tasks 6-7 ✓
- Model base + OpenAIModel (§3,§5) → Task 4 ✓
- evaluate + engine + result types (§3,§5) → Tasks 8-9 ✓
- Surgical: tracing strip (§4 #1) → Task 4 base_model, Task 6 base_metric ✓
- Surgical: lazy provider (§4 #2) → Task 6 metrics/utils ✓
- Surgical: pydantic v2 only (§4 #3) → Task 2 make_model_config, Task 3 ConfigDict ✓
- Surgical: evaluate strip (§4 #4) → Task 9 ✓
- Surgical: telemetry (§4 #5) → Task 9 (no posthog import) + Task 11 verifies ✓
- Error handling (§6) → Task 9 `_run_metric_*` ✓
- Testing strategy incl MockModel (§7) → Task 4/7 conftest ✓
- Agent extension slot (§8) → Task 3 ConversationalTestCase/Turn ✓
- Deps (§9) → Task 1 pyproject ✓
- Public API (§10) → Task 10 ✓

**Placeholder scan:** None — every code step shows full code or exact copy-path + exact edits. The few "grep then replace" instructions name the exact symbol. `tests/test_base_metric.py` Step 6 had an incorrect assertion that is corrected inline in the same step.

**Type consistency:** `MetricData` field aliases (`strictMode`, `evaluationCost`, `inputTokenCount`...) match between `test_run/api.py` (Task 8) and `create_metric_data` (Task 9). `LLMApiTestCase.update_metric_data(MetricData)` signature matches engine usage (Task 9). `evaluate()` signature matches the public API doc (§10) minus `metric_collection`/`cache_config`/`display_config` (intentionally stripped per spec §4). `DeepEvalBaseLLM.generate` returns `Tuple[Union[str, BaseModel], float]` — MockModel + OpenAIModel both honor it.

**Known simplifications (acceptable per spec):** (1) `_build_api_test_case` maps LLMTestCase→LLMApiTestCase fields directly (deepeval's mapper had images_mapping handling; v1 covers the common case, multimodal left as edge). (2) `EvaluationResult` is a plain class, not pydantic (avoids the unported `TestRun` import). (3) `assert_test`/pytest integration is deferred (spec §10 open item). (4) `BaseArenaMetric`/`ArenaTestCase`/`compare()` dropped (not in scope). (5) MCP stubbed (Task 3) — Agent/MCP evaluation is the v2 extension.
