import os
from openai.types.chat.chat_completion import ChatCompletion
from typing import Any, Optional, Tuple, Union, Dict, List
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


def _request_timeout_seconds() -> float:
    timeout = float(os.getenv("AGENTEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "0") or 0)
    return timeout if timeout > 0 else 30.0


def _env_float(name: str):
    v = os.getenv(name)
    return float(v) if v is not None else None


_ALIAS_MAP = {
    "api_key": ["_openai_api_key"],
}


class OpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "OpenAIModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]

        model = model or os.getenv("OPENAI_MODEL_NAME")
        if model is None:
            model = DEFAULT_GPT_MODEL

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
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = (
                SecretStr(os.getenv("OPENAI_API_KEY"))
                if os.getenv("OPENAI_API_KEY")
                else None
            )

        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        # args and kwargs will be passed to the underlying model, in load_model function

        if temperature is not None:
            temperature = float(temperature)
        else:
            env_temperature = _env_float("TEMPERATURE")
            temperature = env_temperature if env_temperature is not None else 0.0

        if isinstance(model, str):
            model = parse_model_name(model)

        self.model_data = OPENAI_MODELS_DATA.get(model)

        # Auto-adjust temperature for known models that require it
        if self.model_data.supports_temperature is False:
            temperature = 1

        # validation
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
        # Extract async_http_client for separate async HTTP client support (#2351).
        # This allows users to provide different httpx clients for sync (httpx.Client)
        # and async (httpx.AsyncClient) operations.
        self.async_http_client = normalized_kwargs.pop(
            "async_http_client", None
        )

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ######################
    # Generate functions #
    ######################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=False)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": content}]

        if schema:
            if self.supports_structured_outputs() is True:
                completion = client.beta.chat.completions.parse(
                    model=self.name,
                    messages=messages,
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                self._update_llm_span_from_completion(completion, messages)
                return structured_output, cost
            if self.supports_json_mode() is True:
                completion = client.beta.chat.completions.parse(
                    model=self.name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                json_output = trim_and_load_json(
                    completion.choices[0].message.content
                )
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                self._update_llm_span_from_completion(completion, messages)
                return schema.model_validate(json_output), cost

        completion = client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        self._update_llm_span_from_completion(completion, messages)
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": content}]

        if schema:
            if self.supports_structured_outputs() is True:
                completion = await client.beta.chat.completions.parse(
                    model=self.name,
                    messages=messages,
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                self._update_llm_span_from_completion(completion, messages)
                return structured_output, cost
            if self.supports_json_mode() is True:
                completion = await client.beta.chat.completions.parse(
                    model=self.name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                json_output = trim_and_load_json(
                    completion.choices[0].message.content
                )
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                self._update_llm_span_from_completion(completion, messages)
                return schema.model_validate(json_output), cost

        completion = await client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        self._update_llm_span_from_completion(completion, messages)
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    ############################
    # Other generate functions #
    ############################

    def _cap_top_logprobs(self, top_logprobs: int) -> int:
        max_log_probs = self.model_data.max_log_probs
        if max_log_probs is None:
            return top_logprobs

        return min(top_logprobs, max_log_probs)

    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        model_name = self.name
        is_multimodal = check_if_multimodal(prompt)

        if self.supports_log_probs() is False:
            raise AgentEvalError(
                f"Model `{model_name}` does not support `logprobs` / `top_logprobs`. "
                "Please use a different OpenAI model (for example `gpt-4.1` or `gpt-4o`) "
                "when calling `generate_raw_response`."
            )

        top_logprobs = self._cap_top_logprobs(top_logprobs)
        client = self.load_model(async_mode=False)
        if is_multimodal:
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        completion = client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        self._update_llm_span_from_completion(completion, messages)

        return completion, cost

    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        model_name = self.name
        is_multimodal = check_if_multimodal(prompt)

        if self.supports_log_probs() is False:
            raise AgentEvalError(
                f"Model `{model_name}` does not support `logprobs` / `top_logprobs`. "
                "Please use a different OpenAI model (for example `gpt-4.1` or `gpt-4o`) "
                "when calling `a_generate_raw_response`."
            )

        top_logprobs = self._cap_top_logprobs(top_logprobs)
        client = self.load_model(async_mode=True)
        if is_multimodal:
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        completion = await client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        self._update_llm_span_from_completion(completion, messages)

        return completion, cost

    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> list[str]:
        client = self.load_model(async_mode=False)
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        response = client.chat.completions.create(
            model=self.name,
            messages=messages,
            n=n,
            temperature=temperature,
            **self.generation_kwargs,
        )
        self._update_llm_span_from_completion(response, messages)
        completions = [choice.message.content for choice in response.choices]
        return completions

    #############
    # Utilities #
    #############

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        if self.model_data.input_price and self.model_data.output_price:
            input_cost = input_tokens * self.model_data.input_price
            output_cost = output_tokens * self.model_data.output_price
            # Carry token counts alongside the cost so metric runs can surface
            # input/output token usage (EvaluationCost subclasses float, so every
            # existing `output, cost = generate(...)` caller is unaffected).
            return EvaluationCost(
                input_cost + output_cost, input_tokens, output_tokens
            )

    #########################
    # Capabilities          #
    #########################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        """
        OpenAI models that natively enforce typed structured outputs.
         Used by generate(...) when a schema is provided.
        """
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        """
        OpenAI models that enforce JSON mode
        """
        return self.model_data.supports_json

    #########
    # Model #
    #########

    def generate_content(
        self, multimodal_input: Optional[List[Union[str, MLLMImage]]] = None
    ):
        multimodal_input = [] if multimodal_input is None else multimodal_input
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": element.url},
                        }
                    )
                else:
                    element.ensure_images_loaded()
                    data_uri = (
                        f"data:{element.mimeType};base64,{element.dataBase64}"
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    )
        return content

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        kwargs["max_retries"] = 0

        if not kwargs.get("timeout"):
            kwargs["timeout"] = _request_timeout_seconds()
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to OpenAIModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )

        # Support separate sync/async HTTP clients (#2351).
        # OpenAI expects httpx.Client; AsyncOpenAI expects httpx.AsyncClient.
        # Passing the wrong type raises TypeError, so we handle them separately.
        if cls is AsyncOpenAI:
            if self.async_http_client is not None:
                kw["http_client"] = self.async_http_client
            elif "http_client" in kw:
                # A sync httpx.Client cannot be used with AsyncOpenAI.
                # Remove it to fall back to the SDK's default async client.
                del kw["http_client"]

        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name}"

    def _update_llm_span_from_completion(
        self, completion, messages=None
    ) -> None:
        return None


def warn_gpt_model_deprecated(stacklevel: int = 3) -> None:
    warnings.warn(
        "'GPTModel' is deprecated and will be removed in a future release. "
        "Use 'OpenAIModel' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def __getattr__(name: str):
    if name == "GPTModel":
        warn_gpt_model_deprecated()
        return OpenAIModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
