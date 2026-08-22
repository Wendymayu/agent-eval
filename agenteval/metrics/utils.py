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
# Import BaseMetric/BaseConversationalMetric directly from the module, not
# `agenteval.metrics`, to avoid a circular import through metrics/__init__.
from agenteval.metrics.base_metric import (
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


def check_at_least_one_metric_has_threshold(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
):
    # Flaky metrics don't count: they must not decide a test case's
    # pass/fail status, so every test case needs at least one non-flaky
    # metric with a threshold to guarantee it gets a verdict.
    if not any(
        metric.threshold is not None and not metric.flaky for metric in metrics
    ):
        raise ValueError(
            "You must provide at least one non-flaky metric with a "
            "'threshold', otherwise test cases can never pass or fail."
        )


def copy_metrics(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
) -> List[Union[BaseMetric, BaseConversationalMetric]]:
    copied_metrics = []
    for metric in metrics:
        metric_class = type(metric)
        args = vars(metric)

        superclasses = metric_class.__mro__

        valid_params = []

        for superclass in superclasses:
            signature = inspect.signature(superclass.__init__)
            superclass_params = signature.parameters.keys()
            valid_params.extend(superclass_params)
        valid_params = set(valid_params)
        valid_args = {key: args[key] for key in valid_params if key in args}

        copied_metrics.append(metric_class(**valid_args))
    return copied_metrics


def format_turns(
    llm_test_cases: List[LLMTestCase], test_case_params: List[SingleTurnParams]
) -> List[Dict[str, Union[str, List[str]]]]:
    res = []
    for llm_test_case in llm_test_cases:
        dict = {}
        for param in test_case_params:
            value = getattr(llm_test_case, param.value)
            if value:
                dict[param.value] = value
        res.append(dict)
    return res


def convert_turn_to_dict(
    turn: Turn,
    turn_params: Optional[List[MultiTurnParams]] = None,
) -> Dict:
    if turn_params is None:
        turn_params = [MultiTurnParams.CONTENT, MultiTurnParams.ROLE]
    result = {}
    for param in turn_params:
        if param in (
            MultiTurnParams.SCENARIO,
            MultiTurnParams.EXPECTED_OUTCOME,
            MultiTurnParams.METADATA,
            MultiTurnParams.TAGS,
        ):
            continue

        if not hasattr(turn, param.value):
            continue

        value = getattr(turn, param.value)
        if value is not None:
            result[param.value] = value

    return result


def get_turns_in_sliding_window(turns: List[Turn], window_size: int):
    for i in range(len(turns)):
        yield turns[max(0, i - window_size + 1) : i + 1]


def get_unit_interactions(turns: List[Turn]) -> List[List[Turn]]:
    units: List[List[Turn]] = []
    current: List[Turn] = []
    has_user = False

    for turn in turns:
        # Boundary: user after assistant, but only if we've already seen a user in current
        if (
            current
            and current[-1].role == "assistant"
            and turn.role == "user"
            and has_user
        ):
            units.append(current)  # finalize previous unit
            current = [turn]  # start new unit with this user
            has_user = True
            continue

        # Otherwise just accumulate
        current.append(turn)
        if turn.role == "user":
            has_user = True

    # Finalize last unit only if it ends with assistant and includes a user
    if (
        current
        and len(current) > 1
        and current[-1].role == "assistant"
        and has_user
    ):
        units.append(current)

    return units


def print_verbose_logs(metric: str, logs: str):
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write(f"{metric} Verbose Logs\n")
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write("\n")
    sys.stdout.write(logs + "\n")
    sys.stdout.write("\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()


def construct_verbose_logs(metric: BaseMetric, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += " \n \n"
    if metric.verbose_mode:
        # only print reason and score for deepeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n \n{steps[-1]}")

    return verbose_logs


def check_conversational_test_case_params(
    test_case: ConversationalTestCase,
    test_case_params: List[MultiTurnParams],
    metric: BaseConversationalMetric,
    require_chatbot_role: bool = False,
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
        if not model or not model.supports_multimodal():
            raise ValueError(
                f"The evaluation model {getattr(model, 'name', model)} does not "
                "support multimodal inputs."
            )

    if isinstance(test_case, ConversationalTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'ConversationalTestCase' using the conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    if (
        MultiTurnParams.EXPECTED_OUTCOME in test_case_params
        and test_case.expected_outcome is None
    ):
        error_str = f"'expected_outcome' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if (
        MultiTurnParams.SCENARIO in test_case_params
        and test_case.scenario is None
    ):
        error_str = f"'scenario' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if (
        MultiTurnParams.METADATA in test_case_params
        and test_case.metadata is None
    ):
        error_str = f"'metadata' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if MultiTurnParams.TAGS in test_case_params and test_case.tags is None:
        error_str = f"'tags' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if require_chatbot_role and test_case.chatbot_role is None:
        error_str = f"'chatbot_role' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if len(test_case.turns) == 0:
        error_str = "'turns' in conversational test case cannot be empty."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)


def check_llm_test_case_params(
    test_case: LLMTestCase,
    test_case_params: List[SingleTurnParams],
    input_image_count: Optional[int],
    actual_output_image_count: Optional[int],
    metric: BaseMetric,
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
        if not model or not model.supports_multimodal():
            raise ValueError(
                f"The evaluation model {getattr(model, 'name', model)} does not "
                "support multimodal inputs."
            )

        if input_image_count:
            count = 0
            for ele in convert_to_multi_modal_array(test_case.input):
                if isinstance(ele, MLLMImage):
                    count += 1
            if count != input_image_count:
                error_str = f"Can only evaluate test cases with '{input_image_count}' input images using the '{metric.__name__}' metric. `{count}` found."
                raise ValueError(error_str)

        if actual_output_image_count:
            count = 0
            for ele in convert_to_multi_modal_array(test_case.actual_output):
                if isinstance(ele, MLLMImage):
                    count += 1
            if count != actual_output_image_count:
                error_str = f"Can only evaluate test cases with '{actual_output_image_count}' output images using the '{metric.__name__}' metric. `{count}` found."
                raise ValueError(error_str)

    if isinstance(test_case, LLMTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'LLMTestCase' using the non-conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    # Centralized: if a metric requires actual_output, reject empty/whitespace
    # (including empty multimodal outputs) as "missing params".
    if SingleTurnParams.ACTUAL_OUTPUT in test_case_params:
        actual_output = getattr(test_case, SingleTurnParams.ACTUAL_OUTPUT.value)
        if isinstance(actual_output, str) and actual_output == "":
            error_str = f"'actual_output' cannot be empty for the '{metric.__name__}' metric"
            metric.error = error_str
            raise MissingTestCaseParamsError(error_str)

    missing_params = []
    for param in test_case_params:
        if getattr(test_case, param.value) is None:
            missing_params.append(f"'{param.value}'")

    if missing_params:
        if len(missing_params) == 1:
            missing_params_str = missing_params[0]
        elif len(missing_params) == 2:
            missing_params_str = " and ".join(missing_params)
        else:
            missing_params_str = (
                ", ".join(missing_params[:-1]) + ", and " + missing_params[-1]
            )

        error_str = f"{missing_params_str} cannot be None for the '{metric.__name__}' metric"
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)


def trimAndLoadJson(
    input_string: Optional[str],
    metric: Optional[BaseMetric] = None,
) -> Any:
    if input_string is None:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)

    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        # Some models emit a trailing comma before a closing ] or }. Strip it
        # and retry, but only after a direct parse fails, so valid JSON string
        # values containing ", ]" or ", }" are never corrupted.
        try:
            return json.loads(re.sub(r",\s*([\]}])", r"\1", jsonStr))
        except json.JSONDecodeError:
            error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
            if metric is not None:
                metric.error = error_str
            raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


SchemaType = TypeVar("SchemaType")
ReturnType = TypeVar("ReturnType")


def accrue_token_usage(
    metric: Union[BaseMetric, BaseConversationalMetric],
    cost: Optional[float],
) -> None:
    """Accrue the input/output token counts that produced ``cost`` onto the
    metric.

    Native models return their cost as an ``EvaluationCost`` (a ``float``
    subclass carrying ``input_tokens``/``output_tokens``). Costs from providers
    that aren't wrapped yet — or ``None`` when pricing is unknown — are plain
    floats with no token data, so tokens are only accrued when ``cost`` actually
    carries them. Call this right after ``metric._accrue_cost(cost)`` so token
    usage tracks cost exactly.
    """
    if isinstance(cost, EvaluationCost):
        metric._accrue_tokens(cost.input_tokens, cost.output_tokens)


def verdict_from_json(
    data: Dict,
    verdict_cls: Type[SchemaType],
    allowed: Tuple[str, ...] = ("yes", "no"),
) -> Optional[SchemaType]:
    """Build a yes/no-style verdict object from a judge's JSON output.

    Judges don't always honor a strict yes/no instruction — a non-native model
    may reply "No, it's off-topic". This normalizes the verbose reply to its
    leading word and drops anything outside ``allowed`` to ``None`` (so the
    caller can exclude it from scoring) rather than crashing a ``Literal`` schema
    or silently miscounting it. ``verdict_cls`` is expected to have ``verdict``
    and ``reason`` fields and to permit every value in ``allowed``.
    """
    match = re.match(r"[a-z]+", str(data.get("verdict", "")).strip().lower())
    leading = match.group(0) if match else ""
    if leading not in allowed:
        return None
    return verdict_cls(verdict=leading, reason=data.get("reason"))


def generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    """
    Synchronous wrapper:
    - calls model.generate_with_schema(...)
    - accrues cost if applicable
    - if schema instance -> extract_schema
      else parse JSON -> extract_json
    """
    if metric.using_native_model:
        result, cost = metric.model.generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = metric.model.generate_with_schema(prompt, schema=schema_cls)
    if isinstance(result, schema_cls):
        return extract_schema(result)
    data = trimAndLoadJson(result, metric)
    return extract_json(data)


async def a_generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    if metric.using_native_model:
        result, cost = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )

    # Handle models that return (result, cost) tuple even when not native
    if isinstance(result, tuple) and len(result) == 2:
        actual_result, cost = result
        if hasattr(metric, "_accrue_cost"):
            metric._accrue_cost(cost)
            accrue_token_usage(metric, cost)
        result = actual_result

    if isinstance(result, schema_cls):
        return extract_schema(result)

    data = trimAndLoadJson(result, metric)
    return extract_json(data)


###############################################
# LLM
###############################################


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
