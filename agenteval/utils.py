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
