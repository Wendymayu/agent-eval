from agenteval.utils import (
    make_model_config,
    get_or_create_event_loop,
    prettify_list,
    check_if_multimodal,
    convert_to_multi_modal_array,
)


def test_make_model_config_returns_configdict():
    from pydantic import ConfigDict
    # ConfigDict is a TypedDict: a plain dict at runtime. TypedDicts do not
    # support isinstance checks, so verify via dict + expected contents.
    cfg = make_model_config(extra="ignore", arbitrary_types_allowed=True)
    assert isinstance(cfg, dict)
    assert cfg["extra"] == "ignore"
    assert cfg["arbitrary_types_allowed"] is True


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


import pytest

@pytest.mark.skip(reason="needs MLLMImage.parse_multimodal_string from Task 3")
def test_convert_to_multi_modal_array_plain_string():
    from agenteval.utils import convert_to_multi_modal_array
    arr = convert_to_multi_modal_array("plain text")
    assert arr == ["plain text"]
