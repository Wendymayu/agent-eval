"""Tests for the per-metric Jinja2 template system.

Prompts live as Jinja2 strings in each metric's ``template.py`` and render
through ``render_template``. These tests cover rendering, conditional blocks
(multimodal / rubric), shared fragment injection, and ``filter_template_kwargs``.
"""

import pytest

from agenteval.metrics.ai_tone.template import AIToneTemplate
from agenteval.metrics.answer_relevance.template import (
    AnswerRelevanceTemplate,
)
from agenteval.metrics.g_eval.template import GEvalTemplate
from agenteval.templates.render import filter_template_kwargs, render_template


# --- render_template: plain rendering + strictness ---


def test_render_template_substitutes_variables():
    out = render_template("Hello {{ name }}!", name="world")
    assert out == "Hello world!"


def test_render_template_injects_multimodal_flag_and_fragments():
    tpl = (
        "{% if multimodal %}{{ _fragments.multimodal_input_rules }}"
        "{% endif %}x"
    )
    on = render_template(tpl, multimodal=True)
    off = render_template(tpl, multimodal=False)
    assert "多模态输入规则" in on
    assert "多模态输入规则" not in off
    assert off == "x"


def test_render_template_strict_undefined_raises():
    # StrictUndefined (default) makes a missing variable a hard error, so a
    # typo in a prompt variable surfaces immediately instead of silently
    # rendering blank.
    with pytest.raises(Exception):
        render_template("{{ missing }}")


def test_render_template_lenient_undefined_is_blank():
    assert render_template("{{ missing }}", strict=False) == ""


# --- GEvalTemplate: three methods + rubric conditional ---


def test_geval_template_has_three_methods():
    for m in (
        "generate_evaluation_steps",
        "generate_evaluation_results",
        "generate_strict_evaluation_results",
    ):
        assert hasattr(GEvalTemplate, m)


def test_geval_steps_renders_criteria_and_parameters():
    out = GEvalTemplate.generate_evaluation_steps(
        criteria="is correct",
        parameters="input and actual_output",
        multimodal=False,
    )
    assert "is correct" in out  # interpolated criteria value
    assert "input and actual_output" in out  # interpolated parameters value
    assert "评分细则：" not in out  # steps template has no rubric block


def test_geval_results_rubric_conditional():
    common = dict(
        evaluation_steps=["a", "b"],
        test_case_content="tc",
        parameters="input and actual_output",
        score_range=[0, 3],
        _additional_context=None,
        multimodal=False,
    )
    without = GEvalTemplate.generate_evaluation_results(rubric=None, **common)
    with_rubric = GEvalTemplate.generate_evaluation_results(rubric="R1", **common)
    # Without a rubric: plain "评估步骤" wording, no rubric block.
    assert "评估步骤与评分细则" not in without
    assert "依据所给评分细则" not in without
    assert "评分细则：\nR1" not in without
    assert "补充上下文：" not in without
    # With a rubric: rubric wording + Rubric block appear.
    assert "评估步骤与评分细则" in with_rubric
    assert "依据所给评分细则" in with_rubric
    assert "评分细则：\nR1" in with_rubric


def test_geval_results_additional_context_conditional():
    out = GEvalTemplate.generate_evaluation_results(
        evaluation_steps=["a"],
        test_case_content="tc",
        parameters="input and output",
        score_range=[0, 1],
        rubric=None,
        _additional_context="SECRET-CTX",
        multimodal=False,
    )
    assert "补充上下文：" in out
    assert "SECRET-CTX" in out  # interpolated additional-context value


def test_geval_strict_results_renders():
    out = GEvalTemplate.generate_strict_evaluation_results(
        evaluation_steps=["a"],
        test_case_content="tc",
        parameters="input and output",
        _additional_context=None,
        multimodal=False,
    )
    assert "严格只能是 1" in out
    assert "tc" in out  # interpolated test-case content


# --- AnswerRelevanceTemplate ---


def test_answer_relevance_renders():
    out = AnswerRelevanceTemplate.generate_relevance_score(
        input="what is 2+2",
        actual_output="4",
        multimodal=False,
    )
    assert "what is 2+2" in out  # interpolated input value
    assert "4" in out  # interpolated actual_output value
    assert "多模态输入规则" not in out


def test_answer_relevance_multimodal_block():
    out = AnswerRelevanceTemplate.generate_relevance_score(
        input="q", actual_output="a", multimodal=True
    )
    assert "多模态输入规则" in out


# --- AIToneTemplate ---


def test_ai_tone_renders():
    out = AIToneTemplate.generate_ai_tone_score(
        actual_output="首先，其次", multimodal=False
    )
    assert "首先，其次" in out  # interpolated actual_output value
    assert "AI 味" in out  # prompt prose


# --- filter_template_kwargs ---


def test_filter_template_kwargs_passthrough_for_var_keyword():
    def fn(**kwargs):
        return kwargs

    ctx = {"a": 1, "b": 2, "multimodal": False, "strict": True}
    assert filter_template_kwargs(fn, ctx) == ctx


def test_filter_template_kwargs_narrows_to_declared():
    def fn(actual_output, multimodal):
        return actual_output, multimodal

    ctx = {"actual_output": "x", "input": "q", "multimodal": True, "strict": True}
    assert filter_template_kwargs(fn, ctx) == {
        "actual_output": "x",
        "multimodal": True,
    }


def test_filter_template_kwargs_non_callable_returns_all():
    # inspect.signature raises on non-callables -> return the full context.
    ctx = {"a": 1}
    assert filter_template_kwargs(123, ctx) == ctx
