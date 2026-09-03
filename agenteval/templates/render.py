"""Jinja2 rendering for metric prompt templates.

Prompt text lives as Jinja2 strings in each metric's ``template.py``. This
module renders them, injecting shared ``_fragments`` (e.g. multimodal rules)
and honoring ``strict`` (StrictUndefined) vs lenient undefined behavior.

``filter_template_kwargs`` narrows the render context to the variables a
custom override actually declares, so subclassing a Template and overriding
one method with an explicit signature still works.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional

import jinja2

from agenteval.templates.fragments import FRAGMENTS

_jinja_strict: Optional[jinja2.Environment] = None
_jinja_lenient: Optional[jinja2.Environment] = None


def _env(strict: bool) -> jinja2.Environment:
    global _jinja_strict, _jinja_lenient
    if strict:
        if _jinja_strict is None:
            _jinja_strict = jinja2.Environment(
                undefined=jinja2.StrictUndefined
            )
        return _jinja_strict
    if _jinja_lenient is None:
        _jinja_lenient = jinja2.Environment()
    return _jinja_lenient


def render_template(
    template_str: str,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    """Render a Jinja2 template string.

    ``multimodal`` and shared ``_fragments`` are always available to the
    template; everything else comes through ``kwargs``.
    """
    return _env(strict).from_string(template_str).render(
        multimodal=multimodal,
        _fragments=FRAGMENTS,
        **kwargs,
    )


def filter_template_kwargs(
    fn: Callable[..., Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Narrow ``kwargs`` to the parameters ``fn`` actually declares.

    A Template override may declare only the variables it interpolates, so
    handing it the full render context (which always includes ``multimodal``
    and ``strict``) would raise ``TypeError``. Functions that accept
    ``**kwargs`` get the full context.
    """
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(kwargs)

    accepted = {
        p.name
        for p in parameters
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted}
