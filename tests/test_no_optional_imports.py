import importlib
import re
import sys
from pathlib import Path

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


# Matches actual import statements that pull in a banned module. The leading
# `(?:import|from)` anchor means plain word mentions in comments, docstrings,
# or attribute names (e.g. the `confident_link` field, "Confident cloud" in a
# docstring, "tracing/rich progress stripped") do NOT match — only real
# `import <x>` / `from <x> import ...` statements do. `agenteval.confident`
# won't match `agenteval.confident_link` because the trailing `\b` requires a
# word boundary and `_` is a word character.
_BANNED_IMPORT_RE = re.compile(
    r"\b(?:import|from)\s+(posthog|opentelemetry|agenteval\.(?:tracing|telemetry|confident))\b"
)


def test_no_banned_imports_in_source():
    """Source-grep safety net: deterministically assert NO banned import
    statements (top-level OR lazy, executed or not) exist anywhere in the
    agenteval package source.

    The sys.modules checks above are blind to two gaps this test closes:
      1. imports inside uncalled functions (a lazy `import posthog` only
         registers in sys.modules when the function runs, so a test that never
         calls it would miss the regression); and
      2. packages not installed in the test env (posthog isn't installed, so it
         can never appear in sys.modules regardless of what the source does).
    Reading source text is immune to both: it catches every `import`/`from`
    statement whether or not it would ever execute.
    """
    import agenteval

    pkg_dir = Path(agenteval.__file__).parent
    py_files = sorted(pkg_dir.rglob("*.py"))
    # Guard against a silent no-op (e.g. path mis-resolution).
    assert py_files, f"no .py files found under {pkg_dir}"

    offenders = []
    for py in py_files:
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = _BANNED_IMPORT_RE.search(line)
            if match:
                offenders.append(f"{py}:{lineno}: {line.strip()}")
    assert not offenders, (
        "banned import statements found in agenteval source "
        "(must be stripped — tracing/telemetry/confident/posthog/opentelemetry):\n  "
        + "\n  ".join(offenders)
    )
