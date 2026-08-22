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
