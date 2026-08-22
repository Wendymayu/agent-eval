class AgentEvalError(Exception):
    """Base class for framework-originated errors."""


class MissingTestCaseParamsError(AgentEvalError):
    """Required test case fields are missing for a metric."""
    pass
