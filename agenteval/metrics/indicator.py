from contextlib import contextmanager


@contextmanager
def metric_progress_indicator(
    metric,
    total: int = 9999,
    transient: bool = True,
    _show_indicator: bool = True,
    _in_component: bool = False,
):
    """No-op progress indicator (tracing/rich progress stripped in agenteval)."""
    yield
