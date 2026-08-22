from contextlib import contextmanager
from typing import Optional


@contextmanager
def metric_progress_indicator(
    metric,
    async_mode: Optional[bool] = None,
    total: int = 9999,
    transient: bool = True,
    _show_indicator: bool = True,
    _in_component: bool = False,
):
    """No-op progress indicator (tracing/rich progress stripped in agenteval)."""
    yield
