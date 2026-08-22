from dataclasses import dataclass


@dataclass
class AsyncConfig:
    run_async: bool = True
    throttle_value: float = 0
    max_concurrent: int = 20

    def __post_init__(self):
        if self.max_concurrent < 1:
            raise ValueError("'max_concurrent' must be at least 1")
        if self.throttle_value < 0:
            raise ValueError("'throttle_value' must be at least 0")


@dataclass
class ErrorConfig:
    ignore_errors: bool = True
    skip_on_missing_params: bool = False
