"""Python binding for the simvec local vector engine."""

from .engine import Engine
from .errors import (
    AlreadyExistsError,
    CapacityExceededError,
    FilterError,
    InvalidConfigError,
    InvalidDimensionError,
    MisuseError,
    NotFoundError,
    SimvecError,
)

__all__ = [
    "Engine",
    "SimvecError",
    "NotFoundError",
    "AlreadyExistsError",
    "InvalidConfigError",
    "InvalidDimensionError",
    "MisuseError",
    "CapacityExceededError",
    "FilterError",
]
