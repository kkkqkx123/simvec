"""Status codes and exception mapping for the simvec C API."""

SIMVEC_OK = 0
SIMVEC_ERROR = 1
SIMVEC_INTERNAL = 2
SIMVEC_NOMEM = 7
SIMVEC_IOERR = 10
SIMVEC_CORRUPT = 11
SIMVEC_NOTFOUND = 12
SIMVEC_ALREADY_EXISTS = 13
SIMVEC_INVALID_CONFIG = 16
SIMVEC_INVALID_DIM = 19
SIMVEC_MISUSE = 20
SIMVEC_NON_FINITE = 21
SIMVEC_CAPACITY_EXCEEDED = 22
SIMVEC_FILTER = 23


class SimvecError(Exception):
    """Base error carrying the raw status code."""


class NotFoundError(SimvecError):
    pass


class AlreadyExistsError(SimvecError):
    pass


class InvalidConfigError(SimvecError):
    pass


class InvalidDimensionError(SimvecError):
    pass


class MisuseError(SimvecError):
    pass


class CapacityExceededError(SimvecError):
    pass


class FilterError(SimvecError):
    pass


_CODE_TO_ERROR = {
    SIMVEC_NOTFOUND: NotFoundError,
    SIMVEC_ALREADY_EXISTS: AlreadyExistsError,
    SIMVEC_INVALID_CONFIG: InvalidConfigError,
    SIMVEC_INVALID_DIM: InvalidDimensionError,
    SIMVEC_MISUSE: MisuseError,
    SIMVEC_CAPACITY_EXCEEDED: CapacityExceededError,
    SIMVEC_FILTER: FilterError,
}


def raise_for_code(code, message):
    """Raise the typed exception matching a non-zero status code."""
    if code == SIMVEC_OK:
        return
    kind = _CODE_TO_ERROR.get(code, SimvecError)
    raise kind(f"[simvec:{code}] {message or 'unknown error'}")
