"""Backend registry and construction."""

import pathlib

from .base import Backend
from .qdrant_backend import QdrantBackend
from .simvec_backend import SimvecBackend

_BACKENDS = {
    "simvec": SimvecBackend,
    "qdrant": QdrantBackend,
}

try:  # pgvector needs a live server; only import when requested
    from .pgvector_backend import PgvectorBackend

    _BACKENDS["pgvector"] = PgvectorBackend
except ImportError:
    pass

BACKEND_NAMES = sorted(_BACKENDS)


def make_backend(name: str, workdir: pathlib.Path, kwargs: dict) -> Backend:
    if name == "pgvector" and "pgvector" not in _BACKENDS:
        raise RuntimeError("pgvector backend requires psycopg and pgvector packages")
    if name == "pgvector":
        return _BACKENDS[name](**kwargs)
    return _BACKENDS[name](workdir=workdir / name, **kwargs)
