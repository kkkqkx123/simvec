"""Backend contract: every engine under test implements these four methods."""

from abc import ABC, abstractmethod

import numpy as np

# simvec collection config uses serde variant names: Cosine / Euclid / Dot.
_DISTANCE_TO_SIMVEC = {"cosine": "Cosine", "l2": "Euclid", "ip": "Dot"}


def simvec_distance(metric: str) -> str:
    return _DISTANCE_TO_SIMVEC[metric]


class Backend(ABC):
    """One instance per (dataset, collection). Instances are single-threaded."""

    name: str = "backend"

    @abstractmethod
    def setup(self, dim: int, metric: str, drop_old: bool) -> None:
        """Create (or drop + recreate) the target collection."""

    @abstractmethod
    def insert(self, ids: list[int], vectors: np.ndarray) -> int:
        """Insert a batch; return the number of accepted rows."""

    @abstractmethod
    def optimize(self) -> None:
        """Block until the index is fully built and searchable."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Return the top-k ids for one query vector."""
