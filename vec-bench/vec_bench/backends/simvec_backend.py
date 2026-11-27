"""simvec backend: drives the local engine through the Python binding."""

import pathlib
import shutil

import numpy as np

from .base import Backend, simvec_distance


class SimvecBackend(Backend):
    name = "simvec"

    def __init__(self, workdir: pathlib.Path, collection: str = "bench", **kwargs):
        import simvec  # provided by bindings/python (SIMVEC_LIB / installed package)

        self._simvec = simvec
        self._workdir = workdir
        self._collection = collection
        self._engine = None

    def setup(self, dim: int, metric: str, drop_old: bool) -> None:
        if self._workdir.exists():
            if drop_old:
                shutil.rmtree(self._workdir)
            else:
                raise RuntimeError(f"data dir {self._workdir} exists; pass drop_old")
        self._engine = self._simvec.Engine(str(self._workdir))
        self._engine.create_collection(
            self._collection,
            {"vector_size": dim, "distance": simvec_distance(metric), "index": "Hnsw"},
        )

    def insert(self, ids: list[int], vectors: np.ndarray) -> int:
        points = [
            {"id": str(int(i)), "vector": row.tolist()}
            for i, row in zip(ids, vectors)
        ]
        self._engine.upsert_batch(self._collection, points)
        return len(points)

    def optimize(self) -> None:
        # build_index returns whether an index was built; drain first so the
        # WAL-pending points are all visible before asking for ANN.
        self._engine.drain_pending(self._collection)
        self._engine.build_index(self._collection)
        self._engine.drain_pending(self._collection)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        hits = self._engine.search_vector(self._collection, query.tolist(), k)
        return [int(h["id"]) for h in hits]

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
