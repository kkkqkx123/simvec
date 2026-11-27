"""Qdrant backend in local (embedded) mode, mirroring VectorDBBench's
qdrant_local client: index disabled during bulk insert, then rebuilt."""

import pathlib
import time

import numpy as np

from .base import Backend, simvec_distance

QDRANT_BATCH = 100
_WAIT_SECONDS = 5


class QdrantBackend(Backend):
    name = "qdrant"

    def __init__(self, workdir: pathlib.Path, collection: str = "bench", **kwargs):
        self._workdir = workdir
        self._collection = collection
        self._client = None

    def setup(self, dim: int, metric: str, drop_old: bool) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(path=str(self._workdir))
        if drop_old and self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                    if metric == "cosine"
                    else models.Distance.EUCLID
                    if metric == "l2"
                    else models.Distance.DOT,
                ),
            )
        # speed up bulk load: no HNSW updates while inserting
        self._client.update_collection(
            self._collection,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    def insert(self, ids: list[int], vectors: np.ndarray) -> int:
        inserted = 0
        for offset in range(0, len(ids), QDRANT_BATCH):
            batch_ids = ids[offset : offset + QDRANT_BATCH]
            batch_vecs = vectors[offset : offset + QDRANT_BATCH]
            self._client.upsert(
                collection_name=self._collection,
                wait=True,
                points=self._models.Batch(
                    ids=[int(i) for i in batch_ids],
                    vectors=batch_vecs.tolist(),
                    payloads=[{} for _ in batch_ids],
                ),
            )
            inserted += len(batch_ids)
        return inserted

    def optimize(self) -> None:
        from qdrant_client.http.models import CollectionStatus

        self._client.update_collection(
            self._collection,
            optimizer_config=self._models.OptimizersConfigDiff(indexing_threshold=100),
        )
        while True:
            info = self._client.get_collection(self._collection)
            if info.status == CollectionStatus.GREEN:
                return
            time.sleep(_WAIT_SECONDS)

    def search(self, query: np.ndarray, k: int) -> list[int]:
        res = self._client.query_points(
            collection_name=self._collection,
            query=query.tolist(),
            limit=k,
            search_params=self._models.SearchParams(hnsw_ef=256),
        ).points
        return [int(p.id) for p in res]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_ = simvec_distance  # kept for parity with other backends
