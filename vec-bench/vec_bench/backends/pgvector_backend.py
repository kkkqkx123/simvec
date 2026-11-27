"""pgvector backend over psycopg (server must already have the vector ext)."""

import pathlib

import numpy as np

from .base import Backend, simvec_distance  # noqa: F401  (parity import)

_OP = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
_OPS_CLASS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}


class PgvectorBackend(Backend):
    name = "pgvector"

    def __init__(
        self,
        dsn: str,
        table_name: str = "bench",
        ef_search: int = 256,
        **kwargs,
    ):
        self._dsn = dsn
        self._table = table_name
        self._ef_search = ef_search
        self._conn = None
        self._cur = None
        self._dim = 0
        self._metric = "cosine"

    def setup(self, dim: int, metric: str, drop_old: bool) -> None:
        import psycopg

        self._dim = dim
        self._metric = metric
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        self._cur = self._conn.cursor()
        if drop_old:
            self._cur.execute(f"DROP TABLE IF EXISTS {self._table}")
        self._cur.execute(f"DROP TABLE IF EXISTS {self._table}_new")
        # load into a scratch table, swapped in optimize() after indexing
        self._cur.execute(
            f"CREATE TABLE {self._table}_new (id bigint PRIMARY KEY, vec vector({dim}))"
        )

    def insert(self, ids: list[int], vectors: np.ndarray) -> int:
        rows = [(int(i), row.tolist()) for i, row in zip(ids, vectors)]
        self._cur.executemany(
            f"INSERT INTO {self._table}_new (id, vec) VALUES (%s, %s)", rows
        )
        return len(rows)

    def optimize(self) -> None:
        self._cur.execute(
            f"CREATE INDEX ON {self._table}_new USING hnsw "
            f"(vec {_OPS_CLASS[self._metric]})"
        )
        self._cur.execute(f"ANALYZE {self._table}_new")
        self._cur.execute(
            f"ALTER TABLE {self._table}_new RENAME TO {self._table}"
        )
        self._cur.execute(
            f"SET hnsw.ef_search = {self._ef_search}"
        )

    def search(self, query: np.ndarray, k: int) -> list[int]:
        self._cur.execute(
            f"SELECT id FROM {self._table} ORDER BY vec {_OP[self._metric]} %s LIMIT %s",
            (query.tolist(), k),
        )
        return [int(r[0]) for r in self._cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._cur = None
