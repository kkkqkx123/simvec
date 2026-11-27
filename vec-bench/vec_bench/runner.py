"""E2E benchmark runner: load dataset, bulk insert, optimize, timed search.

Metric contract: recall is computed on id sets (higher-is-better) against the
dataset's ground-truth neighbors; internal distance direction never leaks into
the comparison. COSINE datasets are searched with each engine's native cosine
handling (simvec normalizes on the fly), so no pre-normalization is applied.
"""

import json
import logging
import pathlib
import time

import numpy as np
from pyarrow.parquet import ParquetFile

from .datasets import DATASETS, GT_FILE, TEST_FILE, download, dataset_files, dataset_dir_name
from .metrics import recall_at_k, summarize_latencies

log = logging.getLogger(__name__)


def ensure_downloaded(name: str, data_root: pathlib.Path) -> pathlib.Path:
    """Download on demand and return the dataset directory."""
    download(name, data_root)
    return data_root / dataset_dir_name(name)


def _vectors_2d(col) -> np.ndarray:
    """Convert a parquet list column to a (rows, dim) float32 array."""
    if hasattr(col, "combine_chunks"):
        col = col.combine_chunks()
    flat = np.asarray(col.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    rows = len(col)
    return flat.reshape(rows, -1)


def _iter_train_vectors(ds_dir: pathlib.Path, file_count: int, batch_rows: int):
    """Yield (ids, vectors) batches across all train shard files."""
    files = [f for f in dataset_files_names(file_count) if f.startswith("train")]
    for fname in files:
        pf = ParquetFile(ds_dir / fname, memory_map=True)
        cols = pf.schema_arrow.names
        id_col = "id"
        vec_col = "emb" if "emb" in cols else cols[-1]
        for record in pf.iter_batches(batch_size=batch_rows, columns=[id_col, vec_col]):
            ids = record.column(0).to_pylist()
            vecs = _vectors_2d(record.column(1))
            yield ids, vecs


def dataset_files_names(file_count: int) -> list[str]:
    if file_count > 1:
        return [f"train-{i:02d}-of-{file_count}.parquet" for i in range(file_count)]
    return ["train.parquet"]


def read_queries(ds_dir: pathlib.Path):
    pf = ParquetFile(ds_dir / TEST_FILE, memory_map=True)
    cols = pf.schema_arrow.names
    vec_col = "emb" if "emb" in cols else cols[-1]
    batch = next(pf.iter_batches(batch_size=pf.metadata.num_rows, columns=[vec_col]))
    return _vectors_2d(batch.column(0))


def read_ground_truth(ds_dir: pathlib.Path) -> dict[int, list[int]]:
    pf = ParquetFile(ds_dir / GT_FILE, memory_map=True)
    cols = pf.schema_arrow.names
    id_col = "id"
    gt_col = "neighbors_id" if "neighbors_id" in cols else cols[-1]
    gt: dict[int, list[int]] = {}
    for record in pf.iter_batches(batch_size=1024, columns=[id_col, gt_col]):
        for qid, neighbors in zip(
            record.column(0).to_pylist(),
            record.column(1).to_pylist(),
        ):
            gt[int(qid)] = [int(n) for n in neighbors]
    return gt


def run(
    backend,
    name: str,
    data_root: pathlib.Path,
    *,
    k: int = 100,
    max_queries: int | None = None,
    insert_batch: int = 1000,
    warmup: int = 50,
    drop_old: bool = True,
) -> dict:
    """Run the full insert -> optimize -> search benchmark for one backend."""
    ds_dir = ensure_downloaded(name, data_root)
    meta = DATASETS[name]
    dim, metric = meta["dim"], meta["metric"]

    # ---- insert phase ----
    backend.setup(dim, metric, drop_old=drop_old)
    inserted = 0
    t0 = time.perf_counter()
    for ids, vecs in _iter_train_vectors(ds_dir, meta["file_count"], insert_batch):
        inserted += backend.insert(ids, vecs)
    insert_s = time.perf_counter() - t0
    log.info("inserted %d vectors in %.1fs (%.0f vec/s)",
             inserted, insert_s, inserted / insert_s if insert_s else 0.0)

    t0 = time.perf_counter()
    backend.optimize()
    optimize_s = time.perf_counter() - t0

    # ---- search phase ----
    queries = read_queries(ds_dir)
    if max_queries is not None:
        queries = queries[:max_queries]
    gt = read_ground_truth(ds_dir)

    for q in queries[:warmup]:  # warm-up, not measured
        backend.search(q, k)

    recalls: list[float] = []
    latencies: list[float] = []
    # test.parquet rows are in query-id order, matching the gt mapping
    for qid, q in enumerate(queries):
        t0 = time.perf_counter()
        result = backend.search(q, k)
        latencies.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(result, gt.get(qid, []), k))

    return {
        "backend": backend.name,
        "dataset": name,
        "metric": metric,
        "k": k,
        "num_queries": len(queries),
        "inserted": inserted,
        "insert_seconds": insert_s,
        "insert_vec_per_s": inserted / insert_s if insert_s else 0.0,
        "optimize_seconds": optimize_s,
        "recall": {
            "mean": float(np.mean(recalls)) if recalls else 0.0,
            "min": float(np.min(recalls)) if recalls else 0.0,
        },
        "latency": summarize_latencies(latencies),
    }


def save_result(result: dict, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{result['backend']}-{result['dataset']}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path
