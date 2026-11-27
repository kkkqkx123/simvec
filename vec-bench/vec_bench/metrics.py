"""Recall and latency aggregation for e2e benchmarks.

All recall math works on result id sets vs. the first `k` ground-truth ids.
Latency is recorded per query in seconds and reported as p50/p95/p99 plus mean.
"""

import numpy as np


def recall_at_k(result_ids: list[int], gt_ids: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    truth = set(gt_ids[:k])
    if not truth:
        return 0.0
    return len(truth.intersection(result_ids[:k])) / len(truth)


def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    arr = np.asarray(latencies) * 1000.0
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
