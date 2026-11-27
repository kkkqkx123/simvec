"""Dataset registry and download from the Zilliz Aliyun OSS mirror.

Downloads use the same public mirror as VectorDBBench
(`assets.zilliz.com.cn/benchmark/`, anonymous access), kept idempotent by
remote-vs-local file size comparison.
"""

import logging
import pathlib

log = logging.getLogger(__name__)

# Zilliz public benchmark mirror (Aliyun OSS, anonymous). Keys are
# `benchmark/<dataset>/<file>` against bucket "benchmark" — same layout
# VectorDBBench's AliyunOSSReader uses.
OSS_ENDPOINT = "https://assets.zilliz.com.cn"

DATASETS: dict[str, dict] = {
    "cohere_small_100k": dict(
        dim=768, metric="cosine", size=100_000, file_count=1, with_gt=True,
    ),
    "cohere_medium_1m": dict(
        dim=768, metric="cosine", size=1_000_000, file_count=1, with_gt=True,
    ),
    "cohere_large_10m": dict(
        dim=768, metric="cosine", size=10_000_000, file_count=10, with_gt=True,
    ),
    "gist_small_100k": dict(
        dim=960, metric="l2", size=100_000, file_count=1, with_gt=True,
    ),
    "gist_medium_1m": dict(
        dim=960, metric="l2", size=1_000_000, file_count=1, with_gt=True,
    ),
}

TEST_FILE = "test.parquet"
GT_FILE = "neighbors.parquet"


def numerize(size: int) -> str:
    if size >= 1_000_000:
        return f"{size // 1_000_000}m"
    if size >= 1_000:
        return f"{size // 1_000}k"
    return str(size)


def dataset_dir_name(name: str) -> str:
    """Mirror VectorDBBench's `dir_name` so both sides can share one cache."""
    ds = DATASETS[name]
    label = "large" if ds["size"] >= 10_000_000 else "medium" if ds["size"] >= 1_000_000 else "small"
    return f"{name.rsplit('_', 2)[0]}_{label}_{numerize(ds['size'])}".lower()


def dataset_files(name: str) -> list[str]:
    ds = DATASETS[name]
    n = ds["file_count"]
    files = (
        [f"train-{i:02d}-of-{n}.parquet" for i in range(n)]
        if n > 1
        else ["train.parquet"]
    )
    files.append(TEST_FILE)
    if ds["with_gt"]:
        files.append(GT_FILE)
    return files


def download(name: str, local_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Download all files of `name` into `local_root/<dir_name>/`, skipping
    files whose local size already matches the remote object."""
    import oss2

    dir_name = dataset_dir_name(name)
    ds_root = local_root / dir_name
    ds_root.mkdir(parents=True, exist_ok=True)

    bucket = oss2.Bucket(oss2.AnonymousAuth(), OSS_ENDPOINT, "benchmark")
    files = dataset_files(name)

    todo: list[tuple[str, pathlib.Path]] = []
    for f in files:
        local = ds_root / f
        key = str(pathlib.PurePosixPath("benchmark", dir_name, f))
        if local.exists():
            try:
                meta = bucket.head_object(key)
                if meta.content_length == local.stat().st_size:
                    continue
                log.info("size mismatch on %s, re-downloading", key)
            except oss2.exceptions.NoSuchKey:
                log.warning("remote key %s missing, keeping local copy", key)
                continue
        todo.append((key, local))

    for key, local in todo:
        log.info("downloading %s", key)
        bucket.get_object_to_file(key, local.absolute().as_posix())
    if todo:
        log.info("downloaded %d files for %s", len(todo), dir_name)
    else:
        log.info("all files of %s already present", dir_name)

    return {f: ds_root / f for f in files}
