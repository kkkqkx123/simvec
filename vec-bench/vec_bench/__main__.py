"""CLI entry point: `python -m vec_bench {download,run} ...`"""

import argparse
import logging
import pathlib
import sys

from .backends import BACKEND_NAMES, make_backend
from .datasets import DATASETS, dataset_dir_name, download
from .runner import run, save_result

log = logging.getLogger("vec_bench")

DEFAULT_DATA_ROOT = pathlib.Path("/tmp/vec-bench/dataset")
DEFAULT_OUT = pathlib.Path("results")


def cmd_download(args: argparse.Namespace) -> int:
    for name in args.dataset:
        if name not in DATASETS:
            log.error("unknown dataset %s (known: %s)", name, ", ".join(DATASETS))
            return 2
        download(name, args.data_root)
        log.info("ready: %s -> %s", name, args.data_root / dataset_dir_name(name))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import ensure_downloaded

    backend = make_backend(args.backend, args.work_root, vars(args.backend_kwargs))
    try:
        ensure_downloaded(args.dataset, args.data_root)
        result = run(
            backend,
            args.dataset,
            args.data_root,
            k=args.k,
            max_queries=args.max_queries,
            insert_batch=args.insert_batch,
            warmup=args.warmup,
            drop_old=not args.keep,
        )
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    path = save_result(result, args.out)
    log.info("result written to %s", path)
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vec_bench", description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT,
                        help="dataset cache root (default %(default)s)")
    parser.add_argument("--work-root", type=pathlib.Path,
                        default=pathlib.Path("/tmp/vec-bench/work"),
                        help="scratch root for engine data dirs (default %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download dataset files from the OSS mirror")
    p_dl.add_argument("--dataset", nargs="+", default=["cohere_small_100k"],
                      help="dataset names (default: cohere_small_100k)")
    p_dl.set_defaults(func=cmd_download)

    p_run = sub.add_parser("run", help="run one backend on one dataset")
    p_run.add_argument("--backend", required=True, choices=BACKEND_NAMES)
    p_run.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p_run.add_argument("-k", type=int, default=100)
    p_run.add_argument("--max-queries", type=int, default=None,
                       help="limit number of test queries (default: all)")
    p_run.add_argument("--insert-batch", type=int, default=1000)
    p_run.add_argument("--warmup", type=int, default=50)
    p_run.add_argument("--keep", action="store_true",
                       help="reuse existing engine data dir instead of dropping it")
    p_run.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    p_run.add_argument("--dsn", default=None,
                       help="pgvector only: connection string, e.g. postgresql://u:p@host/db")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        args.backend_kwargs = {"dsn": args.dsn} if args.dsn else {}
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
