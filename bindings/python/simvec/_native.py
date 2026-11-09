"""Native library loading and raw C signature declarations."""

import ctypes
import os
from pathlib import Path

from . import errors


def _candidate_paths():
    explicit = os.environ.get("SIMVEC_LIB")
    if explicit:
        yield Path(explicit)
    here = Path(__file__).resolve()
    for profile in ("debug", "release"):
        yield here.parents[3] / "target" / profile / "libsimvec.so"
        yield here.parents[3] / "target" / profile / "libsimvec.dylib"
        yield here.parents[3] / "target" / profile / "simvec.dll"


def load_library():
    """Locate and return the compiled libsimvec shared library."""
    tried = []
    for path in _candidate_paths():
        if path is not None and path.exists():
            return ctypes.CDLL(str(path))
        tried.append(str(path))
    raise errors.SimvecError(
        "libsimvec not found; set SIMVEC_LIB or run cargo build --features c_api. "
        f"Tried: {', '.join(tried)}"
    )


_lib = load_library()

_c_str = ctypes.c_char_p
_c_out_str = ctypes.POINTER(ctypes.c_void_p)
_engine_p = ctypes.c_void_p

_lib.simvec_open.argtypes = [_c_str, ctypes.POINTER(_engine_p)]
_lib.simvec_open.restype = ctypes.c_int
_lib.simvec_close.argtypes = [_engine_p]
_lib.simvec_close.restype = ctypes.c_int
_lib.simvec_free_string.argtypes = [ctypes.c_void_p]
_lib.simvec_free_string.restype = None
_lib.simvec_errcode.argtypes = [_engine_p]
_lib.simvec_errcode.restype = ctypes.c_int
_lib.simvec_last_error.argtypes = [_engine_p, _c_out_str]
_lib.simvec_last_error.restype = ctypes.c_int
_lib.simvec_last_error_text.argtypes = []
_lib.simvec_last_error_text.restype = ctypes.c_void_p

_lib.simvec_create_collection.argtypes = [_engine_p, _c_str, _c_str]
_lib.simvec_create_collection.restype = ctypes.c_int
_lib.simvec_delete_collection.argtypes = [_engine_p, _c_str]
_lib.simvec_delete_collection.restype = ctypes.c_int
_lib.simvec_collection_exists.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_int)]
_lib.simvec_collection_exists.restype = ctypes.c_int
_lib.simvec_list_collections.argtypes = [_engine_p, _c_out_str]
_lib.simvec_list_collections.restype = ctypes.c_int
_lib.simvec_collection_info.argtypes = [_engine_p, _c_str, _c_out_str]
_lib.simvec_collection_info.restype = ctypes.c_int
_lib.simvec_collection_config.argtypes = [_engine_p, _c_str, _c_out_str]
_lib.simvec_collection_config.restype = ctypes.c_int

_lib.simvec_upsert.argtypes = [
    _engine_p,
    _c_str,
    _c_str,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    _c_str,
]
_lib.simvec_upsert.restype = ctypes.c_int
_lib.simvec_upsert_batch.argtypes = [_engine_p, _c_str, _c_str]
_lib.simvec_upsert_batch.restype = ctypes.c_int
_lib.simvec_delete.argtypes = [_engine_p, _c_str, _c_str]
_lib.simvec_delete.restype = ctypes.c_int
_lib.simvec_delete_batch.argtypes = [_engine_p, _c_str, _c_str]
_lib.simvec_delete_batch.restype = ctypes.c_int
_lib.simvec_delete_by_filter.argtypes = [_engine_p, _c_str, _c_str, ctypes.POINTER(ctypes.c_uint64)]
_lib.simvec_delete_by_filter.restype = ctypes.c_int
_lib.simvec_set_payload.argtypes = [_engine_p, _c_str, _c_str, _c_str]
_lib.simvec_set_payload.restype = ctypes.c_int
_lib.simvec_merge_payload.argtypes = [_engine_p, _c_str, _c_str, _c_str]
_lib.simvec_merge_payload.restype = ctypes.c_int
_lib.simvec_delete_payload_keys.argtypes = [_engine_p, _c_str, _c_str, _c_str]
_lib.simvec_delete_payload_keys.restype = ctypes.c_int
_lib.simvec_create_payload_index.argtypes = [_engine_p, _c_str, _c_str, _c_str]
_lib.simvec_create_payload_index.restype = ctypes.c_int
_lib.simvec_delete_payload_index.argtypes = [
    _engine_p,
    _c_str,
    _c_str,
    ctypes.POINTER(ctypes.c_int),
]
_lib.simvec_delete_payload_index.restype = ctypes.c_int
_lib.simvec_list_payload_indexes.argtypes = [_engine_p, _c_str, _c_out_str]
_lib.simvec_list_payload_indexes.restype = ctypes.c_int

_lib.simvec_search.argtypes = [_engine_p, _c_str, _c_str, _c_out_str]
_lib.simvec_search.restype = ctypes.c_int
_lib.simvec_search_vector.argtypes = [
    _engine_p,
    _c_str,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_uint64,
    _c_str,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    _c_out_str,
]
_lib.simvec_search_vector.restype = ctypes.c_int
_lib.simvec_get.argtypes = [_engine_p, _c_str, _c_str, _c_out_str]
_lib.simvec_get.restype = ctypes.c_int
_lib.simvec_count.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_uint64)]
_lib.simvec_count.restype = ctypes.c_int
_lib.simvec_scroll.argtypes = [
    _engine_p,
    _c_str,
    ctypes.c_uint64,
    _c_str,
    ctypes.c_int,
    ctypes.c_int,
    _c_out_str,
]
_lib.simvec_scroll.restype = ctypes.c_int

_lib.simvec_build_index.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_int)]
_lib.simvec_build_index.restype = ctypes.c_int
_lib.simvec_build_quantization.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_int)]
_lib.simvec_build_quantization.restype = ctypes.c_int
_lib.simvec_has_index.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_int)]
_lib.simvec_has_index.restype = ctypes.c_int
_lib.simvec_has_quantization.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_int)]
_lib.simvec_has_quantization.restype = ctypes.c_int
_lib.simvec_drop_index.argtypes = [_engine_p, _c_str]
_lib.simvec_drop_index.restype = ctypes.c_int
_lib.simvec_compact.argtypes = [_engine_p, _c_str, ctypes.POINTER(ctypes.c_uint64)]
_lib.simvec_compact.restype = ctypes.c_int
_lib.simvec_drain_pending.argtypes = [_engine_p, _c_str]
_lib.simvec_drain_pending.restype = ctypes.c_int
_lib.simvec_collection_metrics.argtypes = [_engine_p, _c_str, _c_out_str]
_lib.simvec_collection_metrics.restype = ctypes.c_int
_lib.simvec_run_maintenance_sweep.argtypes = [_engine_p]
_lib.simvec_run_maintenance_sweep.restype = ctypes.c_int


def lib():
    """Return the configured native library handle."""
    return _lib
