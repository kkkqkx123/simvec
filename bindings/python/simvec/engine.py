"""High-level engine wrapper over the simvec C API."""

import ctypes
import json

from . import _native
from . import errors

_lib = _native.lib()


def _encode_str(text):
    if text is None:
        return None
    if isinstance(text, str):
        return text.encode("utf-8")
    return text


def _encode_json(value):
    if value is None:
        return None
    return json.dumps(value).encode("utf-8")


def _take_string(out_param):
    # Read through the raw address: `.value` of a `c_char_p` would copy the
    # bytes first and freeing that copy would corrupt the heap.
    addr = ctypes.cast(out_param, ctypes.c_void_p).value
    if not addr:
        return ""
    try:
        return ctypes.string_at(addr).decode("utf-8")
    finally:
        _lib.simvec_free_string(addr)


def _take_json(out_param):
    return json.loads(_take_string(out_param))


def _check(code, handle=None):
    if code == errors.SIMVEC_OK:
        return
    message = ""
    if handle is not None:
        out = ctypes.c_void_p()
        if _lib.simvec_last_error(handle, ctypes.byref(out)) == errors.SIMVEC_OK:
            message = _take_string(out)
    errors.raise_for_code(code, message)


def _vector_array(vector):
    try:
        import numpy as np

        flat = np.ascontiguousarray(vector, dtype=np.float32).ravel()
        return flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), flat.size
    except ImportError:
        items = [float(v) for v in vector]
        array = (ctypes.c_float * len(items))(*items)
        return array, len(items)


class Engine:
    """Owner of one native engine handle rooted at a directory."""

    def __init__(self, path):
        handle = ctypes.c_void_p()
        code = _lib.simvec_open(_encode_str(str(path)), ctypes.byref(handle))
        if code != errors.SIMVEC_OK:
            raw = _lib.simvec_last_error_text()
            message = _take_string(raw)
            errors.raise_for_code(code, message)
        self._handle = handle

    def close(self):
        """Close the engine; safe to call more than once."""
        if self._handle is not None:
            _lib.simvec_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, * _exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- collections --

    def create_collection(self, name, config=None):
        _check(
            _lib.simvec_create_collection(
                self._handle, _encode_str(name), _encode_json(config)
            ),
            self._handle,
        )

    def delete_collection(self, name):
        _check(_lib.simvec_delete_collection(self._handle, _encode_str(name)), self._handle)

    def collection_exists(self, name):
        out = ctypes.c_int()
        _check(
            _lib.simvec_collection_exists(self._handle, _encode_str(name), ctypes.byref(out)),
            self._handle,
        )
        return bool(out.value)

    def list_collections(self):
        out = ctypes.c_void_p()
        _check(_lib.simvec_list_collections(self._handle, ctypes.byref(out)), self._handle)
        return _take_json(out)

    def collection_info(self, name):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_collection_info(self._handle, _encode_str(name), ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    def collection_config(self, name):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_collection_config(self._handle, _encode_str(name), ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    # -- writes --

    @property
    def last_code(self):
        """Most recent status code recorded on the native handle."""
        if self._handle is None:
            return errors.SIMVEC_MISUSE
        return _lib.simvec_errcode(self._handle)

    @property
    def last_message(self):
        """Most recent error text recorded on the native handle."""
        if self._handle is None:
            return "engine is closed"
        out = ctypes.c_void_p()
        if _lib.simvec_last_error(self._handle, ctypes.byref(out)) != errors.SIMVEC_OK:
            return ""
        return _take_string(out)

    def upsert(self, collection, point_id, vector, payload=None):
        data, size = _vector_array(vector)
        _check(
            _lib.simvec_upsert(
                self._handle,
                _encode_str(collection),
                _encode_str(str(point_id)),
                data,
                size,
                _encode_json(payload),
            ),
            self._handle,
        )

    def upsert_batch(self, collection, points):
        _check(
            _lib.simvec_upsert_batch(
                self._handle, _encode_str(collection), _encode_json(points)
            ),
            self._handle,
        )

    def delete(self, collection, point_id):
        _check(
            _lib.simvec_delete(self._handle, _encode_str(collection), _encode_str(str(point_id))),
            self._handle,
        )

    def delete_batch(self, collection, point_ids):
        _check(
            _lib.simvec_delete_batch(
                self._handle, _encode_str(collection), _encode_json([str(i) for i in point_ids])
            ),
            self._handle,
        )

    def delete_by_filter(self, collection, query_filter):
        out = ctypes.c_uint64()
        _check(
            _lib.simvec_delete_by_filter(
                self._handle,
                _encode_str(collection),
                _encode_json(query_filter),
                ctypes.byref(out),
            ),
            self._handle,
        )
        return out.value

    def set_payload(self, collection, point_id, payload):
        _check(
            _lib.simvec_set_payload(
                self._handle,
                _encode_str(collection),
                _encode_str(str(point_id)),
                _encode_json(payload),
            ),
            self._handle,
        )

    def merge_payload(self, collection, point_id, fields):
        _check(
            _lib.simvec_merge_payload(
                self._handle,
                _encode_str(collection),
                _encode_str(str(point_id)),
                _encode_json(fields),
            ),
            self._handle,
        )

    def delete_payload_keys(self, collection, point_id, keys):
        _check(
            _lib.simvec_delete_payload_keys(
                self._handle,
                _encode_str(collection),
                _encode_str(str(point_id)),
                _encode_json(list(keys)),
            ),
            self._handle,
        )

    def set_payload_field(self, collection, point_id, key, value):
        _check(
            _lib.simvec_set_payload_field(
                self._handle,
                _encode_str(collection),
                _encode_str(str(point_id)),
                _encode_str(key),
                _encode_json(value),
            ),
            self._handle,
        )

    def apply_txn(self, ops):
        _check(
            _lib.simvec_apply_txn(self._handle, _encode_json(ops)),
            self._handle,
        )

    def create_payload_index(self, collection, field, schema):
        _check(
            _lib.simvec_create_payload_index(
                self._handle, _encode_str(collection), _encode_str(field), _encode_str(schema)
            ),
            self._handle,
        )

    def delete_payload_index(self, collection, field):
        out = ctypes.c_int()
        _check(
            _lib.simvec_delete_payload_index(
                self._handle, _encode_str(collection), _encode_str(field), ctypes.byref(out)
            ),
            self._handle,
        )
        return bool(out.value)

    def list_payload_indexes(self, collection):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_list_payload_indexes(self._handle, _encode_str(collection), ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    # -- reads --

    def search(self, collection, query):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_search(
                self._handle, _encode_str(collection), _encode_json(query), ctypes.byref(out)
            ),
            self._handle,
        )
        return _take_json(out)

    def search_vector(self, collection, vector, limit, query_filter=None,
                      score_threshold=-1.0, with_payload=True, with_vector=False):
        data, size = _vector_array(vector)
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_search_vector(
                self._handle,
                _encode_str(collection),
                data,
                size,
                limit,
                _encode_json(query_filter),
                score_threshold,
                int(bool(with_payload)),
                int(bool(with_vector)),
                ctypes.byref(out),
            ),
            self._handle,
        )
        return _take_json(out)

    def get(self, collection, point_id):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_get(
                self._handle, _encode_str(collection), _encode_str(str(point_id)), ctypes.byref(out)
            ),
            self._handle,
        )
        return _take_json(out)

    def count(self, collection):
        out = ctypes.c_uint64()
        _check(
            _lib.simvec_count(self._handle, _encode_str(collection), ctypes.byref(out)),
            self._handle,
        )
        return out.value

    def count_by_filter(self, collection, query_filter):
        out = ctypes.c_uint64()
        _check(
            _lib.simvec_count_by_filter(
                self._handle,
                _encode_str(collection),
                _encode_json(query_filter),
                ctypes.byref(out),
            ),
            self._handle,
        )
        return out.value

    def scroll(self, collection, limit, offset=None, with_payload=True, with_vector=False):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_scroll(
                self._handle,
                _encode_str(collection),
                limit,
                _encode_str(offset) if offset is not None else None,
                int(bool(with_payload)),
                int(bool(with_vector)),
                ctypes.byref(out),
            ),
            self._handle,
        )
        return _take_json(out)

    def scroll_by_filter(self, collection, query_filter, limit, offset=None,
                         with_payload=True, with_vector=False):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_scroll_by_filter(
                self._handle,
                _encode_str(collection),
                limit,
                _encode_str(offset) if offset is not None else None,
                _encode_json(query_filter),
                int(bool(with_payload)),
                int(bool(with_vector)),
                ctypes.byref(out),
            ),
            self._handle,
        )
        return _take_json(out)

    # -- maintenance --

    def build_index(self, collection):
        out = ctypes.c_int()
        _check(
            _lib.simvec_build_index(
                self._handle, _encode_str(collection), ctypes.byref(out)
            ),
            self._handle,
        )
        return bool(out.value)

    def build_quantization(self, collection):
        out = ctypes.c_int()
        _check(
            _lib.simvec_build_quantization(
                self._handle, _encode_str(collection), ctypes.byref(out)
            ),
            self._handle,
        )
        return bool(out.value)

    def has_index(self, collection):
        out = ctypes.c_int()
        _check(
            _lib.simvec_has_index(
                self._handle, _encode_str(collection), ctypes.byref(out)
            ),
            self._handle,
        )
        return bool(out.value)

    def has_quantization(self, collection):
        out = ctypes.c_int()
        _check(
            _lib.simvec_has_quantization(
                self._handle, _encode_str(collection), ctypes.byref(out)
            ),
            self._handle,
        )
        return bool(out.value)

    def drop_index(self, collection):
        _check(_lib.simvec_drop_index(self._handle, _encode_str(collection)), self._handle)

    def drop_published_index(self, collection):
        _check(
            _lib.simvec_drop_published_index(self._handle, _encode_str(collection)),
            self._handle,
        )

    def drain_pending(self, collection):
        _check(
            _lib.simvec_drain_pending(self._handle, _encode_str(collection)),
            self._handle,
        )

    def compact(self, collection):
        out = ctypes.c_uint64()
        _check(
            _lib.simvec_compact(self._handle, _encode_str(collection), ctypes.byref(out)),
            self._handle,
        )
        return out.value

    def collection_metrics(self, collection):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_collection_metrics(
                self._handle, _encode_str(collection), ctypes.byref(out)
            ),
            self._handle,
        )
        return _take_json(out)

    def run_maintenance_sweep(self):
        _check(_lib.simvec_run_maintenance_sweep(self._handle), self._handle)

    def promote_temp_collection(self, live, temp, backup_suffix):
        _check(
            _lib.simvec_promote_temp_collection(
                self._handle,
                _encode_str(live),
                _encode_str(temp),
                _encode_str(backup_suffix),
            ),
            self._handle,
        )

    def restore_promote_backup(self, live):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_restore_promote_backup(
                self._handle, _encode_str(live), ctypes.byref(out)
            ),
            self._handle,
        )
        return _take_json(out)

    def list_temp_collections(self):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_list_temp_collections(self._handle, ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    def list_old_backups(self, live):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_list_old_backups(
                self._handle, _encode_str(live), ctypes.byref(out)
            ),
            self._handle,
        )
        return _take_json(out)

    def set_default_hnsw_config(self, config):
        _check(
            _lib.simvec_set_default_hnsw_config(self._handle, _encode_json(config)),
            self._handle,
        )

    def get_default_hnsw_config(self):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_get_default_hnsw_config(self._handle, ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    def set_default_ivf_config(self, config):
        _check(
            _lib.simvec_set_default_ivf_config(self._handle, _encode_json(config)),
            self._handle,
        )

    def get_default_ivf_config(self):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_get_default_ivf_config(self._handle, ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)

    def set_default_quantization_config(self, config):
        _check(
            _lib.simvec_set_default_quantization_config(self._handle, _encode_json(config)),
            self._handle,
        )

    def get_default_quantization_config(self):
        out = ctypes.c_void_p()
        _check(
            _lib.simvec_get_default_quantization_config(self._handle, ctypes.byref(out)),
            self._handle,
        )
        return _take_json(out)
