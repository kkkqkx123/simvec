import { ffi, functions } from "./native.js";
import { MisuseError, SIMVEC_OK, raiseForCode } from "./errors.js";

function encodeJson(value) {
  if (value === undefined || value === null) {
    return null;
  }
  return JSON.stringify(value);
}

function takeString(ptr) {
  if (ptr === 0n || ptr === null || ptr === undefined) {
    return "";
  }
  try {
    return ffi.toString(ptr) ?? "";
  } finally {
    functions.simvec_free_string(ptr);
  }
}

function takeJson(ptr) {
  const text = takeString(ptr);
  if (!text) {
    return null;
  }
  return JSON.parse(text);
}

function check(code, handle) {
  if (code === SIMVEC_OK) {
    return;
  }
  let message = "";
  if (handle !== null && handle !== undefined) {
    const out = Buffer.alloc(8);
    if (functions.simvec_last_error(handle, out) === SIMVEC_OK) {
      message = takeString(out.readBigUInt64LE(0));
    }
  }
  raiseForCode(code, message);
}

function checkNoHandle(code) {
  if (code === SIMVEC_OK) {
    return;
  }
  let message = "";
  const ptr = functions.simvec_last_error_text();
  try {
    message = ptr === 0n ? "" : (ffi.toString(ptr) ?? "");
  } finally {
    if (ptr !== 0n) {
      functions.simvec_free_string(ptr);
    }
  }
  raiseForCode(code, message);
}

function toFloat32(vector) {
  if (vector instanceof Float32Array) {
    return vector;
  }
  return Float32Array.from(vector, (v) => Number(v));
}

function outPointer() {
  return Buffer.alloc(8);
}

function readPointer(out) {
  return out.readBigUInt64LE(0);
}

function outInt32() {
  return Buffer.alloc(4);
}

export class Engine {
  constructor(dirPath) {
    const out = outPointer();
    const code = functions.simvec_open(String(dirPath), out);
    if (code !== SIMVEC_OK) {
      checkNoHandle(code);
    }
    this._handle = readPointer(out);
    this._closed = false;
  }

  get closed() {
    return this._closed;
  }

  _requireOpen() {
    if (this._closed || this._handle === null) {
      throw new MisuseError("[simvec:20] engine is closed", 20);
    }
    return this._handle;
  }

  close() {
    if (this._closed || this._handle === null) {
      return;
    }
    functions.simvec_close(this._handle);
    this._handle = null;
    this._closed = true;
  }

  get lastCode() {
    if (this._closed || this._handle === null) {
      return 20;
    }
    return functions.simvec_errcode(this._handle);
  }

  get lastMessage() {
    if (this._closed || this._handle === null) {
      return "engine is closed";
    }
    const out = outPointer();
    if (functions.simvec_last_error(this._handle, out) !== SIMVEC_OK) {
      return "";
    }
    return takeString(readPointer(out));
  }

  [Symbol.dispose]() {
    this.close();
  }

  createCollection(name, config = null) {
    const handle = this._requireOpen();
    check(functions.simvec_create_collection(handle, String(name), encodeJson(config)), handle);
  }

  deleteCollection(name) {
    const handle = this._requireOpen();
    check(functions.simvec_delete_collection(handle, String(name)), handle);
  }

  collectionExists(name) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(functions.simvec_collection_exists(handle, String(name), out), handle);
    return out.readInt32LE(0) !== 0;
  }

  listCollections() {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_list_collections(handle, out), handle);
    return takeJson(readPointer(out));
  }

  collectionInfo(name) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_collection_info(handle, String(name), out), handle);
    return takeJson(readPointer(out));
  }

  collectionConfig(name) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_collection_config(handle, String(name), out), handle);
    return takeJson(readPointer(out));
  }

  upsert(collection, id, vector, payload = null) {
    const handle = this._requireOpen();
    const data = toFloat32(vector);
    check(
      functions.simvec_upsert(
        handle,
        String(collection),
        String(id),
        data,
        BigInt(data.length),
        encodeJson(payload),
      ),
      handle,
    );
  }

  upsertBatch(collection, points) {
    const handle = this._requireOpen();
    check(
      functions.simvec_upsert_batch(handle, String(collection), encodeJson(points)),
      handle,
    );
  }

  delete(collection, id) {
    const handle = this._requireOpen();
    check(functions.simvec_delete(handle, String(collection), String(id)), handle);
  }

  deleteBatch(collection, ids) {
    const handle = this._requireOpen();
    check(
      functions.simvec_delete_batch(
        handle,
        String(collection),
        encodeJson(Array.from(ids, (id) => String(id))),
      ),
      handle,
    );
  }

  deleteByFilter(collection, filter) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(
      functions.simvec_delete_by_filter(handle, String(collection), encodeJson(filter), out),
      handle,
    );
    return Number(out.readBigUInt64LE(0));
  }

  setPayload(collection, id, payload) {
    const handle = this._requireOpen();
    check(
      functions.simvec_set_payload(
        handle,
        String(collection),
        String(id),
        encodeJson(payload),
      ),
      handle,
    );
  }

  mergePayload(collection, id, fields) {
    const handle = this._requireOpen();
    check(
      functions.simvec_merge_payload(
        handle,
        String(collection),
        String(id),
        encodeJson(fields),
      ),
      handle,
    );
  }

  setPayloadField(collection, id, key, value) {
    const handle = this._requireOpen();
    check(
      functions.simvec_set_payload_field(
        handle,
        String(collection),
        String(id),
        String(key),
        encodeJson(value),
      ),
      handle,
    );
  }

  applyTxn(ops) {
    const handle = this._requireOpen();
    check(functions.simvec_apply_txn(handle, encodeJson(ops)), handle);
  }

  deletePayloadKeys(collection, id, keys) {
    const handle = this._requireOpen();
    check(
      functions.simvec_delete_payload_keys(
        handle,
        String(collection),
        String(id),
        encodeJson(Array.from(keys)),
      ),
      handle,
    );
  }

  createPayloadIndex(collection, field, schema) {
    const handle = this._requireOpen();
    check(
      functions.simvec_create_payload_index(
        handle,
        String(collection),
        String(field),
        String(schema),
      ),
      handle,
    );
  }

  deletePayloadIndex(collection, field) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(
      functions.simvec_delete_payload_index(handle, String(collection), String(field), out),
      handle,
    );
    return out.readInt32LE(0) !== 0;
  }

  listPayloadIndexes(collection) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_list_payload_indexes(handle, String(collection), out), handle);
    return takeJson(readPointer(out));
  }

  search(collection, query) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(
      functions.simvec_search(handle, String(collection), encodeJson(query), out),
      handle,
    );
    return takeJson(readPointer(out));
  }

  searchVector(
    collection,
    vector,
    limit,
    filter = null,
    scoreThreshold = -1,
    withPayload = true,
    withVector = false,
  ) {
    const handle = this._requireOpen();
    const data = toFloat32(vector);
    const out = outPointer();
    check(
      functions.simvec_search_vector(
        handle,
        String(collection),
        data,
        BigInt(data.length),
        BigInt(limit),
        encodeJson(filter),
        Number(scoreThreshold),
        withPayload ? 1 : 0,
        withVector ? 1 : 0,
        out,
      ),
      handle,
    );
    return takeJson(readPointer(out));
  }

  get(collection, id) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_get(handle, String(collection), String(id), out), handle);
    return takeJson(readPointer(out));
  }

  count(collection) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_count(handle, String(collection), out), handle);
    return Number(out.readBigUInt64LE(0));
  }

  scroll(collection, limit, offset = null, withPayload = true, withVector = false) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(
      functions.simvec_scroll(
        handle,
        String(collection),
        BigInt(limit),
        offset === null || offset === undefined ? null : String(offset),
        withPayload ? 1 : 0,
        withVector ? 1 : 0,
        out,
      ),
      handle,
    );
    return takeJson(readPointer(out));
  }

  buildIndex(collection) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(functions.simvec_build_index(handle, String(collection), out), handle);
    return out.readInt32LE(0) !== 0;
  }

  buildQuantization(collection) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(functions.simvec_build_quantization(handle, String(collection), out), handle);
    return out.readInt32LE(0) !== 0;
  }

  hasIndex(collection) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(functions.simvec_has_index(handle, String(collection), out), handle);
    return out.readInt32LE(0) !== 0;
  }

  hasQuantization(collection) {
    const handle = this._requireOpen();
    const out = outInt32();
    check(functions.simvec_has_quantization(handle, String(collection), out), handle);
    return out.readInt32LE(0) !== 0;
  }

  dropIndex(collection) {
    const handle = this._requireOpen();
    check(functions.simvec_drop_index(handle, String(collection)), handle);
  }

  dropPublishedIndex(collection) {
    const handle = this._requireOpen();
    check(functions.simvec_drop_published_index(handle, String(collection)), handle);
  }

  compact(collection) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_compact(handle, String(collection), out), handle);
    return Number(out.readBigUInt64LE(0));
  }

  drainPending(collection) {
    const handle = this._requireOpen();
    check(functions.simvec_drain_pending(handle, String(collection)), handle);
  }

  collectionMetrics(collection) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_collection_metrics(handle, String(collection), out), handle);
    return takeJson(readPointer(out));
  }

  runMaintenanceSweep() {
    const handle = this._requireOpen();
    check(functions.simvec_run_maintenance_sweep(handle), handle);
  }

  promoteTempCollection(live, temp, backupSuffix) {
    const handle = this._requireOpen();
    check(
      functions.simvec_promote_temp_collection(
        handle,
        String(live),
        String(temp),
        String(backupSuffix),
      ),
      handle,
    );
  }

  restorePromoteBackup(live) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(
      functions.simvec_restore_promote_backup(handle, String(live), out),
      handle,
    );
    return takeJson(readPointer(out));
  }

  listTempCollections() {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_list_temp_collections(handle, out), handle);
    return takeJson(readPointer(out));
  }

  listOldBackups(live) {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_list_old_backups(handle, String(live), out), handle);
    return takeJson(readPointer(out));
  }

  setDefaultHnswConfig(config) {
    const handle = this._requireOpen();
    check(functions.simvec_set_default_hnsw_config(handle, encodeJson(config)), handle);
  }

  getDefaultHnswConfig() {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_get_default_hnsw_config(handle, out), handle);
    return takeJson(readPointer(out));
  }

  setDefaultIvfConfig(config) {
    const handle = this._requireOpen();
    check(functions.simvec_set_default_ivf_config(handle, encodeJson(config)), handle);
  }

  getDefaultIvfConfig() {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_get_default_ivf_config(handle, out), handle);
    return takeJson(readPointer(out));
  }

  setDefaultQuantizationConfig(config) {
    const handle = this._requireOpen();
    check(
      functions.simvec_set_default_quantization_config(handle, encodeJson(config)),
      handle,
    );
  }

  getDefaultQuantizationConfig() {
    const handle = this._requireOpen();
    const out = outPointer();
    check(functions.simvec_get_default_quantization_config(handle, out), handle);
    return takeJson(readPointer(out));
  }
}
