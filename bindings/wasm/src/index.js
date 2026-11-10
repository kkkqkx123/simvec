import init, { WasmEngine } from "../pkg/simvec.js";
import { MisuseError, raiseForWasm } from "./errors.js";

let started = null;

async function ensureReady() {
  if (!started) {
    started = init();
  }
  await started;
}

function encodeJson(value) {
  if (value === undefined || value === null) {
    return null;
  }
  return JSON.stringify(value);
}

function toArray(vector) {
  if (vector instanceof Float32Array) {
    return Array.from(vector);
  }
  return Array.from(vector, (v) => Number(v));
}

function callWasm(fn) {
  try {
    return fn();
  } catch (error) {
    raiseForWasm(error);
  }
}

function parseWasmJson(fn) {
  const text = callWasm(fn);
  return JSON.parse(text);
}

export class Engine {
  static async create() {
    await ensureReady();
    return new Engine();
  }

  constructor() {
    this._handle = new WasmEngine();
    this._closed = false;
  }

  get closed() {
    return this._closed;
  }

  _requireOpen() {
    if (this._closed || this._handle === null || this._handle === undefined) {
      throw new MisuseError("[simvec:20] engine is closed", 20);
    }
    return this._handle;
  }

  close() {
    if (this._closed || this._handle === null || this._handle === undefined) {
      return;
    }
    if (typeof this._handle.free === "function") {
      this._handle.free();
    }
    this._handle = null;
    this._closed = true;
  }

  [Symbol.dispose]() {
    this.close();
  }

  createCollection(name, config = null) {
    const handle = this._requireOpen();
    callWasm(() => handle.create_collection(String(name), encodeJson(config)));
  }

  deleteCollection(name) {
    const handle = this._requireOpen();
    callWasm(() => handle.delete_collection(String(name)));
  }

  collectionExists(name) {
    const handle = this._requireOpen();
    return callWasm(() => handle.collection_exists(String(name)));
  }

  listCollections() {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.list_collections());
  }

  collectionInfo(name) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.collection_info(String(name)));
  }

  collectionConfig(name) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.collection_config(String(name)));
  }

  upsert(collection, id, vector, payload = null) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.upsert(String(collection), String(id), toArray(vector), encodeJson(payload)),
    );
  }

  upsertBatch(collection, points) {
    const handle = this._requireOpen();
    callWasm(() => handle.upsert_batch(String(collection), JSON.stringify(points)));
  }

  delete(collection, id) {
    const handle = this._requireOpen();
    callWasm(() => handle.delete(String(collection), String(id)));
  }

  deleteBatch(collection, ids) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.delete_batch(String(collection), JSON.stringify(Array.from(ids, (id) => String(id)))),
    );
  }

  deleteByFilter(collection, filter) {
    const handle = this._requireOpen();
    return callWasm(() => handle.delete_by_filter(String(collection), JSON.stringify(filter)));
  }

  applyTxn(ops) {
    const handle = this._requireOpen();
    callWasm(() => handle.apply_txn(JSON.stringify(ops)));
  }

  setPayload(collection, id, payload) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.set_payload(String(collection), String(id), JSON.stringify(payload)),
    );
  }

  mergePayload(collection, id, fields) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.merge_payload(String(collection), String(id), JSON.stringify(fields)),
    );
  }

  setPayloadField(collection, id, key, value) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.set_payload_field(
        String(collection),
        String(id),
        String(key),
        JSON.stringify(value),
      ),
    );
  }

  deletePayloadKeys(collection, id, keys) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.delete_payload_keys(
        String(collection),
        String(id),
        JSON.stringify(Array.from(keys)),
      ),
    );
  }

  createPayloadIndex(collection, field, schema) {
    const handle = this._requireOpen();
    callWasm(() =>
      handle.create_payload_index(String(collection), String(field), String(schema)),
    );
  }

  deletePayloadIndex(collection, field) {
    const handle = this._requireOpen();
    return callWasm(() => handle.delete_payload_index(String(collection), String(field)));
  }

  listPayloadIndexes(collection) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.list_payload_indexes(String(collection)));
  }

  search(collection, query) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.search(String(collection), JSON.stringify(query)));
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
    const query = { vector: toArray(vector), limit: Number(limit) };
    if (filter !== null && filter !== undefined) {
      query.filter = filter;
    }
    if (Number(scoreThreshold) >= 0) {
      query.score_threshold = Number(scoreThreshold);
    }
    query.with_payload = Boolean(withPayload);
    query.with_vector = Boolean(withVector);
    return this.search(collection, query);
  }

  get(collection, id) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.get(String(collection), String(id)));
  }

  count(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.count(String(collection)));
  }

  countByFilter(collection, filter) {
    const handle = this._requireOpen();
    return callWasm(() => handle.count_by_filter(String(collection), JSON.stringify(filter)));
  }

  scroll(collection, limit, offset = null, withPayload = true, withVector = false) {
    const handle = this._requireOpen();
    return parseWasmJson(() =>
      handle.scroll(
        String(collection),
        Number(limit),
        offset === null || offset === undefined ? null : String(offset),
        Boolean(withPayload),
        Boolean(withVector),
      ),
    );
  }

  scrollByFilter(collection, filter, limit, offset = null, withPayload = true, withVector = false) {
    const handle = this._requireOpen();
    return parseWasmJson(() =>
      handle.scroll_by_filter(
        String(collection),
        JSON.stringify(filter),
        Number(limit),
        offset === null || offset === undefined ? null : String(offset),
        Boolean(withPayload),
        Boolean(withVector),
      ),
    );
  }

  buildIndex(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.build_index(String(collection)));
  }

  buildQuantization(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.build_quantization(String(collection)));
  }

  hasIndex(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.has_index(String(collection)));
  }

  hasQuantization(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.has_quantization(String(collection)));
  }

  dropIndex(collection) {
    const handle = this._requireOpen();
    callWasm(() => handle.drop_index(String(collection)));
  }

  dropPublishedIndex(collection) {
    const handle = this._requireOpen();
    callWasm(() => handle.drop_published_index(String(collection)));
  }

  compact(collection) {
    const handle = this._requireOpen();
    return callWasm(() => handle.compact(String(collection)));
  }

  drainPending(collection) {
    const handle = this._requireOpen();
    callWasm(() => handle.drain_pending(String(collection)));
  }

  collectionMetrics(collection) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.collection_metrics(String(collection)));
  }

  runMaintenanceSweep() {
    const handle = this._requireOpen();
    callWasm(() => handle.run_maintenance_sweep());
  }

  exportSnapshot() {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.export_snapshot());
  }

  importSnapshot(snapshot) {
    const handle = this._requireOpen();
    const text = typeof snapshot === "string" ? snapshot : JSON.stringify(snapshot);
    callWasm(() => handle.import_snapshot(text));
  }

  revision() {
    const handle = this._requireOpen();
    return callWasm(() => handle.revision());
  }

  exportCollection(name) {
    const handle = this._requireOpen();
    return parseWasmJson(() => handle.export_collection(String(name)));
  }

  importCollection(snapshot) {
    const handle = this._requireOpen();
    const text = typeof snapshot === "string" ? snapshot : JSON.stringify(snapshot);
    callWasm(() => handle.import_collection(text));
  }
}
