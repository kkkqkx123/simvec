import ffi, { dlopen } from "node:ffi";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SimvecError } from "./errors.js";

function* candidatePaths() {
  const explicit = process.env.SIMVEC_LIB;
  if (explicit) {
    yield explicit;
  }
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(here, "..", "..", "..");
  for (const profile of ["debug", "release"]) {
    yield path.join(root, "target", profile, "libsimvec.so");
    yield path.join(root, "target", profile, "libsimvec.dylib");
    yield path.join(root, "target", profile, "simvec.dll");
  }
}

const DEFINITIONS = {
  simvec_open: { arguments: ["string", "pointer"], return: "int32" },
  simvec_close: { arguments: ["pointer"], return: "int32" },
  simvec_free_string: { arguments: ["pointer"], return: "void" },
  simvec_errcode: { arguments: ["pointer"], return: "int32" },
  simvec_last_error: { arguments: ["pointer", "pointer"], return: "int32" },
  simvec_last_error_text: { arguments: [], return: "pointer" },
  simvec_create_collection: { arguments: ["pointer", "string", "string"], return: "int32" },
  simvec_delete_collection: { arguments: ["pointer", "string"], return: "int32" },
  simvec_collection_exists: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_list_collections: { arguments: ["pointer", "pointer"], return: "int32" },
  simvec_collection_info: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_collection_config: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_upsert: {
    arguments: ["pointer", "string", "string", "pointer", "uint64", "string"],
    return: "int32",
  },
  simvec_upsert_batch: { arguments: ["pointer", "string", "string"], return: "int32" },
  simvec_delete: { arguments: ["pointer", "string", "string"], return: "int32" },
  simvec_delete_batch: { arguments: ["pointer", "string", "string"], return: "int32" },
  simvec_delete_by_filter: {
    arguments: ["pointer", "string", "string", "pointer"],
    return: "int32",
  },
  simvec_set_payload: {
    arguments: ["pointer", "string", "string", "string"],
    return: "int32",
  },
  simvec_merge_payload: {
    arguments: ["pointer", "string", "string", "string"],
    return: "int32",
  },
  simvec_delete_payload_keys: {
    arguments: ["pointer", "string", "string", "string"],
    return: "int32",
  },
  simvec_create_payload_index: {
    arguments: ["pointer", "string", "string", "string"],
    return: "int32",
  },
  simvec_delete_payload_index: {
    arguments: ["pointer", "string", "string", "pointer"],
    return: "int32",
  },
  simvec_list_payload_indexes: {
    arguments: ["pointer", "string", "pointer"],
    return: "int32",
  },
  simvec_search: {
    arguments: ["pointer", "string", "string", "pointer"],
    return: "int32",
  },
  simvec_search_vector: {
    arguments: [
      "pointer",
      "string",
      "pointer",
      "uint64",
      "uint64",
      "string",
      "float32",
      "int32",
      "int32",
      "pointer",
    ],
    return: "int32",
  },
  simvec_get: { arguments: ["pointer", "string", "string", "pointer"], return: "int32" },
  simvec_count: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_scroll: {
    arguments: ["pointer", "string", "uint64", "string", "int32", "int32", "pointer"],
    return: "int32",
  },
  simvec_build_index: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_build_quantization: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_has_index: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_has_quantization: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_drop_index: { arguments: ["pointer", "string"], return: "int32" },
  simvec_compact: { arguments: ["pointer", "string", "pointer"], return: "int32" },
  simvec_drain_pending: { arguments: ["pointer", "string"], return: "int32" },
  simvec_collection_metrics: {
    arguments: ["pointer", "string", "pointer"],
    return: "int32",
  },
  simvec_run_maintenance_sweep: { arguments: ["pointer"], return: "int32" },
};

function loadLibrary() {
  const tried = [];
  for (const candidate of candidatePaths()) {
    if (candidate && fs.existsSync(candidate)) {
      return dlopen(candidate, DEFINITIONS);
    }
    tried.push(String(candidate));
  }
  throw new SimvecError(
    `libsimvec not found; set SIMVEC_LIB or run cargo build --features c_api. Tried: ${tried.join(", ")}`,
    0,
  );
}

const handle = loadLibrary();

export const lib = handle.lib;
export const functions = handle.functions;
export { ffi };
