function requireOpfsStorage() {
  const storage = globalThis.navigator?.storage;
  if (!storage) {
    throw new Error("OPFS storage is unavailable in this host");
  }
  return storage;
}

export function requireOpfsDirectory() {
  const storage = requireOpfsStorage();
  if (typeof storage.getDirectory !== "function") {
    throw new Error("OPFS storage directory is unavailable in this host");
  }
  return storage.getDirectory();
}

function toSnapshotText(snapshot) {
  return typeof snapshot === "string" ? snapshot : JSON.stringify(snapshot);
}

function parseSnapshotObject(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`OPFS snapshot is corrupt: ${String(error)}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("OPFS snapshot is corrupt: expected an object envelope");
  }
  return parsed;
}

function verifySnapshotText(text) {
  const parsed = parseSnapshotObject(text);
  if (parsed.version !== 1 || !Array.isArray(parsed.collections)) {
    throw new Error("OPFS snapshot is corrupt: expected version 1 with a collections array");
  }
  return parsed;
}

function verifyCollectionText(text) {
  const parsed = parseSnapshotObject(text);
  if (typeof parsed.name !== "string" || !parsed.config || !Array.isArray(parsed.points)) {
    throw new Error("OPFS collection snapshot is corrupt: expected name, config and points");
  }
  return parsed;
}

function verifyManifestText(text) {
  const parsed = parseSnapshotObject(text);
  if (parsed.version !== 1 || !parsed.files || typeof parsed.files !== "object") {
    throw new Error("OPFS manifest is corrupt: expected version 1 with a files map");
  }
  return parsed;
}

async function writeHandleText(handle, text) {
  const writable = await handle.createWritable();
  try {
    await writable.write(text);
    await writable.close();
  } catch (error) {
    try {
      await writable.abort();
    } catch {}
    throw error;
  }
}

async function readHandleText(handle) {
  const file = await handle.getFile();
  return file.text();
}

async function writeFileAtomic(root, targetName, text, verify) {
  const name = String(targetName);
  const tmpName = `${name}.tmp`;
  const backupName = `${name}.bak`;
  const tmpHandle = await root.getFileHandle(tmpName, { create: true });
  try {
    await writeHandleText(tmpHandle, text);
  } catch (error) {
    throw new Error(`OPFS save failed: ${String(error)}`);
  }
  let verifiedTmp;
  try {
    verifiedTmp = await readHandleText(tmpHandle);
  } catch (error) {
    throw new Error(`OPFS save failed: cannot read back temp file: ${String(error)}`);
  }
  try {
    verify(verifiedTmp);
  } catch (error) {
    try {
      await root.removeEntry(tmpName);
    } catch {}
    throw error;
  }
  try {
    const current = await root.getFileHandle(name, { create: false });
    const currentText = await readHandleText(current);
    try {
      verify(currentText);
      const backup = await root.getFileHandle(backupName, { create: true });
      await writeHandleText(backup, currentText);
    } catch {}
  } catch {}
  if (typeof tmpHandle.move === "function") {
    try {
      await tmpHandle.move(name);
      const handle = await root.getFileHandle(name, { create: false });
      const roundtrip = await readHandleText(handle);
      verify(roundtrip);
      return;
    } catch (error) {
      if (String(error?.message ?? error).includes("corrupt")) {
        throw error;
      }
    }
  }
  const handle = await root.getFileHandle(name, { create: true });
  try {
    await writeHandleText(handle, text);
  } catch (error) {
    throw new Error(`OPFS save failed: ${String(error)}`);
  }
  const roundtrip = await readHandleText(handle);
  verify(roundtrip);
  try {
    await root.removeEntry(tmpName);
  } catch {}
}

export async function saveSnapshotToOpfs(engine, fileName = "simvec-snapshot.json") {
  const text = toSnapshotText(engine.exportSnapshot());
  verifySnapshotText(text);
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS save unavailable, snapshot kept in memory only: ${String(error)}`);
  }
  try {
    await writeFileAtomic(root, String(fileName), text, verifySnapshotText);
  } catch (error) {
    if (String(error?.message ?? error).includes("OPFS ")) {
      throw error;
    }
    throw new Error(`OPFS save failed: ${String(error)}`);
  }
}

export async function loadSnapshotFromOpfs(engine, fileName = "simvec-snapshot.json") {
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS load unavailable: ${String(error)}`);
  }
  let text;
  try {
    const handle = await root.getFileHandle(String(fileName), { create: false });
    text = await readHandleText(handle);
  } catch (error) {
    throw new Error(`OPFS snapshot not found: ${String(error)}`);
  }
  const parsed = verifySnapshotText(text);
  engine.importSnapshot(parsed);
}

export async function snapshotExists(fileName = "simvec-snapshot.json") {
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS check unavailable: ${String(error)}`);
  }
  try {
    const handle = await root.getFileHandle(String(fileName), { create: false });
    await handle.getFile();
    return true;
  } catch {
    return false;
  }
}

export async function removeSnapshot(fileName = "simvec-snapshot.json") {
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS remove unavailable: ${String(error)}`);
  }
  try {
    await root.removeEntry(String(fileName));
    return true;
  } catch {
    return false;
  }
}

export async function estimateOpfsQuota() {
  const storage = requireOpfsStorage();
  if (typeof storage.estimate !== "function") {
    throw new Error("OPFS quota estimate is unavailable in this host");
  }
  return storage.estimate();
}

export async function requestOpfsPersistence() {
  const storage = requireOpfsStorage();
  if (typeof storage.persist !== "function") {
    throw new Error("OPFS persistence request is unavailable in this host");
  }
  return storage.persist();
}

const MUTATING_METHODS = [
  "createCollection",
  "deleteCollection",
  "upsert",
  "upsertBatch",
  "delete",
  "deleteBatch",
  "deleteByFilter",
  "applyTxn",
  "setPayload",
  "mergePayload",
  "setPayloadField",
  "deletePayloadKeys",
  "createPayloadIndex",
  "deletePayloadIndex",
  "importSnapshot",
  "importCollection",
];

export function createOpfsAutoSave(engine, options = {}) {
  const fileName = String(options.fileName ?? "simvec-snapshot.json");
  const debounceMs = Number(options.debounceMs ?? 1000);
  const onError = options.onError ?? null;
  const subscribePageHide = options.subscribePageHide ?? true;
  const hasRevision = typeof engine.revision === "function";
  let lastSavedRevision = hasRevision ? engine.revision() : null;
  let fallbackDirty = false;
  let timer = null;
  let disposed = false;
  const originals = new Map();

  async function saveIfDirty() {
    if (disposed) {
      return false;
    }
    if (hasRevision) {
      const current = engine.revision();
      if (current === lastSavedRevision) {
        return false;
      }
      await saveSnapshotToOpfs(engine, fileName);
      lastSavedRevision = engine.revision();
      return true;
    }
    if (!fallbackDirty) {
      return false;
    }
    await saveSnapshotToOpfs(engine, fileName);
    fallbackDirty = false;
    return true;
  }

  function schedule() {
    if (disposed) {
      return;
    }
    if (timer !== null) {
      globalThis.clearTimeout(timer);
    }
    timer = globalThis.setTimeout(() => {
      timer = null;
      saveIfDirty().catch((error) => {
        if (typeof onError === "function") {
          onError(error);
        }
      });
    }, Number.isFinite(debounceMs) && debounceMs >= 0 ? debounceMs : 0);
  }

  for (const method of MUTATING_METHODS) {
    if (typeof engine[method] !== "function") {
      continue;
    }
    const original = engine[method].bind(engine);
    originals.set(method, engine[method]);
    engine[method] = (...args) => {
      const result = original(...args);
      if (!hasRevision) {
        fallbackDirty = true;
      }
      schedule();
      return result;
    };
  }

  const flushOnHide = () => {
    saveIfDirty().catch((error) => {
      if (typeof onError === "function") {
        onError(error);
      }
    });
  };
  const onVisibility = () => {
    if (globalThis.document?.hidden) {
      flushOnHide();
    }
  };
  const canListen = subscribePageHide && typeof globalThis.addEventListener === "function";
  if (canListen) {
    globalThis.addEventListener("pagehide", flushOnHide);
    globalThis.addEventListener("visibilitychange", onVisibility);
  }

  return {
    async saveNow() {
      const saved = await saveIfDirty();
      if (!saved && hasRevision) {
        await saveSnapshotToOpfs(engine, fileName);
        lastSavedRevision = engine.revision();
      } else if (!saved && !hasRevision) {
        await saveSnapshotToOpfs(engine, fileName);
        fallbackDirty = false;
      }
    },
    isDirty() {
      if (disposed) {
        return false;
      }
      if (hasRevision) {
        return engine.revision() !== lastSavedRevision;
      }
      return fallbackDirty;
    },
    dispose() {
      disposed = true;
      if (timer !== null) {
        globalThis.clearTimeout(timer);
        timer = null;
      }
      for (const [method, original] of originals) {
        engine[method] = original;
      }
      originals.clear();
      if (canListen) {
        globalThis.removeEventListener("pagehide", flushOnHide);
        globalThis.removeEventListener("visibilitychange", onVisibility);
      }
    },
  };
}

function collectionFileName(name) {
  return `${encodeURIComponent(String(name))}.json`;
}

async function requireOpfsSubdir(root, dirName, create) {
  if (typeof root.getDirectoryHandle !== "function") {
    throw new Error("OPFS directory handles are unavailable in this host");
  }
  return root.getDirectoryHandle(String(dirName), { create });
}

export async function saveCollectionsToOpfs(
  engine,
  dirName = "simvec-collections",
  options = {},
) {
  if (typeof engine.exportCollection !== "function") {
    throw new Error("OPFS sharded save needs engine.exportCollection; use saveSnapshotToOpfs instead");
  }
  const wanted = options.collections ? Array.from(options.collections, (n) => String(n)) : null;
  const names = wanted ?? engine.listCollections();
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS save unavailable, snapshot kept in memory only: ${String(error)}`);
  }
  let subdir;
  try {
    subdir = await requireOpfsSubdir(root, String(dirName), true);
  } catch (error) {
    throw new Error(`OPFS save failed: ${String(error)}`);
  }
  const files = {};
  try {
    for (const name of names) {
      const text = toSnapshotText(engine.exportCollection(name));
      verifyCollectionText(text);
      const file = collectionFileName(name);
      await writeFileAtomic(subdir, file, text, verifyCollectionText);
      files[name] = file;
    }
    const manifest = JSON.stringify({ version: 1, files });
    await writeFileAtomic(subdir, "manifest.json", manifest, verifyManifestText);
  } catch (error) {
    if (String(error?.message ?? error).includes("OPFS ")) {
      throw error;
    }
    throw new Error(`OPFS save failed: ${String(error)}`);
  }
  return Object.keys(files);
}

export async function loadCollectionsFromOpfs(
  engine,
  dirName = "simvec-collections",
  options = {},
) {
  if (typeof engine.importCollection !== "function") {
    throw new Error("OPFS sharded load needs engine.importCollection; use loadSnapshotFromOpfs instead");
  }
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS load unavailable: ${String(error)}`);
  }
  let subdir;
  try {
    subdir = await requireOpfsSubdir(root, String(dirName), false);
  } catch (error) {
    throw new Error(`OPFS snapshot not found: ${String(error)}`);
  }
  let manifestText;
  try {
    const handle = await subdir.getFileHandle("manifest.json", { create: false });
    manifestText = await readHandleText(handle);
  } catch (error) {
    throw new Error(`OPFS snapshot not found: ${String(error)}`);
  }
  const manifest = verifyManifestText(manifestText);
  const wanted = options.collections ? Array.from(options.collections, (n) => String(n)) : null;
  const names = wanted ?? Object.keys(manifest.files);
  const staged = [];
  for (const name of names) {
    const file = manifest.files[name];
    if (typeof file !== "string") {
      throw new Error(`OPFS manifest is corrupt: missing file entry for ${name}`);
    }
    let text;
    try {
      const handle = await subdir.getFileHandle(file, { create: false });
      text = await readHandleText(handle);
    } catch (error) {
      throw new Error(`OPFS snapshot not found: ${String(error)}`);
    }
    staged.push([name, verifyCollectionText(text)]);
  }
  for (const [, parsed] of staged) {
    engine.importCollection(parsed);
  }
  return names;
}
