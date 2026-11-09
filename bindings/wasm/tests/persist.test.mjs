import { test } from "node:test";
import assert from "node:assert/strict";

import {
  saveSnapshotToOpfs,
  loadSnapshotFromOpfs,
  snapshotExists,
  removeSnapshot,
  estimateOpfsQuota,
  requestOpfsPersistence,
  createOpfsAutoSave,
  saveCollectionsToOpfs,
  loadCollectionsFromOpfs,
} from "../src/persist.js";

function createMockFileHandle(dir, name) {
  const handle = {
    name,
    async createWritable() {
      let buffer = "";
      let closed = false;
      return {
        async write(data) {
          buffer += typeof data === "string" ? data : String(data);
        },
        async close() {
          closed = true;
          dir.files.set(name, buffer);
        },
        async abort() {
          closed = true;
        },
      };
    },
    async getFile() {
      if (!dir.files.has(name)) {
        throw new Error("not found");
      }
      const text = dir.files.get(name);
      return { async text() { return text; } };
    },
  };
  handle.move = async (targetName) => {
    if (!dir.files.has(name)) {
      throw new Error("not found");
    }
    dir.files.set(String(targetName), dir.files.get(name));
    dir.files.delete(name);
    handle.name = String(targetName);
  };
  return handle;
}

function createMockDirectory() {
  const dir = {
    files: new Map(),
    dirs: new Map(),
    handles: new Map(),
    async getFileHandle(name, opts = {}) {
      const key = String(name);
      if (!dir.files.has(key) && !opts.create) {
        throw new Error(`not found: ${key}`);
      }
      if (!dir.files.has(key) && opts.create) {
        dir.files.set(key, "");
      }
      if (!dir.handles.has(key)) {
        dir.handles.set(key, createMockFileHandle(dir, key));
      }
      return dir.handles.get(key);
    },
    async getDirectoryHandle(name, opts = {}) {
      const key = String(name);
      if (!dir.dirs.has(key) && !opts.create) {
        throw new Error(`not found: ${key}`);
      }
      if (!dir.dirs.has(key)) {
        dir.dirs.set(key, createMockDirectory());
      }
      return dir.dirs.get(key);
    },
    async removeEntry(name) {
      const key = String(name);
      if (dir.files.has(key)) {
        dir.files.delete(key);
        dir.handles.delete(key);
        return;
      }
      if (dir.dirs.has(key)) {
        dir.dirs.delete(key);
        return;
      }
      throw new Error(`not found: ${key}`);
    },
  };
  return dir;
}

function installMockOpfs(root, extra = {}) {
  const storage = {
    getDirectory: async () => root,
    ...extra,
  };
  const prevNavigator = globalThis.navigator;
  Object.defineProperty(globalThis, "navigator", {
    value: { storage },
    configurable: true,
    writable: true,
  });
  return () => {
    if (prevNavigator === undefined) {
      delete globalThis.navigator;
    } else {
      Object.defineProperty(globalThis, "navigator", {
        value: prevNavigator,
        configurable: true,
        writable: true,
      });
    }
  };
}

function snapshotText(collections = [{ name: "docs", config: { vector_size: 2 }, points: [], payload_indexes: [] }]) {
  return JSON.stringify({ version: 1, collections });
}

function mockEngine(initial = snapshotText()) {
  let stored = JSON.parse(initial);
  let rev = 0;
  return {
    revision() { return rev; },
    listCollections() { return stored.collections.map((c) => c.name); },
    exportSnapshot() { return structuredClone(stored); },
    importSnapshot(next) {
      stored = typeof next === "string" ? JSON.parse(next) : structuredClone(next);
      rev += 1;
    },
    exportCollection(name) {
      const found = stored.collections.find((c) => c.name === name);
      if (!found) throw new Error("missing");
      return structuredClone(found);
    },
    importCollection(entry) {
      const next = typeof entry === "string" ? JSON.parse(entry) : structuredClone(entry);
      stored.collections = stored.collections.filter((c) => c.name !== next.name);
      stored.collections.push(next);
      rev += 1;
    },
    createCollection() { rev += 1; },
    upsert() { rev += 1; },
    _stored() { return stored; },
  };
}

test("single-file save and load roundtrip", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    const engine = mockEngine();
    await saveSnapshotToOpfs(engine);
    assert.equal(await snapshotExists(), true);
    const fresh = mockEngine(snapshotText([]));
    await loadSnapshotFromOpfs(fresh, "simvec-snapshot.json");
    assert.deepEqual(fresh._stored().collections.map((c) => c.name), ["docs"]);
  } finally {
    restore();
  }
});

test("corrupt target file is rejected on load", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    root.files.set("simvec-snapshot.json", "{bad json");
    const engine = mockEngine();
    await assert.rejects(loadSnapshotFromOpfs(engine), /corrupt/);
  } finally {
    restore();
  }
});

test("second save keeps the previous good snapshot as backup", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    const first = mockEngine(snapshotText([{ name: "a", config: {}, points: [], payload_indexes: [] }]));
    await saveSnapshotToOpfs(first);
    const second = mockEngine(snapshotText([{ name: "b", config: {}, points: [], payload_indexes: [] }]));
    await saveSnapshotToOpfs(second);
    assert.equal(root.files.get("simvec-snapshot.json"), snapshotText([{ name: "b", config: {}, points: [], payload_indexes: [] }]));
    assert.equal(root.files.get("simvec-snapshot.json.bak"), snapshotText([{ name: "a", config: {}, points: [], payload_indexes: [] }]));
  } finally {
    restore();
  }
});

test("missing platform interfaces report explicit errors", async () => {
  const prev = globalThis.navigator;
  Object.defineProperty(globalThis, "navigator", { value: undefined, configurable: true, writable: true });
  try {
    const engine = mockEngine();
    await assert.rejects(saveSnapshotToOpfs(engine), /OPFS/);
    await assert.rejects(loadSnapshotFromOpfs(engine), /OPFS/);
    await assert.rejects(snapshotExists(), /OPFS/);
    await assert.rejects(removeSnapshot(), /OPFS/);
    await assert.rejects(estimateOpfsQuota(), /OPFS/);
    await assert.rejects(requestOpfsPersistence(), /OPFS/);
  } finally {
    if (prev === undefined) {
      delete globalThis.navigator;
    } else {
      Object.defineProperty(globalThis, "navigator", { value: prev, configurable: true, writable: true });
    }
  }
});

test("remove snapshot reports whether anything was removed", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    assert.equal(await removeSnapshot(), false);
    const engine = mockEngine();
    await saveSnapshotToOpfs(engine);
    assert.equal(await removeSnapshot(), true);
    assert.equal(await snapshotExists(), false);
  } finally {
    restore();
  }
});

test("auto save tracks revision and debounces writes", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    const engine = mockEngine();
    const seen = [];
    const auto = createOpfsAutoSave(engine, {
      debounceMs: 5,
      subscribePageHide: false,
      onError: (e) => seen.push(e),
    });
    assert.equal(auto.isDirty(), false);
    engine.upsert();
    assert.equal(auto.isDirty(), true);
    await auto.saveNow();
    assert.equal(auto.isDirty(), false);
    assert.equal(await snapshotExists(), true);
    assert.deepEqual(seen, []);
    auto.dispose();
    assert.equal(auto.isDirty(), false);
  } finally {
    restore();
  }
});

test("sharded save and load roundtrip", async () => {
  const root = createMockDirectory();
  const restore = installMockOpfs(root);
  try {
    const engine = mockEngine(snapshotText([
      { name: "a", config: { vector_size: 2 }, points: [{ id: "p1", vector: [1, 0] }], payload_indexes: [] },
      { name: "b", config: { vector_size: 2 }, points: [], payload_indexes: [] },
    ]));
    const saved = await saveCollectionsToOpfs(engine, "shard");
    assert.deepEqual(saved.sort(), ["a", "b"]);
    const fresh = mockEngine(snapshotText([]));
    const loaded = await loadCollectionsFromOpfs(fresh, "shard");
    assert.deepEqual(loaded.sort(), ["a", "b"]);
    assert.equal(fresh._stored().collections.length, 2);
    const partial = mockEngine(snapshotText([]));
    const names = await loadCollectionsFromOpfs(partial, "shard", { collections: ["a"] });
    assert.deepEqual(names, ["a"]);
    assert.equal(partial._stored().collections.length, 1);
  } finally {
    restore();
  }
});
