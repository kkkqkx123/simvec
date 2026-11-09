function requireOpfsDirectory() {
  const storage = globalThis.navigator?.storage;
  if (!storage || typeof storage.getDirectory !== "function") {
    throw new Error("OPFS storage directory is unavailable in this host");
  }
  return storage.getDirectory();
}

export async function saveSnapshotToOpfs(engine, fileName = "simvec-snapshot.json") {
  const snapshot = engine.exportSnapshot();
  const text = typeof snapshot === "string" ? snapshot : JSON.stringify(snapshot);
  let root;
  try {
    root = await requireOpfsDirectory();
  } catch (error) {
    throw new Error(`OPFS save unavailable, snapshot kept in memory only: ${String(error)}`);
  }
  try {
    const handle = await root.getFileHandle(String(fileName), { create: true });
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
  } catch (error) {
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
  let file;
  try {
    const handle = await root.getFileHandle(String(fileName), { create: false });
    file = await handle.getFile();
  } catch (error) {
    throw new Error(`OPFS snapshot not found: ${String(error)}`);
  }
  const text = await file.text();
  engine.importSnapshot(JSON.parse(text));
}
